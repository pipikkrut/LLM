import contextlib
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizers import Tokenizer

from config import CONFIG

# ---------- конфиг ----------
VOCAB_SIZE = CONFIG["VOCAB_SIZE"]
EMBEDDING_DIM = CONFIG["EMBEDDING_DIM"]
N_HEAD = CONFIG["N_HEAD"]
N_LAYERS = CONFIG["N_LAYERS"]
BLOCK_SIZE = CONFIG["BLOCK_SIZE"]
DROPOUT = CONFIG["DROPOUT"]

BATCH_SIZE = CONFIG["BATCH_SIZE"]
GRAD_ACCUM = CONFIG["GRAD_ACCUM"]
MAX_STEPS = CONFIG["MAX_STEPS"]
WARMUP_STEPS = CONFIG["WARMUP_STEPS"]
LEARNING_RATE = CONFIG["LEARNING_RATE"]
MIN_LR_RATIO = CONFIG["MIN_LR_RATIO"]
WEIGHT_DECAY = CONFIG["WEIGHT_DECAY"]
EVAL_INTERVAL = CONFIG["EVAL_INTERVAL"]
EVAL_ITERS = CONFIG["EVAL_ITERS"]
LOG_INTERVAL = CONFIG["LOG_INTERVAL"]
COMPILE = CONFIG.get("COMPILE", False)  # torch.compile: ускоряет, но первые шаги долгие

HEAD_DIM = EMBEDDING_DIM // N_HEAD
assert EMBEDDING_DIM % N_HEAD == 0
FFN_DIM = 4 * EMBEDDING_DIM

TOKENIZER_PATH = Path("tokenizer/tokenizer.json")
TRAIN_BIN = Path("data/train.bin")
VAL_BIN = Path("data/val.bin")
CKPT_DIR = Path(CONFIG["CKPT_DIR"])
LAST_CKPT = CKPT_DIR / "ckpt_last.pt"   # веса + оптимизатор + шаг (для продолжения)
BEST_CKPT = CKPT_DIR / "model_best.pt"  # только веса лучшей модели по val loss

# ---------- несколько GPU (запуск через torchrun) ----------
DDP_ON = "RANK" in os.environ
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_MAIN = RANK == 0

if torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
    torch.cuda.set_device(DEVICE)
else:
    DEVICE = torch.device("cpu")
USE_AMP = DEVICE.type == "cuda"
EOS_ID = 3  # проверяется в prepare_data.py


# ---------- данные ----------
_data_cache = {}


def get_batch(split):
    path = TRAIN_BIN if split == "train" else VAL_BIN
    if split not in _data_cache:
        _data_cache[split] = np.memmap(path, dtype=np.uint16, mode="r")
    data = _data_cache[split]

    ix = np.random.randint(0, len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    idx = ix[:, None] + np.arange(BLOCK_SIZE)[None, :]   # все окна за одну операцию
    x = torch.from_numpy(data[idx].astype(np.int64))
    y = torch.from_numpy(data[idx + 1].astype(np.int64))
    return x.to(DEVICE), y.to(DEVICE)


# ---------- модель ----------
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBEDDING_DIM)
        self.ln2 = nn.LayerNorm(EMBEDDING_DIM)
        self.qkv = nn.Linear(EMBEDDING_DIM, 3 * EMBEDDING_DIM, bias=False)
        self.proj = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
        self.ffn_in = nn.Linear(EMBEDDING_DIM, FFN_DIM, bias=False)
        self.ffn_out = nn.Linear(FFN_DIM, EMBEDDING_DIM, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        k = k.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)

        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=DROPOUT if self.training else 0.0,
        )
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.dropout(self.proj(a))

        x = x + self.dropout(self.ffn_out(F.gelu(self.ffn_in(self.ln2(x)))))
        return x


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.positional_embedding = nn.Embedding(BLOCK_SIZE, EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.final_ln = nn.LayerNorm(EMBEDDING_DIM)
        self.lm_head = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE, bias=False)

        self.lm_head.weight = self.embedding.weight  # weight tying

        self.apply(self._init_weights)
        # выходные проекции остаточных веток масштабируем (как в GPT-2)
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=0.02 / math.sqrt(2 * N_LAYERS))
            nn.init.normal_(block.ffn_out.weight, std=0.02 / math.sqrt(2 * N_LAYERS))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x, targets=None):
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        h = self.dropout(self.embedding(x) + self.positional_embedding(pos))
        for block in self.blocks:
            h = block(h)
        logits = self.lm_head(self.final_ln(h))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=200, temperature=0.8, top_k=40, top_p=0.95,
                 repetition_penalty=1.2):
        self.eval()
        for _ in range(max_new_tokens):
            window = ids[:, -BLOCK_SIZE:]
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                logits, _ = self(window)
            logits = logits[:, -1, :].float()

            # штраф за повторы: токены из последних 128 становятся менее вероятными
            if repetition_penalty != 1.0:
                recent = torch.unique(ids[0, -128:])
                vals = logits[0, recent]
                logits[0, recent] = torch.where(vals > 0, vals / repetition_penalty,
                                                vals * repetition_penalty)

            logits = logits / max(temperature, 1e-5)

            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = probs.cumsum(dim=-1)
                remove = cum - probs > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

            next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == EOS_ID:
                break
        return ids


# ---------- обучение ----------
def build_optimizer(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and "embedding" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LEARNING_RATE, betas=(0.9, 0.95),
        fused=USE_AMP,  # быстрая версия AdamW для GPU
    )


def get_lr(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return LEARNING_RATE * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * cosine)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(EVAL_ITERS):
            x, y = get_batch(split)
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                _, loss = model(x, y)
            total += loss.item()
        out[split] = total / EVAL_ITERS
    model.train()
    return out


def save_checkpoints(model, optimizer, scaler, step, best_val, is_best):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LAST_CKPT.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "best_val": best_val,
    }, tmp)
    tmp.replace(LAST_CKPT)  # атомарная замена, чтобы файл не побился при обрыве
    if is_best:
        torch.save(model.state_dict(), BEST_CKPT)


def reduce_mean(value):
    """Среднее значение по всем GPU (чтобы val loss был одинаковым на всех процессах)."""
    if not DDP_ON:
        return value
    t = torch.tensor(value, device=DEVICE)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / WORLD_SIZE).item()


def train():
    if DDP_ON:
        dist.init_process_group(backend="nccl" if USE_AMP else "gloo")
    torch.manual_seed(1337)
    np.random.seed(1337 + RANK)  # у каждой GPU свои случайные окна данных

    raw_model = Network().to(DEVICE)
    optimizer = build_optimizer(raw_model)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    assert GRAD_ACCUM % WORLD_SIZE == 0, "GRAD_ACCUM должен делиться на число GPU"
    accum = GRAD_ACCUM // WORLD_SIZE                          # микрошагов на одну GPU
    tokens_per_step = BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM    # токенов за шаг суммарно по всем GPU

    if IS_MAIN:
        n_params = sum(p.numel() for p in raw_model.parameters())
        print(f"Параметров: {n_params / 1e6:.1f}M | GPU: {WORLD_SIZE} | токенов за шаг: {tokens_per_step:,} "
              f"| всего токенов: {tokens_per_step * MAX_STEPS / 1e6:.0f}M | устройство: {DEVICE}")

    step, best_val = 0, float("inf")
    if LAST_CKPT.exists():
        ckpt = torch.load(LAST_CKPT, map_location=DEVICE, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        step, best_val = ckpt["step"], ckpt["best_val"]
        if IS_MAIN:
            print(f"Продолжаем с шага {step}, лучший val loss: {best_val:.4f}")

    model = DDP(raw_model, device_ids=[LOCAL_RANK] if USE_AMP else None) if DDP_ON else raw_model
    if COMPILE:
        model = torch.compile(model)  # оценка и генерация идут через raw_model, поэтому не перекомпилируются

    model.train()
    t0 = time.time()
    tokens_since = 0
    running_loss = torch.zeros((), device=DEVICE)  # копим лосс на GPU, без синхронизации на каждом шаге

    while step < MAX_STEPS:
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for micro in range(accum):
            x, y = get_batch("train")
            # градиенты между GPU синхронизируем только на последнем микрошаге
            skip_sync = DDP_ON and micro < accum - 1
            with (model.no_sync() if skip_sync else contextlib.nullcontext()):
                with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                    _, loss = model(x, y)
                scaler.scale(loss / accum).backward()
            running_loss += loss.detach() / accum

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        tokens_since += tokens_per_step

        if step % LOG_INTERVAL == 0:
            dt = time.time() - t0
            speed = tokens_since / dt
            eta_min = (MAX_STEPS - step) * tokens_per_step / speed / 60
            if IS_MAIN:
                print(f"шаг {step}/{MAX_STEPS} | loss {running_loss.item() / LOG_INTERVAL:.4f} "
                      f"| lr {lr:.2e} | {speed:,.0f} ток/с | осталось ~{eta_min:.0f} мин")
            running_loss.zero_()
            tokens_since, t0 = 0, time.time()

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            losses = {k: reduce_mean(v) for k, v in estimate_loss(raw_model).items()}
            is_best = losses["val"] < best_val
            best_val = min(best_val, losses["val"])

            if IS_MAIN:
                print(f"\n=== шаг {step} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                      f"| ppl {math.exp(losses['val']):.1f}{' | новый лучший' if is_best else ''} ===")
                save_checkpoints(raw_model, optimizer, scaler, step, best_val, is_best)

                for prompt in ("Москва — это", "Привет! Как", "В 1990 году"):  # без пробела в конце
                    sample = generate_text(raw_model, prompt, max_new_tokens=60)
                    print("Пример:", sample.replace("\n", " "))
                print()

            if DDP_ON:
                dist.barrier()  # остальные GPU ждут, пока главный процесс сохранит чекпоинт
            model.train()
            t0 = time.time()
            tokens_since = 0

    if IS_MAIN:
        print("Обучение завершено.")
    if DDP_ON:
        dist.destroy_process_group()


# ---------- генерация ----------
def generate_text(model, prompt, max_new_tokens=200, temperature=0.8, top_k=40, top_p=0.95,
                  repetition_penalty=1.2):
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=DEVICE)
    out = model.generate(ids, max_new_tokens, temperature, top_k, top_p, repetition_penalty)
    return tokenizer.decode(out[0].tolist())


def run_generate(prompt):
    model = Network().to(DEVICE)
    model.load_state_dict(torch.load(BEST_CKPT, map_location=DEVICE, weights_only=True))
    print(generate_text(model, prompt, max_new_tokens=150, temperature=0.8))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "generate"
    if mode == "train":
        train()
    elif mode == "generate":
        run_generate(sys.argv[2] if len(sys.argv) > 2 else "Привет")
    else:
        print("Использование: python main.py train | python main.py generate \"текст\"")