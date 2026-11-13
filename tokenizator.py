"""
Подготовка данных:
1. Обучает BPE-токенизатор на data/docs.jsonl (если его ещё нет)
2. Токенизирует документы, ставит [EOS] после каждого
3. Пишет data/train.bin и data/val.bin (uint16)
   Val выбирается по документам: каждый VAL_EVERY-й документ.

Запуск: python prepare_data.py
"""
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, normalizers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from config import CONFIG

VOCAB_SIZE = CONFIG["VOCAB_SIZE"]  # должно быть 16000

DOCS_PATH = Path("data/docs_ru.jsonl")
TOKENIZER_PATH = Path("tokenizer/tokenizer.json")
TRAIN_BIN = Path("data/train.bin")
VAL_BIN = Path("data/val.bin")

SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[USER]", "[ASSISTANT]"]
TOKENIZER_SAMPLE_CHARS = 400_000_000  # сколько символов отдать на обучение токенизатора
VAL_EVERY = 200                       # каждый 200-й документ уходит в val
CHUNK_DOCS = 1000                     # документов за один вызов encode_batch


def iter_docs():
    with DOCS_PATH.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def limited_docs(max_chars):
    total = 0
    for doc in iter_docs():
        yield doc
        total += len(doc)
        if total >= max_chars:
            break


def train_tokenizer():
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    print("Обучаем токенизатор...")
    tok.train_from_iterator(limited_docs(TOKENIZER_SAMPLE_CHARS), trainer=trainer)

    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(TOKENIZER_PATH))
    print("Токенизатор сохранён:", TOKENIZER_PATH, "| словарь:", tok.get_vocab_size())
    return tok


def load_or_train_tokenizer():
    if TOKENIZER_PATH.exists():
        print("Найден готовый токенизатор:", TOKENIZER_PATH)
        return Tokenizer.from_file(str(TOKENIZER_PATH))
    return train_tokenizer()


def check_tokenizer(tok):
    assert tok.get_vocab_size() <= 65535, "Словарь не помещается в uint16"
    ids = {t: tok.token_to_id(t) for t in SPECIAL_TOKENS}
    print("ID спецтокенов:", ids)
    assert ids["[EOS]"] == 3, "Ожидался [EOS] = 3"

    sample = "Привет! Я обучаю собственную языковую модель."
    enc = tok.encode(sample)
    print("Тест:", sample)
    print("Токенов:", len(enc.ids), "| восстановлено верно:",
          tok.decode(enc.ids) == sample)


def tokenize_corpus(tok):
    eos_id = tok.token_to_id("[EOS]")

    TRAIN_BIN.parent.mkdir(parents=True, exist_ok=True)
    n_train = n_val = n_docs = total_chars = 0

    with TRAIN_BIN.open("wb") as f_train, VAL_BIN.open("wb") as f_val:
        buffer = []

        def flush(buf, start_index):
            nonlocal n_train, n_val, total_chars
            encoded = tok.encode_batch(buf)
            for k, enc in enumerate(encoded):
                ids = enc.ids + [eos_id]
                arr = np.asarray(ids, dtype=np.uint16)
                total_chars += len(buf[k])
                if (start_index + k) % VAL_EVERY == 0:
                    f_val.write(arr.tobytes())
                    n_val += len(arr)
                else:
                    f_train.write(arr.tobytes())
                    n_train += len(arr)

        for doc in iter_docs():
            buffer.append(doc)
            if len(buffer) == CHUNK_DOCS:
                flush(buffer, n_docs)
                n_docs += len(buffer)
                buffer = []
                if (n_docs // CHUNK_DOCS) % 10 == 0:
                    print(f"Документов: {n_docs} | train токенов: {n_train:,} | val токенов: {n_val:,}")

        if buffer:
            flush(buffer, n_docs)
            n_docs += len(buffer)

    print("\n===== ГОТОВО =====")
    print(f"Документов:      {n_docs}")
    print(f"Train токенов:   {n_train:,}")
    print(f"Val токенов:     {n_val:,}")
    print(f"Символов/токен:  {total_chars / max(n_train + n_val, 1):.2f}")
    print(f"Файлы: {TRAIN_BIN} ({TRAIN_BIN.stat().st_size / 1024**2:.0f} MB), "
          f"{VAL_BIN} ({VAL_BIN.stat().st_size / 1024**2:.0f} MB)")

    if n_train < 300_000_000:
        print("\nВнимание: токенов меньше 300M. Для модели ~20M параметров "
              "добавьте второй источник данных или уменьшите модель/MAX_STEPS.")


def main():
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"Нет файла {DOCS_PATH}. Сначала запустите скрипт загрузки и очистки.")

    tok = load_or_train_tokenizer()
    check_tokenizer(tok)
    tokenize_corpus(tok)


if __name__ == "__main__":
    main()