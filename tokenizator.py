
import json
import numpy as np

from pathlib import Path
from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    normalizers,
    decoders
)

from config import CONFIG


# ============================================================
# SETTINGS
# ============================================================

FINEWEB_PATH = Path("data/fineweb_ru.jsonl")
WIKI_PATH = Path("data/docs_ru.jsonl")

TOKENIZER_PATH = Path("tokenizer/mixed_tokenizer.json")

TRAIN_BIN = Path("data/mixed_train.bin")
VAL_BIN = Path("data/mixed_val.bin")

VOCAB_SIZE = CONFIG["VOCAB_SIZE"]

# Общий лимит — 150 млн токенов
MAX_TOKENS = {
    "fineweb": 240_000_000,
    "wiki": 60_000_000
}

TOKENIZER_SAMPLE_CHARS = 400_000_000

CHUNK_DOCS = 1000
VAL_EVERY = 200

SPECIAL_TOKENS = [
    "[UNK]",
    "[PAD]",
    "[BOS]",
    "[EOS]",
    "[USER]",
    "[ASSISTANT]"
]


# ============================================================
# DOCUMENT ITERATORS
# ============================================================

def iter_file(path):

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            try:
                doc = json.loads(line)
                text = doc.get("text", "").strip()

                if len(text) >= 500:
                    yield text

            except (json.JSONDecodeError, AttributeError):
                continue


def iter_mixed_docs():

    fineweb = iter(iter_file(FINEWEB_PATH))
    wiki = iter(iter_file(WIKI_PATH))

    fineweb_active = True
    wiki_active = True

    while fineweb_active or wiki_active:

        # Приблизительно 4 документа FineWeb на 1 документ Wiki

        for _ in range(4):

            if not fineweb_active:
                break

            try:
                yield "fineweb", next(fineweb)

            except StopIteration:
                fineweb_active = False

        if wiki_active:

            try:
                yield "wiki", next(wiki)

            except StopIteration:
                wiki_active = False


# ============================================================
# TRAIN TOKENIZER
# ============================================================

def train_tokenizer():

    print("\nTraining mixed BPE tokenizer...")

    tokenizer = Tokenizer(
        models.BPE(unk_token="[UNK]")
    )

    tokenizer.normalizer = normalizers.NFKC()

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    def tokenizer_iterator():

        total_chars = 0

        for source, text in iter_mixed_docs():

            if total_chars >= TOKENIZER_SAMPLE_CHARS:
                break

            remaining = TOKENIZER_SAMPLE_CHARS - total_chars

            if len(text) > remaining:
                text = text[:remaining]

            total_chars += len(text)

            yield text

        print(f"\nTokenizer sample: {total_chars:,} characters")

    tokenizer.train_from_iterator(
        tokenizer_iterator(),
        trainer=trainer
    )

    TOKENIZER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer.save(str(TOKENIZER_PATH))

    print("Tokenizer saved:", TOKENIZER_PATH)

    return tokenizer


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer():

    if TOKENIZER_PATH.exists():

        print("Loading existing mixed tokenizer...")

        return Tokenizer.from_file(
            str(TOKENIZER_PATH)
        )

    return train_tokenizer()


# ============================================================
# TOKENIZE DATASET
# ============================================================

def tokenize_corpus(tokenizer):

    eos_id = tokenizer.token_to_id("[EOS]")

    if eos_id is None:
        raise ValueError("[EOS] not found")

    if tokenizer.get_vocab_size() > 65535:
        raise ValueError("Vocabulary is too large for uint16")

    TRAIN_BIN.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    written = {
        "fineweb": 0,
        "wiki": 0
    }

    train_tokens = 0
    val_tokens = 0
    total_tokens = 0
    total_docs = 0

    buffer = []

    with TRAIN_BIN.open("wb") as f_train, \
         VAL_BIN.open("wb") as f_val:

        def flush():

            nonlocal train_tokens
            nonlocal val_tokens
            nonlocal total_tokens
            nonlocal total_docs

            if not buffer:
                return

            texts = [item[1] for item in buffer]

            encoded = tokenizer.encode_batch(texts)

            for (source, text), enc in zip(buffer, encoded):

                source_limit = MAX_TOKENS[source]
                remaining = source_limit - written[source]

                if remaining <= 0:
                    continue

                ids = enc.ids + [eos_id]

                if len(ids) > remaining:

                    ids = ids[:remaining]

                    if ids:
                        ids[-1] = eos_id

                if not ids:
                    continue

                arr = np.asarray(
                    ids,
                    dtype=np.uint16
                )

                # Каждый 200-й документ — validation
                if total_docs % VAL_EVERY == 0:

                    f_val.write(arr.tobytes())
                    val_tokens += len(arr)

                else:

                    f_train.write(arr.tobytes())
                    train_tokens += len(arr)

                written[source] += len(arr)
                total_tokens += len(arr)
                total_docs += 1

            buffer.clear()

        for source, text in iter_mixed_docs():

            if all(
                written[name] >= MAX_TOKENS[name]
                for name in MAX_TOKENS
            ):
                break

            if written[source] >= MAX_TOKENS[source]:
                continue

            buffer.append((source, text))

            if len(buffer) >= CHUNK_DOCS:

                flush()

                print(
                    f"Documents: {total_docs:,} | "
                    f"FineWeb: {written['fineweb']:,} | "
                    f"Wiki: {written['wiki']:,} | "
                    f"Total: {total_tokens:,} / {MAX_TOKENS["fineweb"] + MAX_TOKENS["wiki"]}"
                )

        flush()

    print("\n========== COMPLETE ==========")

    print(f"FineWeb tokens: {written['fineweb']:,}")
    print(f"Wiki tokens:    {written['wiki']:,}")

    print(f"\nTrain tokens:   {train_tokens:,}")
    print(f"Val tokens:     {val_tokens:,}")
    print(f"Total tokens:   {total_tokens:,}")

    print("\nFile sizes:")

    print(
        f"Train: {TRAIN_BIN.stat().st_size / 1024**2:.2f} MB"
    )

    print(
        f"Val:   {VAL_BIN.stat().st_size / 1024**2:.2f} MB"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    for path in [FINEWEB_PATH, WIKI_PATH]:

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

    print("Mixed dataset: FineWeb + Wikipedia")

    print("FineWeb target: 120M tokens")
    print("Wikipedia target: 30M tokens")
    print("Total target: 150M tokens")

    tokenizer = load_tokenizer()

    tokenize_corpus(tokenizer)


if __name__ == "__main__":
    main()
