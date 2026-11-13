
import json
import re
import unicodedata

from pathlib import Path
from datasets import load_dataset


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = Path("data")

# Wikipedia
WIKI_DATASET = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.ru"

WIKI_OUTPUT = DATA_DIR / "docs_ru.jsonl"

# FineWeb-2
FINEWEB_DATASET = "HuggingFaceFW/fineweb-2"
FINEWEB_CONFIG = "rus_Cyrl"

FINEWEB_OUTPUT = DATA_DIR / "fineweb_ru.jsonl"

# Combined dataset
MIXED_OUTPUT = DATA_DIR / "mixed_docs.jsonl"

# Объём загрузки в символах.
# Это ориентиры, а не точное количество токенов.
TARGET_WIKI_CHARS = 400_000_000
TARGET_FINEWEB_CHARS = 1_400_000_000

# Shuffle
SHUFFLE_BUFFER = 20_000
SEED = 42

# Filter
MIN_CHARS = 500
MIN_KEEP_RATIO = 0.5

# Пропорция при объединении документов
FINEWEB_PER_WIKI = 4


# ============================================================
# REGEX
# ============================================================

BAD_CHAR = re.compile(
    r"""[^а-яёА-ЯЁ0-9\s.,:;!?\-–—«»"'()%№°/+=×€$₽]"""
)

FOREIGN_LETTER = re.compile(
    r"[^\W\d_а-яёА-ЯЁ]"
)

RU_LETTER = re.compile(
    r"[а-яёА-ЯЁ]"
)

ROUND_BRACKETS = re.compile(
    r" ?\(([^()]*)\)"
)

SQUARE_BRACKETS = re.compile(
    r" ?\[([^\[\]]*)\]"
)

EMPTY_BRACKETS = re.compile(
    r" ?\(\s*[,;:]?\s*\)"
)

SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+"
)

CATEGORY_LINE = re.compile(
    r"^(категори[яи]|см\. также|шаблон|портал)\s*:",
    re.IGNORECASE
)

INITIALS = re.compile(
    r"\b[А-ЯЁ]\."
)


# ============================================================
# NORMALIZATION
# ============================================================

TRANSLATE = str.maketrans({
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',

    "‘": "'",
    "’": "'",
    "‚": "'",

    "−": "-",
    "‐": "-",
    "‑": "-",
    "‒": "-",

    "\u00ad": None,
    "\u0301": None,
    "\u0300": None,
})


# ============================================================
# BRACKETS FILTER
# ============================================================

def drop_round(match):

    inner = match.group(1)

    if FOREIGN_LETTER.search(inner):
        return ""

    return match.group(0)


def drop_square(match):

    inner = match.group(1)

    if FOREIGN_LETTER.search(inner) or not RU_LETTER.search(inner):
        return ""

    return match.group(0)


def strip_foreign_brackets(text):

    for _ in range(3):

        new = ROUND_BRACKETS.sub(drop_round, text)

        new = SQUARE_BRACKETS.sub(drop_square, new)

        new = EMPTY_BRACKETS.sub("", new)

        if new == text:
            break

        text = new

    return text


# ============================================================
# LINE FILTER
# ============================================================

def is_junk_line(line):

    if CATEGORY_LINE.match(line):
        return True

    no_space = len(line.replace(" ", "")) or 1

    digits = sum(c.isdigit() for c in line)

    if digits / no_space > 0.2:
        return True

    if len(INITIALS.findall(line)) >= 6:
        return True

    if line.count("«") >= 6:
        return True

    if line.count(" — ") >= 6:
        return True

    return False


def filter_line(line):

    if is_junk_line(line):
        return ""

    sentences = SENTENCE_SPLIT.split(line)

    kept = [
        sentence
        for sentence in sentences
        if not BAD_CHAR.search(sentence)
    ]

    line = " ".join(kept).strip()

    return line if RU_LETTER.search(line) else ""


# ============================================================
# DOCUMENT FILTER
# ============================================================

def filter_text(text):

    text = unicodedata.normalize(
        "NFKC",
        text
    ).translate(TRANSLATE)

    # Исходная длина до удаления скобок
    before = len(text)

    text = strip_foreign_brackets(text)

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text
    )

    output = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        cleaned = filter_line(line)

        if cleaned:
            output.append(cleaned)

    result = re.sub(
        r"\n{3,}",
        "\n\n",
        "\n".join(output)
    ).strip()

    ratio = len(result) / max(before, 1)

    return result, ratio


def clean_document(text):

    if not isinstance(text, str):
        return None

    cleaned, ratio = filter_text(text)

    if len(cleaned) < MIN_CHARS:
        return None

    if ratio < MIN_KEEP_RATIO:
        return None

    return cleaned


# ============================================================
# DOWNLOAD AND CLEAN DATASET
# ============================================================

def download_dataset(
    dataset_name,
    config,
    output_path,
    target_chars,
    source_name
):

    print("\n" + "=" * 60)
    print(f"DOWNLOADING: {source_name}")
    print("=" * 60)

    print("Dataset:", dataset_name)
    print("Config:", config)

    dataset = load_dataset(
        dataset_name,
        name=config,
        split="train",
        streaming=True
    )

    dataset = dataset.shuffle(
        seed=SEED,
        buffer_size=SHUFFLE_BUFFER
    )

    total_docs = 0
    kept_docs = 0

    chars_in = 0
    chars_out = 0

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as output:

        for row in dataset:

            total_docs += 1

            text = row.get("text", "")

            if not isinstance(text, str):
                continue

            chars_in += len(text)

            cleaned = clean_document(text)

            if cleaned is None:
                continue

            output.write(
                json.dumps(
                    {
                        "text": cleaned,
                        "source": source_name
                    },
                    ensure_ascii=False
                ) + "\n"
            )

            kept_docs += 1
            chars_out += len(cleaned)

            if kept_docs % 1000 == 0:

                print(
                    f"{source_name} | "
                    f"Просмотрено: {total_docs:,} | "
                    f"Оставлено: {kept_docs:,} | "
                    f"Символов: {chars_out:,} / "
                    f"{target_chars:,}"
                )

            if chars_out >= target_chars:
                break

    print(f"\n{source_name} COMPLETE")

    print(f"Документов: {total_docs:,}")
    print(f"Оставлено: {kept_docs:,}")

    print(f"Исходных символов: {chars_in:,}")
    print(f"Очищенных символов: {chars_out:,}")

    print("Файл:", output_path)

    return {
        "documents": kept_docs,
        "characters": chars_out
    }


# ============================================================
# MIX DATASETS
# ============================================================

def read_jsonl(path):

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            try:
                yield json.loads(line)

            except json.JSONDecodeError:
                continue


def create_mixed_dataset():

    print("\n" + "=" * 60)
    print("CREATING MIXED DATASET")
    print("=" * 60)

    fineweb = iter(read_jsonl(FINEWEB_OUTPUT))
    wiki = iter(read_jsonl(WIKI_OUTPUT))

    fineweb_active = True
    wiki_active = True

    total = 0
    fineweb_count = 0
    wiki_count = 0

    with MIXED_OUTPUT.open(
        "w",
        encoding="utf-8"
    ) as output:

        while fineweb_active or wiki_active:

            # 4 документа FineWeb

            for _ in range(FINEWEB_PER_WIKI):

                if not fineweb_active:
                    break

                try:
                    doc = next(fineweb)

                except StopIteration:
                    fineweb_active = False
                    break

                output.write(
                    json.dumps(doc, ensure_ascii=False) + "\n"
                )

                fineweb_count += 1
                total += 1

            # 1 документ Wikipedia

            if wiki_active:

                try:
                    doc = next(wiki)

                except StopIteration:
                    wiki_active = False
                    continue

                output.write(
                    json.dumps(doc, ensure_ascii=False) + "\n"
                )

                wiki_count += 1
                total += 1

            if total % 10000 == 0:

                print(
                    f"Документов: {total:,} | "
                    f"FineWeb: {fineweb_count:,} | "
                    f"Wiki: {wiki_count:,}"
                )

    print("\nMIXED DATASET COMPLETE")

    print("Всего документов:", total)
    print("FineWeb:", fineweb_count)
    print("Wikipedia:", wiki_count)
    print("Файл:", MIXED_OUTPUT)


# ============================================================
# MAIN
# ============================================================

def main():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 1. Скачать и очистить FineWeb

    download_dataset(
        dataset_name=FINEWEB_DATASET,
        config=FINEWEB_CONFIG,
        output_path=FINEWEB_OUTPUT,
        target_chars=TARGET_FINEWEB_CHARS,
        source_name="fineweb"
    )

    # 2. Скачать и очистить Wikipedia

    download_dataset(
        dataset_name=WIKI_DATASET,
        config=WIKI_CONFIG,
        output_path=WIKI_OUTPUT,
        target_chars=TARGET_WIKI_CHARS,
        source_name="wiki"
    )

    # 3. Объединить

    create_mixed_dataset()

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)

    print("FineWeb:", FINEWEB_OUTPUT)
    print("Wikipedia:", WIKI_OUTPUT)
    print("Mixed:", MIXED_OUTPUT)


if __name__ == "__main__":
    main()
