import json
import re
import unicodedata
from pathlib import Path

INPUT_PATH = Path("data/docs.jsonl")
OUTPUT_PATH = Path("data/docs_ru.jsonl")

MIN_CHARS = 500          # минимальная длина документа после фильтра
MIN_KEEP_RATIO = 0.5     # сколько текста должно остаться (иначе статья «слишком иностранная»)

# Разрешённые символы. Хотите больше (украинские і ї є ґ и т.д.) - добавляйте сюда.
BAD_CHAR = re.compile(r"[^а-яёА-ЯЁ0-9\s.,:;!?\-–—«»\"'()%№°/+=×€$₽]")
FOREIGN_LETTER = re.compile(r"[^\W\d_а-яёА-ЯЁ]")   # любая буква, кроме русской
RU_LETTER = re.compile(r"[а-яёА-ЯЁ]")

ROUND_BRACKETS = re.compile(r" ?\(([^()]*)\)")
SQUARE_BRACKETS = re.compile(r" ?\[([^\[\]]*)\]")
EMPTY_BRACKETS = re.compile(r" ?\(\s*[,;:]?\s*\)")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

TRANSLATE = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‘": "'", "’": "'", "‚": "'",
    "−": "-", "‐": "-", "‑": "-", "‒": "-",
    "­": None,   # мягкий перенос
    "́": None,   # знак ударения
    "̀": None,
})


def _drop_round(m):
    return "" if FOREIGN_LETTER.search(m.group(1)) else m.group(0)


def _drop_square(m):
    inner = m.group(1)
    # сноски [1], транскрипции [ˈslɔvə], [sic] - всё, где нет русских слов или есть чужие буквы
    if FOREIGN_LETTER.search(inner) or not RU_LETTER.search(inner):
        return ""
    return m.group(0)


def strip_foreign_brackets(text):
    for _ in range(3):  # несколько проходов для вложенных скобок
        new = ROUND_BRACKETS.sub(_drop_round, text)
        new = SQUARE_BRACKETS.sub(_drop_square, new)
        new = EMPTY_BRACKETS.sub("", new)
        if new == text:
            break
        text = new
    return text


def filter_line(line):
    sentences = SENTENCE_SPLIT.split(line)
    kept = [s for s in sentences if not BAD_CHAR.search(s)]
    line = " ".join(kept).strip()
    # строки совсем без русских букв (остатки таблиц, годы и т.п.) не нужны
    return line if RU_LETTER.search(line) else ""


def filter_text(text):
    """Возвращает (очищенный_текст, доля_сохранённого_после_удаления_скобок)."""
    text = unicodedata.normalize("NFKC", text).translate(TRANSLATE)
    text = strip_foreign_brackets(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    before = len(text)
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            out.append("")
            continue
        cleaned = filter_line(line)
        if cleaned:
            out.append(cleaned)

    result = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return result, len(result) / max(before, 1)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Нет файла {INPUT_PATH}")

    total = kept = 0
    chars_in = chars_out = 0

    with INPUT_PATH.open(encoding="utf-8") as src, OUTPUT_PATH.open("w", encoding="utf-8") as dst:
        for line in src:
            total += 1
            text = json.loads(line)["text"]
            chars_in += len(text)

            cleaned, ratio = filter_text(text)
            if len(cleaned) < MIN_CHARS or ratio < MIN_KEEP_RATIO:
                continue

            dst.write(json.dumps({"text": cleaned}, ensure_ascii=False) + "\n")
            kept += 1
            chars_out += len(cleaned)

            if total % 5000 == 0:
                print(f"Просмотрено {total} | оставлено {kept} | текста {chars_out / chars_in:.0%} от исходного")

    print("\n===== ГОТОВО =====")
    print(f"Документов: {total} -> {kept} ({kept / max(total, 1):.0%})")
    print(f"Символов:   {chars_in / 1e6:.0f}M -> {chars_out / 1e6:.0f}M ({chars_out / max(chars_in, 1):.0%})")
    print(f"Файл: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()