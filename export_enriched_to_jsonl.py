#!/usr/bin/env python3
"""
Экспорт обогащённого датасета (save_to_disk) в несколько файлов.
Сейчас: первый файл — qa_pairs.jsonl с полями question_id, question, answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, TextIO


def _write_qa_pairs_jsonl(rows: Any, out: TextIO) -> int:
    n = 0
    for ex in rows:
        rec: Dict[str, Any] = {
            "question_id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default="nq_open_validation_enriched",
        help="Каталог датасета (load_from_disk).",
    )
    parser.add_argument(
        "--qa-pairs",
        type=str,
        default="qa_pairs.jsonl",
        help="Путь к первому выходному файлу (jsonl).",
    )
    args = parser.parse_args()

    from datasets import load_from_disk

    print(f"Загрузка: {args.input}", file=sys.stderr)
    ds = load_from_disk(args.input)
    required = {"id", "question", "answer"}
    missing = required - set(ds.column_names)
    if missing:
        print(f"Ошибка: в датасете нет колонок: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    print(f"Запись: {args.qa_pairs}", file=sys.stderr)
    with open(args.qa_pairs, "w", encoding="utf-8") as f:
        n = _write_qa_pairs_jsonl(ds, f)
    print(f"Готово, строк: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
