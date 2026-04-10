#!/usr/bin/env python3
"""
Экспорт documents_collection.jsonl из обогащённого датасета.

Каждая строка — один пример:
- question_id: из id
- documents: список из одного элемента {"doc_id": "...", "text": "..."}

doc_id: nq_doc_ + 16 hex SHA256 от нормализованного URL (nq_common).

Полный текст: span_clean_text по всему document.tokens (как в export_chunks_collection).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from nq_common import doc_id_from_url, span_clean_text


def _example_to_record(ex: Dict[str, Any]) -> Dict[str, Any]:
    doc = ex["document"]
    doc_id = doc_id_from_url(doc["url"])
    tokens = doc["tokens"]
    n = min(len(tokens["token"]), len(tokens["is_html"]))
    text = span_clean_text(tokens, 0, n)
    return {
        "question_id": ex["id"],
        "documents": [{"doc_id": doc_id, "text": text}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default="nq_open_validation_enriched",
        help="Каталог датасета (load_from_disk).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="documents_collection.jsonl",
        help="Выходной jsonl.",
    )
    args = parser.parse_args()

    from datasets import load_from_disk

    print(f"Загрузка: {args.input}", file=sys.stderr)
    ds = load_from_disk(args.input)
    need = {"id", "document"}
    missing = need - set(ds.column_names)
    if missing:
        print(f"Ошибка: нет колонок: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    print(f"Запись: {args.output}", file=sys.stderr)
    n = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds, start=1):
            rec = _example_to_record(ex)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if i % 500 == 0:
                print(f"  обработано примеров: {i}", file=sys.stderr)
    print(f"Готово, строк: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
