#!/usr/bin/env python3
"""
Экспорт chunks_collection.jsonl из обогащённого датасета.

Каждая строка — один пример (документ + вопрос):
- doc_id: nq_doc_ + 16 hex SHA256 от нормализованного URL (см. nq_common.normalize_url)
- url, title из document
- question_id из id
- chuks: кандидаты long_answer_candidates; chunk_id — по очищенному тексту (nq_common.chunk_id_from_span)

Текст чанка: склейка токенов document.tokens.token[i] для i в [start_token, end_token)
только там, где is_html[i] == False.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from nq_common import chunk_id_from_span, doc_id_from_url, span_clean_text


def _example_to_record(ex: Dict[str, Any]) -> Dict[str, Any]:
    doc = ex["document"]
    url = doc["url"]
    doc_id = doc_id_from_url(url)
    title = doc["title"]
    question_id = ex["id"]
    lac = ex["long_answer_candidates"]
    tok = doc["tokens"]
    n_cand = len(lac["start_token"])
    chuks: List[Dict[str, str]] = []
    for j in range(n_cand):
        st = lac["start_token"][j]
        et = lac["end_token"][j]
        text = span_clean_text(tok, st, et)
        cid = chunk_id_from_span(doc_id, st, et, text)
        chuks.append({"id": cid, "text": text})
    return {
        "doc_id": doc_id,
        "url": url,
        "title": title,
        "question_id": question_id,
        "chuks": chuks,
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
        default="chunks_collection.jsonl",
        help="Выходной jsonl.",
    )
    args = parser.parse_args()

    from datasets import load_from_disk

    print(f"Загрузка: {args.input}", file=sys.stderr)
    ds = load_from_disk(args.input)
    need = {"id", "document", "long_answer_candidates"}
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
