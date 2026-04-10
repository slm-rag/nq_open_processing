#!/usr/bin/env python3
"""
Экспорт chunk_relevance.jsonl из обогащённого датасета.

Формат строки (аналог SimpleQA):
{"question_id": "...", "documents": [{"doc_id": "...", "chunks": [
  {"id": "nq_chunk_...", "relevant": 1}, ...
]}]}

doc_id и chunk id совпадают с export_chunks_collection.py (nq_common).

Релевантность: relevant=1, если хотя бы у одного разметчика в annotations["long_answer"]
поле candidate_index равно индексу кандидата и >= 0 (объединение по всем аннотациям).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Set

from nq_common import chunk_id_from_span, doc_id_from_url, span_clean_text


def _relevant_candidate_indices(annotations: Dict[str, Any]) -> Set[int]:
    out: Set[int] = set()
    for la in annotations["long_answer"]:
        ci = la.get("candidate_index", -1)
        try:
            cii = int(ci)
        except (TypeError, ValueError):
            continue
        if cii >= 0:
            out.add(cii)
    return out


def _example_to_record(ex: Dict[str, Any]) -> Dict[str, Any]:
    doc = ex["document"]
    doc_id = doc_id_from_url(doc["url"])
    question_id = ex["id"]
    lac = ex["long_answer_candidates"]
    tok = doc["tokens"]
    n_cand = len(lac["start_token"])
    relevant = _relevant_candidate_indices(ex["annotations"])

    chunks_out: List[Dict[str, Any]] = []
    for j in range(n_cand):
        st = lac["start_token"][j]
        et = lac["end_token"][j]
        text = span_clean_text(tok, st, et)
        cid = chunk_id_from_span(doc_id, st, et, text)
        rel = 1 if j in relevant else 0
        chunks_out.append({"id": cid, "relevant": rel})

    return {
        "question_id": question_id,
        "documents": [{"doc_id": doc_id, "chunks": chunks_out}],
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
        default="chunk_relevance.jsonl",
        help="Выходной jsonl.",
    )
    args = parser.parse_args()

    from datasets import load_from_disk

    print(f"Загрузка: {args.input}", file=sys.stderr)
    ds = load_from_disk(args.input)
    need = {"id", "long_answer_candidates", "annotations", "document"}
    missing = need - set(ds.column_names)
    if missing:
        print(f"Ошибка: нет колонок: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    print(f"Запись: {args.output}", file=sys.stderr)
    n = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds, start=1):
            rel_set = _relevant_candidate_indices(ex["annotations"])
            n_cand = len(ex["long_answer_candidates"]["start_token"])
            bad = [c for c in rel_set if c >= n_cand]
            if bad:
                print(
                    f"Предупреждение: question_id={ex['id']} — candidate_index вне диапазона: {bad}",
                    file=sys.stderr,
                )
            rec = _example_to_record(ex)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if i % 500 == 0:
                print(f"  обработано примеров: {i}", file=sys.stderr)
    print(f"Готово, строк: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
