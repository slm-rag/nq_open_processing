#!/usr/bin/env python3
"""
Сопоставляет validation nq_open с validation Natural Questions (config dev):
- question и answer остаются из nq_open;
- id, document, long_answer_candidates, annotations — из исходного NQ.

Исходный датасет большой (~3.4G на сплит); читается в потоковом режиме один проход.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _nq_question_text(example: Dict[str, Any]) -> str:
    q = example["question"]
    if isinstance(q, dict):
        return str(q.get("text", ""))
    return str(q)


def normalize_question(text: str) -> str:
    return text.strip().lower()


def _row_without_question(example: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(example)
    out.pop("question", None)
    return out


def build_nq_index_for_keys(
    needed_keys: Set[str],
    *,
    cache_dir: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Один проход по validation Natural Questions. В индекс попадают только ключи из needed_keys.
    Возвращает (индекс, список предупреждений о коллизиях).
    """
    from datasets import load_dataset

    stream = load_dataset(
        "google-research-datasets/natural_questions",
        "dev",
        split="validation",
        streaming=True,
        cache_dir=cache_dir,
    )

    index: Dict[str, Dict[str, Any]] = {}
    warnings_list: List[str] = []
    seen_collision: Set[str] = set()

    for i, ex in enumerate(stream):
        if (i + 1) % 500 == 0:
            print(f"  NQ validation: обработано строк: {i + 1}, найдено из нужного множества: {len(index)}", file=sys.stderr)
        row = dict(ex)
        key = normalize_question(_nq_question_text(row))
        if key not in needed_keys:
            continue
        if key in index:
            if key not in seen_collision:
                warnings_list.append(f"Несколько строк NQ с одним и тем же вопросом (нормализованным); берём первую: {key[:120]!r}...")
                seen_collision.add(key)
            continue
        index[key] = _row_without_question(row)
        if len(index) == len(needed_keys):
            print(f"  Все {len(needed_keys)} вопросов найдены после {i + 1} строк NQ.", file=sys.stderr)
            break

    return index, warnings_list


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="nq_open_validation_enriched",
        help="Каталог для save_to_disk (Dataset).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="",
        help="Опционально: JSON с отчётом (missing, warnings, stats).",
    )
    parser.add_argument("--cache-dir", type=str, default=None, help="HF datasets cache_dir.")
    args = parser.parse_args()

    from datasets import Dataset, load_dataset

    print("Загрузка nq_open validation…", file=sys.stderr)
    nq_open = load_dataset(
        "google-research-datasets/nq_open",
        split="validation",
        cache_dir=args.cache_dir,
    )

    needed: Set[str] = set()
    open_order_keys: List[str] = []
    for row in nq_open:
        k = normalize_question(row["question"])
        needed.add(k)
        open_order_keys.append(k)

    if len(open_order_keys) != len(needed):
        print(
            f"Замечание: в nq_open validation есть дубликаты вопросов после нормализации: "
            f"{len(open_order_keys)} строк, {len(needed)} уникальных ключей.",
            file=sys.stderr,
        )

    print("Потоковое сканирование Natural Questions validation (один проход)…", file=sys.stderr)
    index, warn = build_nq_index_for_keys(needed, cache_dir=args.cache_dir)

    for w in warn:
        print(f"Предупреждение: {w}", file=sys.stderr)

    missing = sorted(needed - set(index.keys()))
    if missing:
        print("ОШИБКА: не все вопросы nq_open найдены в NQ validation:", file=sys.stderr)
        for m in missing[:50]:
            print(f"  missing: {m!r}", file=sys.stderr)
        if len(missing) > 50:
            print(f"  … и ещё {len(missing) - 50}", file=sys.stderr)
        sys.exit(1)

    merged: List[Dict[str, Any]] = []
    for row in nq_open:
        k = normalize_question(row["question"])
        base = dict(index[k])
        base["question"] = row["question"]
        base["answer"] = row["answer"]
        merged.append(base)

    out_ds = Dataset.from_list(merged)
    out_ds.save_to_disk(args.output)
    print(f"Сохранено: {args.output} ({len(out_ds)} примеров)", file=sys.stderr)

    if args.report:
        rep = {
            "nq_open_validation_rows": len(nq_open),
            "unique_normalized_questions": len(needed),
            "enriched_rows": len(merged),
            "missing_count": 0,
            "warnings": warn,
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"Отчёт: {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
