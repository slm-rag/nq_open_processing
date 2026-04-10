#!/usr/bin/env python3
"""
Собирает статистику по обогащённому датасету (save_to_disk) в JSON
с ключами верхнего уровня на русском (как в шаблоне).

Детали:
- Уникальные документы: по нормализованному URL (nq_common.normalize_url).
- Длина документа: очищенный текст (span_clean_text на всех токенах); «токены» = число
  токенов с is_html=False.
- Long answer на вопрос: средняя длина по уникальным gold candidate_index из аннотаций
  (объединение разметчиков), затем агрегаты по вопросам.
- Yes/No по вопросу: среди голосов != -1 большинство (1=YES, 0=NO); при ничьей или
  отсутствии голосов — NONE.
- «Среднее количество short answers на вопрос»: среднее по вопросам от
  (число непустых строк short answer / число аннотаций на этот вопрос).

Запуск:
  python collect_dataset_stats.py --input nq_open_validation_enriched --output stats.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set

from nq_common import doc_id_from_url, normalize_url, span_clean_text


def _non_html_token_count(tokens: Dict[str, Any], start_token: int, end_token: int) -> int:
    toks = tokens["token"]
    is_html = tokens["is_html"]
    n = min(len(toks), len(is_html))
    lo = max(0, int(start_token))
    hi = min(n, int(end_token))
    if hi <= lo:
        return 0
    return sum(1 for i in range(lo, hi) if not bool(is_html[i]))


def _safe_stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def _stats_dict(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "Средняя": 0.0,
            "Медиана": 0.0,
            "Минимум": 0.0,
            "Максимум": 0.0,
            "Стандартное отклонение": 0.0,
        }
    return {
        "Средняя": statistics.mean(values),
        "Медиана": float(statistics.median(values)),
        "Минимум": float(min(values)),
        "Максимум": float(max(values)),
        "Стандартное отклонение": _safe_stdev(values),
    }


def _yes_no_label_for_question(ys: List[int]) -> str:
    """1=YES, 0=NO, -1=NONE; по вопросу — большинство среди голосов != -1, при ничьей — NONE."""
    votes = [int(y) for y in ys if int(y) != -1]
    if not votes:
        return "NONE"
    n_yes = votes.count(1)
    n_no = votes.count(0)
    if n_yes > n_no:
        return "YES"
    if n_no > n_yes:
        return "NO"
    return "NONE"


def _yes_no_annotation_label(y: int) -> str:
    y = int(y)
    if y == -1:
        return "NONE"
    if y == 1:
        return "YES"
    if y == 0:
        return "NO"
    return "NONE"


def collect_stats(ds) -> Dict[str, Any]:
    n_rows = len(ds)

    unique_urls: Set[str] = set()
    doc_lengths_chars: List[float] = []
    doc_lengths_toks: List[float] = []

    docs_per_question: Dict[str, Set[str]] = defaultdict(set)

    long_para_chars_per_q: List[float] = []
    long_para_toks_per_q: List[float] = []

    short_answer_chars: List[float] = []
    short_answer_toks: List[float] = []

    n_with_annotations = 0
    n_with_long = 0
    n_with_short = 0
    n_with_yesno_any = 0

    ann_counts: List[int] = []
    short_counts_per_q: List[float] = []

    yes_no_per_question: Counter = Counter()
    yes_no_per_annotation: Counter = Counter()

    for ex in ds:
        doc = ex["document"]
        url = doc["url"]
        norm_url = normalize_url(url)
        unique_urls.add(norm_url)

        qid = ex["id"]
        did = doc_id_from_url(url)
        docs_per_question[qid].add(did)

        tok = doc["tokens"]
        n_tok_full = min(len(tok["token"]), len(tok["is_html"]))
        full_text = span_clean_text(tok, 0, n_tok_full)
        doc_lengths_chars.append(float(len(full_text)))
        doc_lengths_toks.append(float(_non_html_token_count(tok, 0, n_tok_full)))

        ann = ex["annotations"]
        n_ann = len(ann["long_answer"])
        ann_counts.append(n_ann)
        if n_ann > 0:
            n_with_annotations += 1

        rel_idx: Set[int] = set()
        for la in ann["long_answer"]:
            ci = la.get("candidate_index", -1)
            try:
                cii = int(ci)
            except (TypeError, ValueError):
                continue
            if cii >= 0:
                rel_idx.add(cii)

        lac = ex["long_answer_candidates"]
        if rel_idx:
            n_with_long += 1
            chars_spans = []
            toks_spans = []
            for ci in sorted(rel_idx):
                if ci >= len(lac["start_token"]):
                    continue
                st = lac["start_token"][ci]
                et = lac["end_token"][ci]
                t = span_clean_text(tok, st, et)
                chars_spans.append(float(len(t)))
                toks_spans.append(float(_non_html_token_count(tok, st, et)))
            if chars_spans:
                long_para_chars_per_q.append(statistics.mean(chars_spans))
                long_para_toks_per_q.append(statistics.mean(toks_spans))

        has_short = False
        total_short_strings = 0
        for j in range(len(ann["short_answers"])):
            sa = ann["short_answers"][j]
            texts = sa.get("text") or []
            starts = sa.get("start_token") or []
            ends = sa.get("end_token") or []
            for k, s in enumerate(texts):
                if not s:
                    continue
                has_short = True
                total_short_strings += 1
                short_answer_chars.append(float(len(s)))
                if k < len(starts) and k < len(ends):
                    short_answer_toks.append(
                        float(_non_html_token_count(tok, starts[k], ends[k]))
                    )
                else:
                    short_answer_toks.append(float(len(str(s).split())))
        if has_short:
            n_with_short += 1
        if n_ann > 0:
            short_counts_per_q.append(total_short_strings / float(n_ann))

        yn_list = ann["yes_no_answer"]
        if any(int(y) != -1 for y in yn_list):
            n_with_yesno_any += 1
        yes_no_per_question[_yes_no_label_for_question(yn_list)] += 1
        for y in yn_list:
            yes_no_per_annotation[_yes_no_annotation_label(y)] += 1

    counts_docs_per_q = [len(s) for s in docs_per_question.values()]

    out: Dict[str, Any] = {
        "Общая статистика": {
            "Всего вопросов": n_rows,
            "Всего документов": n_rows,
            "Уникальных документов (по URL)": len(unique_urls),
        },
        "Количество документов на вопрос": {
            "Среднее": statistics.mean(counts_docs_per_q) if counts_docs_per_q else 0.0,
            "Медиана": float(statistics.median(counts_docs_per_q)) if counts_docs_per_q else 0.0,
            "Минимум": float(min(counts_docs_per_q)) if counts_docs_per_q else 0,
            "Максимум": float(max(counts_docs_per_q)) if counts_docs_per_q else 0,
        },
        "Длина документа (символы)": _stats_dict(doc_lengths_chars),
        "Длина документа (токены/слова)": _stats_dict(doc_lengths_toks),
        "Длина параграфа (long answer) - символы": _stats_dict(long_para_chars_per_q),
        "Длина параграфа (long answer) - токены": _stats_dict(long_para_toks_per_q),
        "Дополнительная статистика": {
            "Вопросов с аннотациями": n_with_annotations,
            "Вопросов с long answer": n_with_long,
            "Вопросов с short answer": n_with_short,
            "Вопросов с yes/no answer": n_with_yesno_any,
            "Среднее количество аннотаций на вопрос": statistics.mean(ann_counts)
            if ann_counts
            else 0.0,
            "Среднее количество short answers на вопрос": statistics.mean(short_counts_per_q)
            if short_counts_per_q
            else 0.0,
        },
        "Распределение yes/no ответов (по вопросам)": {
            k: int(yes_no_per_question.get(k, 0)) for k in ("NONE", "YES", "NO")
        },
        "Распределение yes/no ответов (по аннотациям)": {
            k: int(yes_no_per_annotation.get(k, 0)) for k in ("NONE", "YES", "NO")
        },
        "Длина short answer (символы)": _stats_dict(short_answer_chars),
        "Длина short answer (токены)": _stats_dict(short_answer_toks),
    }
    return out


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
        default="dataset_stats.json",
        help="Куда записать JSON.",
    )
    args = parser.parse_args()

    from datasets import load_from_disk

    print(f"Загрузка: {args.input}", file=sys.stderr)
    ds = load_from_disk(args.input)
    print("Подсчёт…", file=sys.stderr)
    stats = collect_stats(ds)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Записано: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
