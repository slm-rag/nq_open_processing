#!/usr/bin/env python3
"""
Проверка согласованности идентификаторов с nq_common:

1) Один нормализованный URL → ровно один doc_id (иначе exit 1).
2) Для строк с одним и тем же URL: если long_answer_candidates (спаны) совпадают между всеми парами строк, списки chunk_id по индексам должны совпадать (иначе exit 2).
3) Если у одного URL разные спаны или разный размер документа — это не ошибка экспорта:
   в датасете редко встречается одна и та же ссылка при слегка разных снимках страницы.
   Такие случаи печатаются в stderr и учитываются в сводке (exit 0, если нет ошибок 1–2).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from nq_common import chunk_id_from_span, doc_id_from_url, normalize_url, span_clean_text


def _lac_signature(lac) -> Tuple[tuple, tuple]:
    st = tuple(int(x) for x in lac["start_token"])
    et = tuple(int(x) for x in lac["end_token"])
    return st, et


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default="nq_open_validation_enriched",
        help="Каталог датасета (load_from_disk).",
    )
    args = parser.parse_args()

    from datasets import load_from_disk

    print(f"Загрузка: {args.input}", file=sys.stderr)
    ds = load_from_disk(args.input)

    url_to_doc_ids: Dict[str, Set[str]] = defaultdict(set)
    url_to_rows: Dict[str, List[Tuple[int, str, str, List[str], Tuple[tuple, tuple], int]]] = (
        defaultdict(list)
    )

    for idx, ex in enumerate(ds):
        url = ex["document"]["url"]
        norm = normalize_url(url)
        did = doc_id_from_url(url)
        url_to_doc_ids[norm].add(did)

        lac = ex["long_answer_candidates"]
        tok = ex["document"]["tokens"]
        ntok = min(len(tok["token"]), len(tok["is_html"]))
        sig = _lac_signature(lac)
        cids: List[str] = []
        for j in range(len(lac["start_token"])):
            st, et = lac["start_token"][j], lac["end_token"][j]
            text = span_clean_text(tok, st, et)
            cids.append(chunk_id_from_span(did, st, et, text))
        url_to_rows[norm].append((idx, ex["id"], did, cids, sig, ntok))

    bad_doc = [u for u, s in url_to_doc_ids.items() if len(s) != 1]
    print(f"Уникальных нормализованных URL: {len(url_to_doc_ids)}", file=sys.stderr)
    print(f"URL с несколькими разными doc_id: {len(bad_doc)}", file=sys.stderr)
    if bad_doc:
        for u in bad_doc[:15]:
            print(f"  BAD url={u!r} doc_ids={url_to_doc_ids[u]}", file=sys.stderr)
        sys.exit(1)

    multi = [(u, rows) for u, rows in url_to_rows.items() if len(rows) > 1]
    bad_chunks = 0
    divergent_snapshots = 0

    for norm_url, rows in multi:
        sigs = {r[4] for r in rows}
        ntokens = {r[5] for r in rows}
        if len(sigs) > 1 or len(ntokens) > 1:
            divergent_snapshots += 1
            if divergent_snapshots <= 5:
                print(
                    f"  Предупреждение: один URL, разные снимки документа "
                    f"(число вариантов спанов={len(sigs)}, токенов={sorted(ntokens)}): "
                    f"{norm_url[:95]}…",
                    file=sys.stderr,
                )
            continue

        ref_cids = rows[0][3]
        for _, qid1, _, cids1, _, _ in rows[1:]:
            if cids1 != ref_cids:
                bad_chunks += 1
                for j, (a, b) in enumerate(zip(ref_cids, cids1)):
                    if a != b:
                        print(
                            f"  ОШИБКА: при совпадающих спанах разный chunk_id j={j}: "
                            f"{a} vs {b} (question_id {rows[0][1]} vs {qid1})",
                            file=sys.stderr,
                        )
                        break

    print(f"URL с несколькими вопросами (строками): {len(multi)}", file=sys.stderr)
    print(f"Из них разные снимки/спаны при том же URL: {divergent_snapshots}", file=sys.stderr)
    print(f"Ошибок chunk_id при одинаковых спанах: {bad_chunks}", file=sys.stderr)

    if bad_chunks:
        sys.exit(2)
    print(
        "OK: doc_id стабилен по URL; для строк с одним и тем же набором кандидатов "
        "chunk_id совпадают.",
        file=sys.stderr,
    )
    if divergent_snapshots:
        print(
            f"Учтите: у {divergent_snapshots} URL несколько разных версий документа в данных — "
            "chunk_id на одном индексе может различаться.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
