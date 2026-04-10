"""
Общая логика для экспорта NQ: нормализация URL, doc_id, chunk_id, очистка текста по токенам.

doc_id = nq_doc_ + первые 16 hex-символов SHA256(UTF-8 канонического URL).
Канонический URL: html.unescape; scheme и host в нижнем регистре; path = unquote,
схлопнутые слэши, без завершающего / (кроме пустого пути); query — пары parse_qsl,
отсортированные по (ключ, значение); без fragment.

chunk_id = nq_chunk_ + первые 16 hex SHA256(UTF-8 очищенного текста чанка).
Одинаковый текст → один и тот же chunk_id (в т.ч. для разных вопросов с одним URL).
Пустой текст после очистки: в хэш идёт doc_id\\nstart_token\\nend_token (разные пустые спаны).
"""

from __future__ import annotations

import hashlib
import html
import re
from typing import Any, Dict, List
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

DOC_PREFIX = "nq_doc_"
CHUNK_PREFIX = "nq_chunk_"


def normalize_url(url: str) -> str:
    s = html.unescape((url or "").strip())
    p = urlparse(s)
    scheme = (p.scheme or "").lower()
    netloc = (p.netloc or "").lower()
    path = unquote(p.path or "")
    path = re.sub(r"/+", "/", path)
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    pairs = parse_qsl(p.query, keep_blank_values=True)
    pairs.sort(key=lambda kv: (kv[0], kv[1]))
    query = urlencode(pairs, doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


def doc_id_from_url(url: str) -> str:
    canonical = normalize_url(url)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return DOC_PREFIX + digest


def span_clean_text(tokens: Dict[str, Any], start_token: int, end_token: int) -> str:
    """Текст спана [start_token, end_token) без токенов с is_html=True."""
    toks = tokens["token"]
    is_html = tokens["is_html"]
    n = min(len(toks), len(is_html))
    lo = max(0, int(start_token))
    hi = min(n, int(end_token))
    if hi <= lo:
        return ""
    parts: List[str] = []
    for i in range(lo, hi):
        if not bool(is_html[i]):
            parts.append(toks[i])
    text = " ".join(parts)
    return re.sub(r"\s+", " ", text).strip()


def chunk_id_from_span(doc_id: str, start_token: int, end_token: int, clean_text: str) -> str:
    if clean_text.strip():
        payload = clean_text.encode("utf-8")
    else:
        payload = f"{doc_id}\n{int(start_token)}\n{int(end_token)}".encode("utf-8")
    return CHUNK_PREFIX + hashlib.sha256(payload).hexdigest()[:16]
