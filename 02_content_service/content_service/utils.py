from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return name or "file"


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def significant_terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Za-z0-9 \-:,()]{0,80}:?)$")


def split_text_into_units(text: str, max_chars: int = 1800) -> list[str]:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return [""]
    lines = normalized.split("\n")
    blocks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            block = "\n".join(current).strip()
            if block:
                blocks.append(block)
        current = []
        current_len = 0

    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            if current_len >= max_chars * 0.45:
                flush()
            elif current:
                current.append("")
            continue
        is_heading = bool(HEADING_RE.match(stripped))
        if is_heading and current and current_len >= max_chars * 0.25:
            flush()
        if len(stripped) > max_chars:
            wrapped = textwrap.wrap(stripped, width=max(60, max_chars // 3))
            for piece in wrapped:
                if current_len + len(piece) + 1 > max_chars and current:
                    flush()
                current.append(piece)
                current_len += len(piece) + 1
            continue
        if current_len + len(stripped) + 1 > max_chars and current:
            flush()
        current.append(stripped)
        current_len += len(stripped) + 1
    flush()
    return blocks or [normalized]


def extract_title_guess(text: str) -> Optional[str]:
    for line in (text or "").splitlines():
        candidate = line.strip().strip("#").strip()
        if len(candidate) >= 3:
            return candidate[:160]
    return None


def quality_rank(value: Optional[str]) -> int:
    mapping = {"low": 1, "medium": 2, "high": 3}
    return mapping.get((value or "low").lower(), 1)


def clamp_quality(value: Optional[str]) -> str:
    value = (value or "low").lower()
    if value in {"low", "medium", "high"}:
        return value
    return "low"


def summarize_quality(values: Sequence[str], previews_available: bool, has_text_ratio: float) -> Tuple[str, Optional[str]]:
    if not values:
        return "low", "No slide-level content was extracted."
    highs = sum(1 for v in values if v == "high")
    mediums = sum(1 for v in values if v == "medium")
    lows = sum(1 for v in values if v == "low")
    total = len(values)
    if previews_available and highs == total and has_text_ratio >= 0.85:
        return "high", None
    if previews_available and (highs + mediums) / total >= 0.6 and has_text_ratio >= 0.35:
        note = None
        if lows:
            note = f"{lows} slide(s) had limited or noisy extraction."
        return "medium", note
    note = "Text extraction was limited or partial; use slide previews cautiously."
    if not previews_available:
        note = "Preview generation was incomplete and text extraction was limited."
    return "low", note


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def json_dumps(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def best_snippet(text: str, query_text: str, max_len: int = 320) -> str:
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if len(clean) <= max_len:
        return clean
    terms = significant_terms(query_text)
    if not terms:
        return clean[: max_len - 1].rstrip() + "…"
    lowered = clean.lower()
    best_idx = -1
    for term in terms:
        idx = lowered.find(term.lower())
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx = idx
    if best_idx == -1:
        return clean[: max_len - 1].rstrip() + "…"
    start = max(0, best_idx - max_len // 4)
    end = min(len(clean), start + max_len)
    snippet = clean[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(clean):
        snippet = snippet.rstrip() + "…"
    return snippet


def stable_citation_id(*parts: str) -> str:
    joined = "|".join(parts)
    return f"cit_{sha1_text(joined)[:24]}"


def stable_bundle_id(*parts: str) -> str:
    joined = "|".join(parts)
    return f"bundle_{sha1_text(joined)[:24]}"


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
