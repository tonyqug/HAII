from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "being", "by",
    "can", "do", "does", "for", "from", "how", "if", "in", "into", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "this", "to", "used", "using", "what",
    "when", "where", "which", "with", "your", "you", "we", "our", "they", "these", "those",
    "than", "then", "also", "more", "most", "less", "very", "one", "two", "three", "such",
    "via", "while", "under", "over", "each", "about", "into", "out", "up", "down", "not",
    "only", "may", "might", "should", "could", "would", "must", "will", "lecture", "slide",
    "slides", "materials", "material", "topic", "course", "student", "students",
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)



def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))



def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())



def tokenize(text: str) -> List[str]:
    return [token.lower() for token in WORD_RE.findall(text or "")]



def informative_tokens(text: str) -> List[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 2]



def keyword_counter(texts: Sequence[str]) -> Counter:
    counter: Counter = Counter()
    for text in texts:
        counter.update(informative_tokens(text))
    return counter



def top_keywords(texts: Sequence[str], limit: int = 5) -> List[str]:
    return [word for word, _count in keyword_counter(texts).most_common(limit)]



def first_sentence(text: str) -> str:
    clean = normalize_whitespace(text)
    if not clean:
        return ""
    parts = SENTENCE_SPLIT_RE.split(clean)
    return parts[0].strip()



def split_sentences(text: str) -> List[str]:
    clean = normalize_whitespace(text)
    if not clean:
        return []
    return [segment.strip() for segment in SENTENCE_SPLIT_RE.split(clean) if segment.strip()]



def take_sentences(text: str, count: int = 2) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    return " ".join(sentences[:count])



def safe_excerpt(text: str, max_length: int = 240) -> str:
    clean = normalize_whitespace(text)
    if len(clean) <= max_length:
        return clean
    truncated = clean[: max_length - 3].rsplit(" ", 1)[0].strip()
    return f"{truncated}..."



def dedupe_citations(citations: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output: List[Dict[str, Any]] = []
    for citation in citations:
        if not citation:
            continue
        citation_id = citation.get("citation_id") or json.dumps(citation, sort_keys=True)
        if citation_id in seen:
            continue
        seen.add(citation_id)
        output.append(citation)
    return output



def distinct_slide_numbers(citations: Iterable[Dict[str, Any]]) -> List[int]:
    values = {
        int(citation.get("slide_number"))
        for citation in citations
        if citation and citation.get("slide_number") is not None
    }
    return sorted(values)



def infer_concept_label(text: str, fallback: str = "Key concept") -> str:
    clean = normalize_whitespace(text)
    if not clean:
        return fallback

    patterns = [
        r"^([A-Z][A-Za-z0-9\- ]{2,60}?)\s+(?:is|are|refers to|means|uses|adds|reduces|helps|describes|compares)\b",
        r"^([A-Za-z][A-Za-z0-9\- ]{2,60}?)\s*[:\-]",
    ]
    for pattern in patterns:
        match = re.match(pattern, clean)
        if match:
            value = normalize_whitespace(match.group(1))
            if 1 <= len(value.split()) <= 8:
                return value

    keywords = top_keywords([clean], limit=3)
    if keywords:
        return " ".join(word.capitalize() for word in keywords[:2])
    return fallback



def summarize_texts(texts: Sequence[str], sentence_count: int = 2) -> str:
    combined_sentences: List[str] = []
    seen = set()
    for text in texts:
        for sentence in split_sentences(text):
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            combined_sentences.append(sentence)
            if len(combined_sentences) >= sentence_count:
                return " ".join(combined_sentences)
    return " ".join(combined_sentences)



def lexical_overlap_score(query: str, text: str) -> int:
    query_tokens = set(informative_tokens(query))
    if not query_tokens:
        return 0
    text_tokens = set(informative_tokens(text))
    shared = query_tokens & text_tokens
    score = len(shared)
    query_lower = normalize_whitespace(query).lower()
    text_lower = normalize_whitespace(text).lower()
    if query_lower and query_lower in text_lower:
        score += 5
    for token in query_tokens:
        if token in text_lower:
            score += 1
    return score



def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
