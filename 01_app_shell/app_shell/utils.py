from __future__ import annotations

import copy
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def deep_copy(value: Any) -> Any:
    return copy.deepcopy(value)



def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return cleaned or "item"



def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return deep_copy(default)
    return json.loads(path.read_text(encoding="utf-8"))



def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")



def absolutize_url(base_url: str, url: str | None) -> str | None:
    if not url:
        return None
    text = str(url).strip()
    if not text:
        return None
    if text.startswith(("http://", "https://", "data:")):
        return text
    return urljoin(base_url.rstrip("/") + "/", text.lstrip("/"))
