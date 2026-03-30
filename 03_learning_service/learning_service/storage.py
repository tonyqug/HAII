from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import atomic_write_json, read_json


class JsonStore:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self._lock = threading.RLock()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for name in ["jobs", "study_plans", "conversations", "practice_sets"]:
            (self.root_dir / name).mkdir(parents=True, exist_ok=True)

    def _path(self, collection: str, object_id: str) -> Path:
        return self.root_dir / collection / f"{object_id}.json"

    def save(self, collection: str, object_id: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            atomic_write_json(self._path(collection, object_id), payload)

    def load(self, collection: str, object_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(collection, object_id)
        if not path.exists():
            return None
        with self._lock:
            return read_json(path)

    def list_all(self, collection: str) -> List[Dict[str, Any]]:
        directory = self.root_dir / collection
        if not directory.exists():
            return []
        with self._lock:
            items = [read_json(path) for path in directory.glob("*.json")]
        return items

    def update(self, collection: str, object_id: str, **updates: Any) -> Dict[str, Any]:
        with self._lock:
            current = self.load(collection, object_id)
            if current is None:
                raise KeyError(f"Missing object {collection}/{object_id}")
            current.update(updates)
            atomic_write_json(self._path(collection, object_id), current)
            return current
