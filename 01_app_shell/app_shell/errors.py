from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ShellError(Exception):
    message: str
    status_code: int = 400
    details: Dict[str, Any] | None = None

    def as_payload(self) -> dict:
        return {
            "error": {
                "message": self.message,
                "details": self.details or {},
            }
        }
