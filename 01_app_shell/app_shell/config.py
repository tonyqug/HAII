from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

from app_shell import __version__


DEFAULT_CONTENT_SERVICE_URL = "http://127.0.0.1:38410"
DEFAULT_LEARNING_SERVICE_URL = "http://127.0.0.1:38420"


def _parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.lstrip("\ufeff").strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        normalized_key = key.strip().lstrip("\ufeff")
        if not normalized_key:
            continue
        normalized_value = value.strip()
        if len(normalized_value) >= 2 and normalized_value[0] == normalized_value[-1] and normalized_value[0] in {'"', "'"}:
            normalized_value = normalized_value[1:-1]
        elif " #" in normalized_value:
            normalized_value = normalized_value.split(" #", 1)[0].rstrip()
        values[normalized_key] = normalized_value
    return values


def _to_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class AppConfig:
    version: str
    app_dir: Path
    project_root: Path
    host: str
    port: int
    content_service_url: str
    learning_service_url: str
    local_data_dir: Path
    auto_open_browser: bool
    mode: str
    testing: bool

    @property
    def ui_base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def api_base_url(self) -> str:
        return self.ui_base_url

    @classmethod
    def load(cls, app_dir: Path, env_override: Mapping[str, str] | None = None) -> "AppConfig":
        project_root = app_dir.parent
        env_values: Dict[str, str] = {}
        env_values.update(_parse_env_file(project_root / ".env"))
        env_values.update({key: value for key, value in os.environ.items()})
        if env_override:
            env_values.update({key: str(value) for key, value in env_override.items()})

        testing = _to_bool(env_values.get("APP_SHELL_TESTING"), False)
        mode = env_values.get("APP_SHELL_MODE", "auto").strip().lower() or "auto"
        allowed_modes = {"auto", "integrated"}
        if testing:
            allowed_modes.add("mock")
        if mode not in allowed_modes:
            mode = "auto"

        local_data_raw = env_values.get("LOCAL_DATA_DIR", "./local_data")
        local_data_dir = Path(local_data_raw)
        if not local_data_dir.is_absolute():
            local_data_dir = (project_root / local_data_dir).resolve()

        return cls(
            version=__version__,
            app_dir=app_dir,
            project_root=project_root,
            host="127.0.0.1",
            port=int(env_values.get("APP_SHELL_PORT", "38400")),
            content_service_url=env_values.get("CONTENT_SERVICE_URL", DEFAULT_CONTENT_SERVICE_URL).rstrip("/"),
            learning_service_url=env_values.get("LEARNING_SERVICE_URL", DEFAULT_LEARNING_SERVICE_URL).rstrip("/"),
            local_data_dir=local_data_dir,
            auto_open_browser=_to_bool(env_values.get("AUTO_OPEN_BROWSER"), True),
            mode=mode,
            testing=testing,
        )
