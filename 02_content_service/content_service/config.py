from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 38410
    local_data_dir: Path = Path("./local_data")
    service_name: str = "content_service"
    version: str = "1.0.0"
    import_workers: int = 2
    libreoffice_bin: str = "soffice"

    @property
    def api_base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def db_path(self) -> Path:
        return self.local_data_dir / "content_service.sqlite3"

    @property
    def storage_dir(self) -> Path:
        return self.local_data_dir / "storage"


ENV_KEY_VALUE = "="
INTEGRATED_SERVICE_DIR = "02_content_service"


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists() or not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.lstrip("\ufeff").strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if ENV_KEY_VALUE not in line:
            continue
        key, value = line.split(ENV_KEY_VALUE, 1)
        key = key.strip().lstrip("\ufeff")
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def _looks_like_integrated_root(candidate: Path, service_dir_name: str = INTEGRATED_SERVICE_DIR) -> bool:
    if not candidate.exists() or not candidate.is_dir():
        return False
    service_dir = candidate / service_dir_name
    if not service_dir.is_dir():
        return False
    for child in candidate.iterdir():
        if not child.is_dir() or child.name == service_dir_name:
            continue
        if len(child.name) >= 4 and child.name[:2].isdigit() and child.name[2] == "_":
            return True
    return False


def detect_service_root(service_root: Optional[Path] = None) -> Path:
    if service_root is not None:
        return service_root.expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def find_integrated_project_root(service_root: Optional[Path] = None) -> Optional[Path]:
    resolved_service_root = detect_service_root(service_root)
    if resolved_service_root.name == INTEGRATED_SERVICE_DIR:
        parent = resolved_service_root.parent
        if _looks_like_integrated_root(parent, resolved_service_root.name):
            return parent.resolve()
    return None


def _resolve_path(raw_value: str, *, base_dir: Path) -> Path:
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _choose_env_value(
    key: str,
    *,
    environ: Mapping[str, str],
    root_env: Mapping[str, str],
    default: str,
) -> tuple[str, str]:
    if key in environ and str(environ[key]).strip() != "":
        return str(environ[key]), "process"
    if key in root_env and str(root_env[key]).strip() != "":
        return str(root_env[key]), "root_env"
    return default, "default"


def load_settings(
    *,
    service_root: Optional[Path] = None,
    cwd: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Settings:
    env_map = dict(os.environ if environ is None else environ)
    service_root_path = detect_service_root(service_root)
    cwd_path = (cwd or Path.cwd()).expanduser().resolve()
    integrated_root = find_integrated_project_root(service_root_path)
    root_env_path = integrated_root / ".env" if integrated_root else None
    root_env = _parse_env_file(root_env_path) if root_env_path else {}

    port_raw, _ = _choose_env_value("CONTENT_SERVICE_PORT", environ=env_map, root_env=root_env, default="38410")
    import_workers_raw, _ = _choose_env_value("CONTENT_SERVICE_IMPORT_WORKERS", environ=env_map, root_env=root_env, default="2")
    libreoffice_bin, _ = _choose_env_value("LIBREOFFICE_BIN", environ=env_map, root_env=root_env, default="soffice")
    version, _ = _choose_env_value("CONTENT_SERVICE_VERSION", environ=env_map, root_env=root_env, default="1.0.0")
    local_data_raw, local_data_source = _choose_env_value("LOCAL_DATA_DIR", environ=env_map, root_env=root_env, default="./local_data")

    if local_data_source == "process":
        local_data_dir = _resolve_path(local_data_raw, base_dir=cwd_path)
    elif local_data_source == "root_env" and root_env_path is not None:
        local_data_dir = _resolve_path(local_data_raw, base_dir=root_env_path.parent)
    else:
        default_base = integrated_root if integrated_root is not None else cwd_path
        local_data_dir = (default_base / "local_data").resolve()

    return Settings(
        port=int(port_raw),
        local_data_dir=local_data_dir,
        import_workers=max(int(import_workers_raw), 1),
        libreoffice_bin=libreoffice_bin,
        version=version,
    )
