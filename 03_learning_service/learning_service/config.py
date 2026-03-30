from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


SERVICE_DIR = Path(__file__).resolve().parents[1]
INTEGRATED_LAYOUT_FOLDERS = ("01_app_shell", "02_content_service", "03_learning_service")


@dataclass(frozen=True)
class RuntimeContext:
    service_dir: Path
    integrated_root: Optional[Path]
    config_base_dir: Path
    loaded_env_files: tuple[Path, ...]


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_integrated_root(candidate: Path) -> bool:
    return all((candidate / folder_name).is_dir() for folder_name in INTEGRATED_LAYOUT_FOLDERS)


def _iter_parent_candidates(*paths: Optional[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(raw_path).resolve()
        bases = [path] if path.is_dir() else [path.parent]
        for base in bases:
            for candidate in (base, *base.parents):
                if candidate in seen:
                    continue
                seen.add(candidate)
                yield candidate


def detect_integrated_root(*, service_dir: Optional[Path] = None, cwd: Optional[Path] = None) -> Optional[Path]:
    resolved_service_dir = Path(service_dir or SERVICE_DIR).resolve()
    resolved_cwd = Path(cwd or Path.cwd()).resolve()
    for candidate in _iter_parent_candidates(resolved_cwd, resolved_service_dir, resolved_service_dir.parent):
        if _is_integrated_root(candidate):
            return candidate
    return None


def _read_dotenv_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists() or not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _build_runtime_context(
    *,
    service_dir: Optional[Path] = None,
    cwd: Optional[Path] = None,
    dotenv_path: Optional[Path] = None,
) -> tuple[RuntimeContext, dict[str, str]]:
    resolved_service_dir = Path(service_dir or SERVICE_DIR).resolve()
    resolved_cwd = Path(cwd or Path.cwd()).resolve()

    env_files: list[Path] = []
    integrated_root = detect_integrated_root(service_dir=resolved_service_dir, cwd=resolved_cwd)

    if dotenv_path is not None:
        explicit_path = Path(dotenv_path).resolve()
        env_files.append(explicit_path)
        config_base_dir = explicit_path.parent
    elif integrated_root is not None:
        config_base_dir = integrated_root
        env_files.append(integrated_root / ".env")
        service_env = resolved_service_dir / ".env"
        if service_env not in env_files:
            env_files.append(service_env)
    else:
        cwd_env = resolved_cwd / ".env"
        service_env = resolved_service_dir / ".env"
        env_files.append(cwd_env)
        if service_env not in env_files:
            env_files.append(service_env)
        if cwd_env.exists() and cwd_env.is_file():
            config_base_dir = resolved_cwd
        elif service_env.exists() and service_env.is_file():
            config_base_dir = resolved_service_dir
        else:
            config_base_dir = resolved_service_dir

    merged_env: dict[str, str] = {}
    loaded_env_files: list[Path] = []
    for env_path in env_files:
        if not env_path.exists() or not env_path.is_file():
            continue
        loaded_env_files.append(env_path)
        for key, value in _read_dotenv_file(env_path).items():
            merged_env.setdefault(key, value)

    merged_env.update(os.environ)
    context = RuntimeContext(
        service_dir=resolved_service_dir,
        integrated_root=integrated_root,
        config_base_dir=config_base_dir,
        loaded_env_files=tuple(loaded_env_files),
    )
    return context, merged_env


def load_dotenv_if_present(dotenv_path: Optional[Path] = None) -> None:
    for key, value in _read_dotenv_file(dotenv_path or Path(".env")).items():
        os.environ.setdefault(key, value)


def resolve_local_data_dir(raw_value: str, *, base_dir: Path) -> Path:
    candidate = Path(raw_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


@dataclass(frozen=True)
class Settings:
    service_name: str = "learning_service"
    version: str = "0.1.0"
    host: str = "127.0.0.1"
    port: int = 38420
    content_service_url: str = "http://127.0.0.1:38410"
    gemini_api_key: str = ""
    local_data_dir: Path = Path("./local_data")
    gemini_model: str = "gemini-2.5-flash"
    use_heuristic_fallback: bool = True
    request_timeout_seconds: int = 15

    @property
    def api_base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @classmethod
    def from_env(
        cls,
        *,
        service_dir: Optional[Path] = None,
        cwd: Optional[Path] = None,
        dotenv_path: Optional[Path] = None,
    ) -> "Settings":
        context, env = _build_runtime_context(service_dir=service_dir, cwd=cwd, dotenv_path=dotenv_path)
        return cls(
            port=int(env.get("LEARNING_SERVICE_PORT", "38420")),
            content_service_url=env.get("CONTENT_SERVICE_URL", "http://127.0.0.1:38410"),
            gemini_api_key=env.get("GEMINI_API_KEY", "").strip(),
            local_data_dir=resolve_local_data_dir(env.get("LOCAL_DATA_DIR", "./local_data"), base_dir=context.config_base_dir),
            gemini_model=env.get("GEMINI_MODEL", "gemini-2.5-flash"),
            use_heuristic_fallback=_parse_bool(env.get("USE_HEURISTIC_FALLBACK"), True),
            request_timeout_seconds=int(env.get("LEARNING_SERVICE_REQUEST_TIMEOUT_SECONDS", "15")),
        )
