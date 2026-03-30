from __future__ import annotations

from pathlib import Path

from learning_service.app import create_app
from learning_service.config import Settings

from .conftest import LocalClient, wait_for_job


RELEVANT_ENV_KEYS = [
    "GEMINI_API_KEY",
    "LOCAL_DATA_DIR",
    "CONTENT_SERVICE_URL",
    "LEARNING_SERVICE_PORT",
    "USE_HEURISTIC_FALLBACK",
    "LEARNING_SERVICE_REQUEST_TIMEOUT_SECONDS",
]


def _write_integrated_root_env(root: Path) -> None:
    (root / ".env").write_text(
        "\n".join(
            [
                "GEMINI_API_KEY=root-gemini-key",
                "LOCAL_DATA_DIR=./local_data",
                "CONTENT_SERVICE_URL=http://127.0.0.1:48888",
                "LEARNING_SERVICE_PORT=38888",
            ]
        )
        + "\n",
        encoding="utf-8",
    )



def _prepare_integrated_layout(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "integrated_project"
    for folder_name in ("01_app_shell", "02_content_service", "03_learning_service"):
        (root / folder_name).mkdir(parents=True, exist_ok=True)
    _write_integrated_root_env(root)
    return root, root / "03_learning_service"



def test_integrated_root_startup_paths_load_shared_root_env(monkeypatch, tmp_path):
    root, service_dir = _prepare_integrated_layout(tmp_path)
    for key in RELEVANT_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("learning_service.config.SERVICE_DIR", service_dir)

    observed = []
    for cwd in (root, root / "01_app_shell", root / "03_learning_service"):
        monkeypatch.chdir(cwd)
        settings = Settings.from_env()
        observed.append(
            (
                settings.port,
                settings.content_service_url,
                settings.gemini_api_key,
                settings.local_data_dir,
            )
        )

    assert len(set(observed)) == 1
    port, content_service_url, gemini_api_key, local_data_dir = observed[0]
    assert port == 38888
    assert content_service_url == "http://127.0.0.1:48888"
    assert gemini_api_key == "root-gemini-key"
    assert local_data_dir == (root / "local_data").resolve()



def test_jobs_and_artifacts_persist_under_root_local_data(monkeypatch, tmp_path, bundle):
    root, service_dir = _prepare_integrated_layout(tmp_path)
    for key in RELEVANT_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("learning_service.config.SERVICE_DIR", service_dir)
    monkeypatch.chdir(root / "01_app_shell")

    settings = Settings.from_env()
    client = LocalClient(create_app(settings))

    response = client.post(
        "/v1/study-plans",
        json={
            "workspace_id": bundle["workspace_id"],
            "evidence_bundle": bundle,
            "topic_text": "validation",
            "time_budget_minutes": 30,
            "grounding_mode": "lecture_with_fallback",
            "student_context": {},
            "include_annotations": True,
        },
    )
    assert response.status_code == 200
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded"

    root_local_data = (root / "local_data").resolve()
    assert root_local_data.exists()
    assert list((root_local_data / "jobs").glob("*.json"))
    assert list((root_local_data / "study_plans").glob("*.json"))
    assert not (service_dir / "local_data").exists()
