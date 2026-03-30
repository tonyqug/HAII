from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict

import httpx
import pytest

from learning_service.app import create_app
from learning_service.config import Settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
FINAL_JOB_STATUSES = {"succeeded", "failed", "needs_user_input"}


class LocalClient:
    def __init__(self, app):
        self.app = app
        self.transport = httpx.ASGITransport(app=app)
        self.base_url = "http://testserver"

    async def _request_async(self, method: str, url: str, **kwargs):
        async with httpx.AsyncClient(transport=self.transport, base_url=self.base_url) as client:
            return await client.request(method, url, **kwargs)

    def request(self, method: str, url: str, **kwargs):
        return asyncio.run(self._request_async(method, url, **kwargs))

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)


@pytest.fixture()
def bundle() -> Dict[str, Any]:
    return json.loads((SAMPLE_DATA_DIR / "standalone_evidence_bundle.json").read_text(encoding="utf-8"))


@pytest.fixture()
def template_bundle() -> Dict[str, Any]:
    return json.loads((SAMPLE_DATA_DIR / "standalone_template_mimic_bundle.json").read_text(encoding="utf-8"))


@pytest.fixture()
def client(tmp_path: Path) -> LocalClient:
    settings = Settings(
        port=38420,
        content_service_url="http://127.0.0.1:39999",
        gemini_api_key="",
        local_data_dir=tmp_path / "local_data",
        use_heuristic_fallback=True,
    )
    app = create_app(settings)
    return LocalClient(app)


@pytest.fixture()
def app_factory(tmp_path: Path):
    def _factory(data_dir: Path | None = None, *, gemini_api_key: str = "") -> LocalClient:
        root = data_dir or (tmp_path / "factory_data")
        settings = Settings(
            port=38420,
            content_service_url="http://127.0.0.1:39999",
            gemini_api_key=gemini_api_key,
            local_data_dir=root,
            use_heuristic_fallback=True,
        )
        return LocalClient(create_app(settings))

    return _factory


def fetch_job(client: LocalClient, job_id: str) -> Dict[str, Any]:
    response = client.get(f"/v1/jobs/{job_id}")
    assert response.status_code == 200, response.text
    return response.json()


def wait_for_job(client: LocalClient, job_id: str, timeout: float = 5.0, interval: float = 0.02) -> Dict[str, Any]:
    deadline = time.perf_counter() + timeout
    last = fetch_job(client, job_id)
    while time.perf_counter() < deadline:
        if last["status"] in FINAL_JOB_STATUSES:
            return last
        time.sleep(interval)
        last = fetch_job(client, job_id)
    raise AssertionError(f"job {job_id} did not finish within {timeout} seconds: {last}")
