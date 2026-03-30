from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, List

from app_shell.mock_data import build_blank_workspace, build_workspace_from_fixture
from app_shell.utils import deep_copy, ensure_directory, make_id, read_json, utc_now_iso, write_json


class LocalStorage:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir / "app_shell"
        ensure_directory(self.root_dir)
        self.state_file = self.root_dir / "state.json"
        self._lock = threading.RLock()
        self._state = read_json(
            self.state_file,
            {
                "workspaces": {},
                "jobs": {},
                "metadata": {"mock_fixture_seeded": False},
            },
        )
        self._persist()

    def _persist(self) -> None:
        write_json(self.state_file, self._state)

    def maybe_seed_mock_fixture(self) -> None:
        with self._lock:
            if self._state.get("metadata", {}).get("mock_fixture_seeded"):
                return
            workspace = build_workspace_from_fixture()
            self._state.setdefault("workspaces", {})[workspace["workspace_id"]] = workspace
            self._state.setdefault("metadata", {})["mock_fixture_seeded"] = True
            self._persist()

    def list_workspaces(self) -> List[dict]:
        with self._lock:
            workspaces = [deep_copy(workspace) for workspace in self._state.get("workspaces", {}).values() if not workspace.get("deleted")]
        workspaces.sort(key=lambda item: item.get("last_opened_at") or item.get("created_at") or "", reverse=True)
        return workspaces

    def create_workspace(self, display_name: str) -> dict:
        workspace_id = make_id("workspace")
        workspace = build_blank_workspace(workspace_id, display_name)
        with self._lock:
            self._state.setdefault("workspaces", {})[workspace_id] = workspace
            self._persist()
        return deep_copy(workspace)

    def save_workspace(self, workspace: dict) -> dict:
        with self._lock:
            workspace["last_opened_at"] = utc_now_iso()
            self._state.setdefault("workspaces", {})[workspace["workspace_id"]] = deep_copy(workspace)
            self._persist()
            return deep_copy(workspace)

    def get_workspace(self, workspace_id: str) -> dict | None:
        with self._lock:
            workspace = self._state.get("workspaces", {}).get(workspace_id)
            if workspace is None or workspace.get("deleted"):
                return None
            return deep_copy(workspace)

    def update_workspace(self, workspace_id: str, updater: Callable[[dict], dict | None]) -> dict:
        with self._lock:
            workspace = self._state.get("workspaces", {}).get(workspace_id)
            if workspace is None or workspace.get("deleted"):
                raise KeyError(workspace_id)
            updated = updater(deep_copy(workspace))
            if updated is None:
                updated = workspace
            updated["last_opened_at"] = utc_now_iso()
            self._state.setdefault("workspaces", {})[workspace_id] = updated
            self._persist()
            return deep_copy(updated)

    def duplicate_workspace(self, workspace_id: str) -> dict:
        original = self.get_workspace(workspace_id)
        if original is None:
            raise KeyError(workspace_id)
        duplicated = deep_copy(original)
        duplicated["workspace_id"] = make_id("workspace")
        duplicated["display_name"] = f"{original.get('display_name', 'Workspace')} (Copy)"
        duplicated["created_at"] = utc_now_iso()
        duplicated["last_opened_at"] = duplicated["created_at"]
        duplicated["archived"] = False
        duplicated["deleted"] = False
        duplicated["history"] = deep_copy(original.get("history", []))
        self.save_workspace(duplicated)
        return duplicated

    def archive_workspace(self, workspace_id: str) -> dict:
        return self.update_workspace(workspace_id, lambda workspace: {**workspace, "archived": True})

    def delete_workspace(self, workspace_id: str) -> None:
        def _delete(workspace: dict) -> dict:
            workspace["deleted"] = True
            return workspace

        self.update_workspace(workspace_id, _delete)

    def create_job(self, job: dict) -> dict:
        with self._lock:
            self._state.setdefault("jobs", {})[job["job_id"]] = deep_copy(job)
            self._persist()
            return deep_copy(job)

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._state.get("jobs", {}).get(job_id)
            if job is None:
                return None
            return deep_copy(job)

    def update_job(self, job_id: str, updater: Callable[[dict], dict | None]) -> dict:
        with self._lock:
            job = self._state.get("jobs", {}).get(job_id)
            if job is None:
                raise KeyError(job_id)
            updated = updater(deep_copy(job))
            if updated is None:
                updated = job
            self._state.setdefault("jobs", {})[job_id] = updated
            self._persist()
            return deep_copy(updated)

    def list_jobs_for_workspace(self, workspace_id: str) -> List[dict]:
        with self._lock:
            jobs = [deep_copy(job) for job in self._state.get("jobs", {}).values() if job.get("workspace_id") == workspace_id]
        jobs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return jobs

    def replace_state_for_testing(self, state: Dict[str, Any]) -> None:
        with self._lock:
            self._state = deep_copy(state)
            self._persist()
