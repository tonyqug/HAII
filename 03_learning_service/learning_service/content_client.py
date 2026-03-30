from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from .config import Settings


class ContentServiceError(RuntimeError):
    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class ContentServiceClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.content_service_url.rstrip("/")
        self.timeout = settings.request_timeout_seconds

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/healthz", timeout=min(3, self.timeout))
            return response.status_code == 200
        except Exception:
            return False

    def _extract_bundle(self, payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        candidate = payload
        if isinstance(payload.get("evidence_bundle"), dict):
            candidate = payload["evidence_bundle"]
        elif isinstance(payload.get("bundle"), dict):
            candidate = payload["bundle"]
        if isinstance(candidate, dict) and candidate.get("bundle_id") and isinstance(candidate.get("items"), list):
            return candidate
        return None

    def _payload_variants(
        self,
        *,
        workspace_id: str,
        material_ids: list[str],
        query_text: Optional[str],
        bundle_mode: str,
        include_annotations: bool,
    ) -> list[dict[str, Any]]:
        base_payload: dict[str, Any] = {
            "workspace_id": workspace_id,
            "material_ids": material_ids,
            "query_text": query_text,
            "bundle_mode": bundle_mode,
            "include_annotations": include_annotations,
        }
        variants = [base_payload]
        if query_text is None:
            variants.append({key: value for key, value in base_payload.items() if key != "query_text"})
        return variants

    def _response_detail(self, response: Any) -> Any:
        try:
            return response.json()
        except Exception:
            return response.text

    def fetch_evidence_bundle(
        self,
        *,
        workspace_id: str,
        material_ids: list[str],
        query_text: Optional[str],
        bundle_mode: str,
        include_annotations: bool,
    ) -> Dict[str, Any]:
        candidate_paths = [
            "/v1/evidence-bundles",
            "/v1/retrieval/bundle",
            "/v1/evidence/bundles",
            "/v1/retrieval/evidence-bundles",
            "/v1/retrieve/evidence-bundle",
            "/v1/materials/evidence-bundle",
        ]
        # Respect the configured timeout so evidence retrieval has enough time
        # on larger workspaces; keep only a small safety floor.
        request_timeout = max(float(self.timeout), 0.2)
        compatibility_retry_statuses = {400, 404, 405, 409, 415, 422}
        payload_variants = self._payload_variants(
            workspace_id=workspace_id,
            material_ids=material_ids,
            query_text=query_text,
            bundle_mode=bundle_mode,
            include_annotations=include_annotations,
        )

        errors: list[str] = []
        retryable = False

        for path in candidate_paths:
            url = f"{self.base_url}{path}"
            for payload in payload_variants:
                payload_label = "without_query_text" if "query_text" not in payload else "with_query_text"
                try:
                    response = requests.post(url, json=payload, timeout=request_timeout)
                except requests.RequestException as exc:
                    raise ContentServiceError(
                        f"Content service request failed while contacting {path}: {exc}",
                        retryable=True,
                    ) from exc

                if 200 <= response.status_code < 300:
                    try:
                        data = response.json()
                    except Exception as exc:
                        errors.append(f"{path} [{payload_label}]: invalid JSON ({exc})")
                        continue

                    bundle = self._extract_bundle(data)
                    if bundle is not None:
                        return bundle
                    errors.append(f"{path} [{payload_label}]: unexpected evidence bundle payload")
                    continue

                detail = self._response_detail(response)
                errors.append(f"{path} [{payload_label}]: {response.status_code} {detail}")
                if response.status_code >= 500:
                    retryable = True
                if response.status_code not in compatibility_retry_statuses and response.status_code < 500:
                    # Keep trying compatibility paths in case another endpoint accepts the same request.
                    continue

        if errors:
            raise ContentServiceError(
                "Content service evidence bundle request failed across compatible endpoints. "
                f"Tried: {'; '.join(errors)}",
                retryable=retryable,
            )
        raise ContentServiceError("Content service evidence bundle request failed", retryable=True)
