from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from .config import Settings
from .db import get_conn
from .utils import now_iso


ROLE_VALUES = {"slides", "notes", "practice_template"}
ANNOTATION_TYPES = {"user_correction", "study_note", "focus", "exclude_from_grounding"}
SCOPE_VALUES = {"workspace", "material", "slide"}
PROCESSING_VALUES = {"queued", "running", "ready", "failed"}


def row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}


class Repository:
    def __init__(self, settings: Settings):
        self.settings = settings

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with get_conn(self.settings) as conn:
            conn.execute(sql, params)

    def reserve_material(
        self,
        *,
        material_id: str,
        workspace_id: str,
        title: str,
        original_filename: Optional[str],
        role: str,
        kind: str,
        source_kind: str,
    ) -> None:
        ts = now_iso()
        with get_conn(self.settings) as conn:
            conn.execute(
                """
                INSERT INTO materials (
                    material_id, workspace_id, title, original_filename, role, kind, source_kind,
                    created_at, updated_at, processing_status, page_count, slide_count,
                    ready_for_retrieval, previews_available, quality_overall
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0, 0, 0, 0, 'low')
                """,
                (material_id, workspace_id, title, original_filename, role, kind, source_kind, ts, ts),
            )

    def update_material(self, material_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = now_iso()
        columns = ", ".join(f"{name} = ?" for name in fields.keys())
        values = list(fields.values()) + [material_id]
        with get_conn(self.settings) as conn:
            conn.execute(f"UPDATE materials SET {columns} WHERE material_id = ?", values)

    def get_material(self, material_id: str) -> Optional[dict[str, Any]]:
        with get_conn(self.settings) as conn:
            row = conn.execute("SELECT * FROM materials WHERE material_id = ?", (material_id,)).fetchone()
        return row_to_dict(row)

    def list_materials(self, workspace_id: str, include_hidden: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM materials WHERE workspace_id = ?"
        params: list[Any] = [workspace_id]
        if not include_hidden:
            sql += " AND is_hidden = 0"
        sql += " ORDER BY created_at ASC, material_id ASC"
        with get_conn(self.settings) as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [row_to_dict(row) for row in rows if row is not None]

    def delete_material(self, material_id: str) -> Optional[dict[str, Any]]:
        material = self.get_material(material_id)
        if material is None:
            return None
        with get_conn(self.settings) as conn:
            conn.execute(
                "DELETE FROM annotations WHERE material_id = ? OR slide_id IN (SELECT slide_id FROM slides WHERE material_id = ?)",
                (material_id, material_id),
            )
            conn.execute("DELETE FROM materials WHERE material_id = ?", (material_id,))
        return material

    def replace_slides(self, material_id: str, slides: list[dict[str, Any]]) -> None:
        with get_conn(self.settings) as conn:
            conn.execute("DELETE FROM slides WHERE material_id = ?", (material_id,))
            conn.executemany(
                """
                INSERT INTO slides (
                    slide_id, material_id, slide_number, title_guess, extracted_text,
                    extraction_quality, quality_notes, has_text, preview_relpath,
                    token_count, text_checksum
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        slide["slide_id"],
                        material_id,
                        slide["slide_number"],
                        slide.get("title_guess"),
                        slide.get("extracted_text", ""),
                        slide["extraction_quality"],
                        slide.get("quality_notes"),
                        1 if slide.get("has_text") else 0,
                        slide.get("preview_relpath"),
                        int(slide.get("token_count", 0)),
                        slide.get("text_checksum"),
                    )
                    for slide in slides
                ],
            )

    def list_slides(self, material_id: str) -> list[dict[str, Any]]:
        with get_conn(self.settings) as conn:
            rows = conn.execute(
                "SELECT * FROM slides WHERE material_id = ? ORDER BY slide_number ASC",
                (material_id,),
            ).fetchall()
        return [row_to_dict(row) for row in rows if row is not None]

    def get_slide(self, material_id: str, slide_id: str) -> Optional[dict[str, Any]]:
        with get_conn(self.settings) as conn:
            row = conn.execute(
                "SELECT * FROM slides WHERE material_id = ? AND slide_id = ?",
                (material_id, slide_id),
            ).fetchone()
        return row_to_dict(row)

    def get_slide_by_id(self, slide_id: str) -> Optional[dict[str, Any]]:
        with get_conn(self.settings) as conn:
            row = conn.execute("SELECT * FROM slides WHERE slide_id = ?", (slide_id,)).fetchone()
        return row_to_dict(row)

    def resolve_slide_scope(self, workspace_id: str, slide_id: str) -> Optional[Tuple[str, str]]:
        with get_conn(self.settings) as conn:
            row = conn.execute(
                """
                SELECT s.material_id, s.slide_id
                FROM slides AS s
                JOIN materials AS m ON m.material_id = s.material_id
                WHERE s.slide_id = ? AND m.workspace_id = ? AND m.is_hidden = 0
                """,
                (slide_id, workspace_id),
            ).fetchone()
        if row is None:
            return None
        return str(row["material_id"]), str(row["slide_id"])

    def reserve_job(
        self,
        *,
        job_id: str,
        job_type: str,
        stage: str,
        message: str,
        result_type: Optional[str],
        result_id: Optional[str],
    ) -> None:
        ts = now_iso()
        with get_conn(self.settings) as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, job_type, status, progress, stage, message,
                    result_type, result_id, error_code, error_message, error_retryable,
                    created_at, updated_at
                ) VALUES (?, ?, 'queued', 0, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
                """,
                (job_id, job_type, stage, message, result_type, result_id, ts, ts),
            )

    def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = now_iso()
        columns = ", ".join(f"{name} = ?" for name in fields.keys())
        values = list(fields.values()) + [job_id]
        with get_conn(self.settings) as conn:
            conn.execute(f"UPDATE jobs SET {columns} WHERE job_id = ?", values)

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with get_conn(self.settings) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return row_to_dict(row)

    def next_annotation_slide_number(self, workspace_id: str) -> int:
        with get_conn(self.settings) as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(virtual_slide_number), 0) + 1 AS next_num FROM annotations WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
        return int(row["next_num"]) if row else 1

    def create_annotation(
        self,
        *,
        annotation_id: str,
        workspace_id: str,
        annotation_type: str,
        scope: str,
        material_id: Optional[str],
        slide_id: Optional[str],
        text: str,
        virtual_slide_number: int,
        preview_relpath: Optional[str],
    ) -> None:
        ts = now_iso()
        with get_conn(self.settings) as conn:
            conn.execute(
                """
                INSERT INTO annotations (
                    annotation_id, workspace_id, annotation_type, scope, material_id, slide_id,
                    text, created_at, virtual_slide_number, preview_relpath
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    annotation_id,
                    workspace_id,
                    annotation_type,
                    scope,
                    material_id,
                    slide_id,
                    text,
                    ts,
                    virtual_slide_number,
                    preview_relpath,
                ),
            )

    def get_annotation(self, workspace_id: str, annotation_id: str) -> Optional[dict[str, Any]]:
        with get_conn(self.settings) as conn:
            row = conn.execute(
                "SELECT * FROM annotations WHERE workspace_id = ? AND annotation_id = ?",
                (workspace_id, annotation_id),
            ).fetchone()
        return row_to_dict(row)

    def list_annotations(self, workspace_id: str) -> list[dict[str, Any]]:
        with get_conn(self.settings) as conn:
            rows = conn.execute(
                "SELECT * FROM annotations WHERE workspace_id = ? ORDER BY created_at ASC, annotation_id ASC",
                (workspace_id,),
            ).fetchall()
        return [row_to_dict(row) for row in rows if row is not None]

    def delete_annotation(self, workspace_id: str, annotation_id: str) -> Optional[dict[str, Any]]:
        annotation = self.get_annotation(workspace_id, annotation_id)
        if annotation is None:
            return None
        with get_conn(self.settings) as conn:
            conn.execute(
                "DELETE FROM annotations WHERE workspace_id = ? AND annotation_id = ?",
                (workspace_id, annotation_id),
            )
        return annotation

    def upsert_citation(
        self,
        *,
        citation_id: str,
        workspace_id: str,
        source_type: str,
        material_id: Optional[str],
        slide_id: Optional[str],
        slide_number: Optional[int],
        annotation_id: Optional[str],
        snippet_text: str,
        support_type: str,
        confidence: str,
    ) -> None:
        ts = now_iso()
        with get_conn(self.settings) as conn:
            conn.execute(
                """
                INSERT INTO citations (
                    citation_id, workspace_id, source_type, material_id, slide_id,
                    slide_number, annotation_id, snippet_text, support_type,
                    confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(citation_id) DO UPDATE SET
                    workspace_id = excluded.workspace_id,
                    source_type = excluded.source_type,
                    material_id = excluded.material_id,
                    slide_id = excluded.slide_id,
                    slide_number = excluded.slide_number,
                    annotation_id = excluded.annotation_id,
                    snippet_text = excluded.snippet_text,
                    support_type = excluded.support_type,
                    confidence = excluded.confidence
                """,
                (
                    citation_id,
                    workspace_id,
                    source_type,
                    material_id,
                    slide_id,
                    slide_number,
                    annotation_id,
                    snippet_text,
                    support_type,
                    confidence,
                    ts,
                ),
            )

    def get_citations(self, citation_ids: Iterable[str]) -> list[dict[str, Any]]:
        ids = list(citation_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with get_conn(self.settings) as conn:
            rows = conn.execute(
                f"SELECT * FROM citations WHERE citation_id IN ({placeholders}) ORDER BY created_at ASC",
                tuple(ids),
            ).fetchall()
        return [row_to_dict(row) for row in rows if row is not None]
