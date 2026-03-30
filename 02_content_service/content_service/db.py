from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import Settings


def connect(settings: Settings) -> sqlite3.Connection:
    settings.local_data_dir.mkdir(parents=True, exist_ok=True)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


@contextmanager
def get_conn(settings: Settings) -> Iterator[sqlite3.Connection]:
    conn = connect(settings)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(settings: Settings) -> None:
    settings.local_data_dir.mkdir(parents=True, exist_ok=True)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    with get_conn(settings) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS materials (
                material_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                title TEXT NOT NULL,
                original_filename TEXT,
                role TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                processing_status TEXT NOT NULL,
                page_count INTEGER NOT NULL DEFAULT 0,
                slide_count INTEGER NOT NULL DEFAULT 0,
                ready_for_retrieval INTEGER NOT NULL DEFAULT 0,
                previews_available INTEGER NOT NULL DEFAULT 0,
                quality_overall TEXT NOT NULL DEFAULT 'low',
                quality_notes TEXT,
                extraction_notes TEXT,
                source_relpath TEXT,
                converted_pdf_relpath TEXT,
                is_hidden INTEGER NOT NULL DEFAULT 0,
                failure_code TEXT,
                failure_message TEXT
            );

            CREATE TABLE IF NOT EXISTS slides (
                slide_id TEXT PRIMARY KEY,
                material_id TEXT NOT NULL,
                slide_number INTEGER NOT NULL,
                title_guess TEXT,
                extracted_text TEXT,
                extraction_quality TEXT NOT NULL,
                quality_notes TEXT,
                has_text INTEGER NOT NULL DEFAULT 0,
                preview_relpath TEXT,
                token_count INTEGER NOT NULL DEFAULT 0,
                text_checksum TEXT,
                FOREIGN KEY(material_id) REFERENCES materials(material_id) ON DELETE CASCADE
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_slides_material_slide_number ON slides(material_id, slide_number);
            CREATE INDEX IF NOT EXISTS idx_slides_material_id ON slides(material_id);

            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                stage TEXT NOT NULL,
                message TEXT NOT NULL,
                result_type TEXT,
                result_id TEXT,
                error_code TEXT,
                error_message TEXT,
                error_retryable INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS annotations (
                annotation_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                annotation_type TEXT NOT NULL,
                scope TEXT NOT NULL,
                material_id TEXT,
                slide_id TEXT,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                virtual_slide_number INTEGER NOT NULL,
                preview_relpath TEXT,
                FOREIGN KEY(material_id) REFERENCES materials(material_id) ON DELETE CASCADE,
                FOREIGN KEY(slide_id) REFERENCES slides(slide_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_annotations_workspace ON annotations(workspace_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_annotations_scope ON annotations(workspace_id, scope, annotation_type);

            CREATE TABLE IF NOT EXISTS citations (
                citation_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                material_id TEXT,
                slide_id TEXT,
                slide_number INTEGER,
                annotation_id TEXT,
                snippet_text TEXT NOT NULL,
                support_type TEXT NOT NULL,
                confidence TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(material_id) REFERENCES materials(material_id) ON DELETE CASCADE,
                FOREIGN KEY(slide_id) REFERENCES slides(slide_id) ON DELETE CASCADE,
                FOREIGN KEY(annotation_id) REFERENCES annotations(annotation_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_citations_workspace ON citations(workspace_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_citations_material_slide ON citations(material_id, slide_id);
            """
        )
