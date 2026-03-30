02_content_service
==================

Local content-grounding service for lecture materials. This service is fully standalone and does not require Docker, a cloud service, an external database, or a vector database.

What it provides
----------------
- material import for PDF, PPTX, uploaded text files, and pasted text
- persistent local storage of originals, derived previews, normalized text, citations, annotations, and job state
- slide/page listing and slide/page inspection
- slide/page preview URLs and source-open URLs
- focused retrieval search with slide-level citations
- broader retrieval bundles for precision, coverage, full-material sweeps, and canonical evidence bundles
- stable citation resolution by citation_id
- workspace annotations for user corrections, study notes, focus boosts, and exclude-from-grounding directives
- backward-compatible import and annotation alias handling for integrated localhost app tolerance
- restart persistence using only local files plus SQLite
- machine-readable API docs at /openapi.json and interactive docs at /docs

Folder contract
---------------
Required folder: 02_content_service
Required launcher: run_local.py
Default bind address: 127.0.0.1
Default port: 38410

Environment variables
---------------------
- CONTENT_SERVICE_PORT, default 38410
- LOCAL_DATA_DIR, default ./local_data
- CONTENT_SERVICE_IMPORT_WORKERS, default 2
- LIBREOFFICE_BIN, default soffice
- CONTENT_SERVICE_VERSION, default 1.0.0

Root .env and shared local-data behavior
----------------------------------------
When this service lives inside the integrated sibling-folder project layout, launching 02_content_service/run_local.py automatically:
- reads the root project .env if present
- keeps already-set process environment variables as the highest-precedence override
- defaults LOCAL_DATA_DIR to the integrated project root ./local_data instead of a service-local folder

This makes direct standalone launch from 02_content_service behave the same as launcher-driven integrated startup by default.

Standalone launch
-----------------
1. Open a shell in this folder.
2. Optionally copy values from .env.example into the integrated project root .env or export them in your shell.
3. Start the service:

   python run_local.py

4. Verify:

   curl http://127.0.0.1:38410/healthz
   curl http://127.0.0.1:38410/manifest

5. Open docs in a browser:

   http://127.0.0.1:38410/docs
   http://127.0.0.1:38410/openapi.json

Notes on implementation
-----------------------
- SQLite stores metadata, slides, jobs, annotations, and citations.
- Files under LOCAL_DATA_DIR store originals, rendered previews, and normalized exports.
- PDF parsing and preview rendering use PyMuPDF.
- PPTX text extraction uses python-pptx.
- PPTX slide previews are rendered by converting the deck to PDF with LibreOffice and then rasterizing pages with PyMuPDF.
- If PPTX conversion fails, the service falls back to text-based preview images and marks quality accordingly.
- No OCR text is invented. If text cannot be extracted, the service preserves previews and reports lower extraction quality.
- POST /v1/evidence-bundles and POST /v1/retrieval/bundle share the same retrieval and citation logic and return the same bundle shape.
- Evidence-bundle requests may omit query_text or send query_text as null for lecture-wide coverage and full-material sweeps, matching retrieval-bundle null-query behavior.

Compatibility layer
-------------------
Import compatibility accepted in addition to the canonical contract:
- file present with missing source_kind -> inferred as file
- text or source_text -> treated as text_body
- missing source_kind with kind=pasted_text -> inferred as pasted_text

Annotation compatibility accepted when mapping is unambiguous:
- kind -> annotation_type for supported semantic types and straightforward aliases
- target_type -> scope for workspace, material, or slide
- target_id -> material_id or slide_id when the target_type makes that mapping obvious

Unsupported artifact-specific target types such as study_plan_item, practice_question, and chat_message return a clean 400.

Quick smoke test with bundled sample data
-----------------------------------------
Bundled smoke files live under sample_data/:
- sample_slides.pdf
- sample_deck.pptx
- sample_notes.md

Example 1: import a PDF deck
----------------------------
From this folder:

curl -X POST http://127.0.0.1:38410/v1/materials/import \
  -F workspace_id=demo_workspace \
  -F role=slides \
  -F source_kind=file \
  -F file=@sample_data/sample_slides.pdf

Expected response shape:
{
  "material_id": "<id>",
  "job_id": "<id>",
  "processing_status": "queued"
}

Poll the job:

curl http://127.0.0.1:38410/v1/jobs/<job_id>

List materials:

curl "http://127.0.0.1:38410/v1/materials?workspace_id=demo_workspace"

Inspect one material:

curl http://127.0.0.1:38410/v1/materials/<material_id>

List slides/pages for a material:

curl http://127.0.0.1:38410/v1/materials/<material_id>/slides

Open a slide preview URL in the browser or fetch it directly:

curl -I http://127.0.0.1:38410/v1/materials/<material_id>/slides/<slide_id>/preview

Open a slide source page in the browser:

http://127.0.0.1:38410/v1/materials/<material_id>/slides/<slide_id>/source

Example 2: import a PPTX deck
-----------------------------

curl -X POST http://127.0.0.1:38410/v1/materials/import \
  -F workspace_id=demo_workspace \
  -F role=slides \
  -F source_kind=file \
  -F file=@sample_data/sample_deck.pptx

Poll the returned job_id until status becomes succeeded or failed.

Example 3: import pasted notes
------------------------------

curl -X POST http://127.0.0.1:38410/v1/materials/import \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": "demo_workspace",
    "role": "notes",
    "source_kind": "pasted_text",
    "title": "Lecture notes",
    "text_body": "ATP is produced during cellular respiration. Chloroplasts drive photosynthesis."
  }'

Example 4: focused retrieval search
-----------------------------------

curl -X POST http://127.0.0.1:38410/v1/retrieval/search \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": "demo_workspace",
    "material_ids": ["<material_id_1>", "<material_id_2>"],
    "query_text": "Where does ATP come from?",
    "top_k": 6,
    "retrieval_mode": "coverage",
    "include_annotations": true,
    "min_extraction_quality": "low"
  }'

Example 5: retrieval bundle and canonical evidence bundle
---------------------------------------------------------

curl -X POST http://127.0.0.1:38410/v1/retrieval/bundle \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": "demo_workspace",
    "material_ids": ["<material_id_1>"],
    "query_text": "photosynthesis",
    "bundle_mode": "full_material",
    "token_budget": 0,
    "max_items": 20,
    "include_annotations": true
  }'

curl -X POST http://127.0.0.1:38410/v1/evidence-bundles \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": "demo_workspace",
    "material_ids": ["<material_id_1>"],
    "query_text": "photosynthesis",
    "bundle_mode": "full_material",
    "token_budget": 0,
    "max_items": 20,
    "include_annotations": true
  }'

For coverage or full-material sweeps without a user query, query_text may be omitted entirely or sent as null:

curl -X POST http://127.0.0.1:38410/v1/evidence-bundles \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": "demo_workspace",
    "material_ids": ["<material_id_1>"],
    "bundle_mode": "coverage",
    "token_budget": 0,
    "max_items": 20,
    "include_annotations": true
  }'

Example 6: resolve citations later by citation_id alone
-------------------------------------------------------

curl -X POST http://127.0.0.1:38410/v1/citations/resolve \
  -H "Content-Type: application/json" \
  -d '{
    "citation_ids": ["<citation_id_1>", "<citation_id_2>"]
  }'

Example 7: annotations
----------------------
Create a workspace note:

curl -X POST http://127.0.0.1:38410/v1/workspaces/demo_workspace/annotations \
  -H "Content-Type: application/json" \
  -d '{
    "annotation_type": "study_note",
    "scope": "workspace",
    "text": "ATP is the main energy currency of the cell."
  }'

Create a slide-specific exclusion:

curl -X POST http://127.0.0.1:38410/v1/workspaces/demo_workspace/annotations \
  -H "Content-Type: application/json" \
  -d '{
    "annotation_type": "exclude_from_grounding",
    "scope": "slide",
    "material_id": "<material_id>",
    "slide_id": "<slide_id>",
    "text": "Exclude this title slide from future grounding."
  }'

Legacy annotation compatibility example:

curl -X POST http://127.0.0.1:38410/v1/workspaces/demo_workspace/annotations \
  -H "Content-Type: application/json" \
  -d '{
    "kind": "focus",
    "target_type": "material",
    "target_id": "<material_id>",
    "text": "Prioritize this material during grounding."
  }'

List annotations:

curl http://127.0.0.1:38410/v1/workspaces/demo_workspace/annotations

Delete an annotation:

curl -X DELETE http://127.0.0.1:38410/v1/workspaces/demo_workspace/annotations/<annotation_id>

Running tests
-------------
From this folder:

pytest -q

Current automated test coverage includes:
- health and manifest
- canonical import for PDF, PPTX, and pasted text
- canonical annotations create/list/delete
- evidence-bundle endpoint equivalence with retrieval-bundle endpoint
- legacy import aliases
- legacy annotation aliases and clean failure cases
- slide preview/source URLs
- retrieval search
- retrieval bundle and evidence bundle
- citation resolution
- exclude-from-grounding behavior
- deletion semantics for material-scoped versus workspace-scoped annotations
- restart persistence
- failed-job behavior for bad uploads
- integrated root .env and shared LOCAL_DATA_DIR resolution

No Docker required
------------------
This service is intentionally local-first. It runs directly with Python and local system dependencies only.
