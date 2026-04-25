STUDY HELPER MVP — 01_app_shell

This folder contains the browser-facing application shell for the Study Helper MVP. It is the only end-user entry point for the localhost app. The shell owns:
- workspace library and lifecycle
- durable local persistence under LOCAL_DATA_DIR/app_shell
- material import and ingestion-progress UI
- grounded chat, practice, history, and source-viewer UI
- integrated launcher / degraded-mode handling for sibling localhost services
- canonical request normalization between the browser UI and the sibling services

REQUIRED ENTRY POINT

Run the shell with one local command:

  python 01_app_shell/run_local.py

Default bind address and port:
  http://127.0.0.1:38400

Required shell endpoints:
- GET /healthz
- GET /manifest

ROOT .env BEHAVIOR

The shell reads the shared root .env file from the parent directory of 01_app_shell.
Expected integrated sibling-folder layout:
  <project root>/
    .env
    01_app_shell/
    02_content_service/
    03_learning_service/
    local_data/   (optional, auto-created)

Launcher contract folder names are fixed and required:
- 01_app_shell
- 02_content_service
- 03_learning_service

Supported environment variables:
- APP_SHELL_PORT (default 38400)
- CONTENT_SERVICE_PORT (recommended 38410 for sibling service startup consistency)
- LEARNING_SERVICE_PORT (recommended 38420 for sibling service startup consistency)
- CONTENT_SERVICE_URL (default http://127.0.0.1:38410)
- LEARNING_SERVICE_URL (default http://127.0.0.1:38420)
- LOCAL_DATA_DIR (default ./local_data)
- AUTO_OPEN_BROWSER (default true)
- APP_SHELL_MODE (auto | integrated, default auto; mock is test-only when APP_SHELL_TESTING=true)

The overall project secret GEMINI_API_KEY belongs to the learning service. The shell does not require it directly.

LAUNCH MODES

1. mock (test-only)
   APP_SHELL_MODE=mock python 01_app_shell/run_local.py
   Available only when APP_SHELL_TESTING=true. Uses fixture data only and demonstrates a ready workspace, chat thread, practice set, and working source viewer.

2. integrated
   APP_SHELL_MODE=integrated python 01_app_shell/run_local.py
   Requires real localhost sibling services. If a sibling folder exists beside 01_app_shell, the shell attempts to launch:
   - ../02_content_service/run_local.py
   - ../03_learning_service/run_local.py
   If a sibling still does not become healthy, the shell remains available and clearly labels the unavailable features.

3. auto
   APP_SHELL_MODE=auto python 01_app_shell/run_local.py
   Tries the integrated launch path and keeps the UI honest about service availability. It does not silently fall back to mock mode.

INTEGRATED ONE-COMMAND STARTUP

From project root or from inside 01_app_shell:

  python 01_app_shell/run_local.py

Expected integrated startup sequence:
1) load root .env from the shared project root
2) check Team 2 health at http://127.0.0.1:38410/healthz
3) if unhealthy and sibling folder exists, start python 02_content_service/run_local.py
4) wait for healthy or clearly unavailable
5) check Team 3 health at http://127.0.0.1:38420/healthz
6) if unhealthy and sibling folder exists, start python 03_learning_service/run_local.py
7) wait for healthy or clearly unavailable
8) start app shell at http://127.0.0.1:38400
9) optionally open the default browser automatically

DEGRADED-MODE EXPECTATIONS

- If Team 2 content service is missing/unhealthy:
  - app shell still launches
  - source-grounded/material-import features are marked unavailable
  - shell remains renderable for local review

- If Team 3 learning service is missing/unhealthy:
  - app shell still launches
  - content flows can still work when Team 2 is healthy
  - chat/practice actions are disabled with a clear explanation

- If only app shell is present:
  - APP_SHELL_MODE=auto keeps the shell live, but unavailable features remain disabled instead of swapping to mock behavior

CORRECTED BOUNDARY NORMALIZATION

The shell now acts as the canonical boundary-normalization layer required by the correction spec.
It converts UI payloads into the sibling-service contracts instead of forwarding raw browser state.

Important corrected mappings:
- material import
  file uploads -> workspace_id, role, source_kind=file, title, file
  pasted text -> workspace_id, role, source_kind=pasted_text, title, text_body
- conversation creation
  workspace_id, material_ids, grounding_mode, title, include_annotations=true
- conversation message
  message_text, response_style, grounding_mode, include_citations=true
- practice generation
  workspace_id, material_ids, generation_mode, template_material_id when applicable,
  question_count, coverage_mode, difficulty_profile, include_answer_key,
  include_rubrics, grounding_mode, include_annotations=true
- practice revision
  instruction_text, target_question_ids, locked_question_ids, maintain_coverage

MATERIAL SELECTION POLICY

For every generation request:
- excluded materials are never sent in material_ids
- only ready materials are sent in material_ids
- if some materials are still processing, the shell proceeds with the ready subset and surfaces a warning
- if no materials are ready, the shell blocks the request with a clear human-readable error
- focus annotations do not silently switch the request into “focus only” mode

JOB-BASED CHAT FLOW

The shell does not assume chat POST returns an assistant message.
Instead it:
- submits the canonical message request
- receives a job_id
- keeps the user’s pending message in local workspace state
- polls /api/jobs/{job_id}
- refreshes the workspace when the job succeeds
- surfaces failed and needs_user_input states without losing the conversation

ANNOTATION SYNCHRONIZATION

The shell hydrates remote Team 2 annotations during workspace refresh and merges them without duplication.
Material-preference UI maps to canonical Team 2 annotations:
- focus -> annotation_type=focus, scope=material
- exclude -> annotation_type=exclude_from_grounding, scope=material
- default -> removes stale focus/exclude annotations

Human-in-the-loop correction notes are always kept in local shell history.
When the note should affect future grounding, the shell also creates a canonical Team 2 annotation at the narrowest valid scope the note supports.

LOCAL PERSISTENCE

The shell persists durable workspace state in:
  LOCAL_DATA_DIR/app_shell/state.json

Persisted shell-owned state includes:
- workspace metadata and library state
- material cache and preferences
- known practice sets and conversations
- active artifact pointers
- local feedback history
- local pending jobs and last-known job status
- local UI preferences and unsent edit state containers
- last successful sync timestamps and sibling-service availability snapshots

DEPENDENCIES

The shell runtime dependencies are listed in requirements.txt.
The integration-facing version note is in integration_dependencies.txt.
Validated shell runtime:
- Python 3.11+
- FastAPI 0.111.0
- Uvicorn 0.30.1
- HTTPX 0.27.0
- python-multipart 0.0.9

TESTING

Run the automated tests with:

  cd 01_app_shell
  pytest -q

The test suite covers:
- health, manifest, and root UI
- mock fixture bootstrapping and source viewer previews
- durable local persistence across restart
- canonical content-service import request normalization
- canonical learning-service request normalization
- job-based chat polling and conversation persistence
- annotation hydration and focus/exclude sync behavior
- correction-note annotation sync behavior
- degraded integrated-mode error handling without fake success objects
- one-command run_local startup

NO EXTRA PLATFORM REQUIREMENTS

This deliverable does not require:
- Docker
- Node
- any external database
