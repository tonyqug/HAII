03_learning_service

Learning-intelligence service for grounded study plans, grounded chat, practice generation, artifact revision, uncertainty handling, local persistence, and true background jobs.

What is included
- run_local.py
- learning_service/ FastAPI implementation
- .env.example
- sample_data/ standalone evidence bundles for local testing
- tests/ offline automated tests
- requirements.txt

Key behavior
- Binds only to 127.0.0.1
- Default port: 38420
- Exposes /healthz and /manifest
- Supports both grounding modes:
  - strict_lecture_only
  - lecture_with_fallback
- Supports both grounding input modes:
  - integrated mode via workspace_id + material_ids
  - standalone mode via workspace_id + evidence_bundle
- Persists study plans, conversations, practice sets, and jobs locally in LOCAL_DATA_DIR
- Keeps artifact history instead of mutating prior versions in place
- Returns explicit insufficient-evidence / clarification outcomes instead of fabricating certainty
- Executes generation and revision endpoints as real background jobs that return immediately with job_id
- Uses Gemini as the primary structured generation path when GEMINI_API_KEY is configured, with deterministic grounded fallback when Gemini is unavailable or validation fails
- Accepts canonical inputs plus the correction-spec alias inputs used by the shared app shell

Runtime compatibility
- The request layer no longer depends on fragile Pydantic validator behavior for alias handling.
- requirements.txt no longer pins an isolated Pydantic version that could conflict with the shared FastAPI stack.
- run_local.py launches cleanly in the shared local environment using the installed FastAPI/Pydantic combination.
- In the standard integrated sibling-folder layout, startup resolves configuration from the shared project-root .env by default.
- In that same integrated layout, a relative LOCAL_DATA_DIR such as ./local_data resolves to the shared project root, not the service folder.

Launch behavior
1) Standalone mode (no content service required)
- Use this mode when requests include evidence_bundle.
- The service does not require the content service to be running.
- If GEMINI_API_KEY is missing, /healthz reports ready=false because primary Gemini generation is not configured, but the service still starts and deterministic grounded fallback remains available for local development and testing.

2) Integrated mode (content service required)
- Use this mode when requests include material_ids.
- In the standard integrated sibling-folder layout (01_app_shell, 02_content_service, 03_learning_service), the service loads the shared project-root .env by default whether it is started directly or auto-launched by Team 1.
- In that integrated layout, LOCAL_DATA_DIR=./local_data resolves to the shared root local_data directory, so jobs and artifacts are written under the shared runtime store instead of 03_learning_service/local_data.
- The service calls CONTENT_SERVICE_URL to fetch an evidence bundle.
- Endpoint order for Team 2 compatibility:
  1. POST /v1/evidence-bundles
  2. POST /v1/retrieval/bundle
  3. a few older neighboring fallback paths kept only for compatibility
- If Team 2 returns the bundle directly, that payload is accepted as the canonical evidence bundle.
- If the content service is unreachable, integrated jobs fail with a stored job error instead of crashing the process.

Environment variables
Required by the spec
- LEARNING_SERVICE_PORT=38420
- CONTENT_SERVICE_URL=http://127.0.0.1:38410
- GEMINI_API_KEY=
- LOCAL_DATA_DIR=./local_data

Optional convenience variables used by this implementation
- GEMINI_MODEL=gemini-2.5-flash
- USE_HEURISTIC_FALLBACK=true
- LEARNING_SERVICE_REQUEST_TIMEOUT_SECONDS=15

Quick start
1. Install requirements:
   pip install -r requirements.txt
2. Copy the example env file if you want local overrides:
   cp .env.example .env
3. Start the service:
   python run_local.py
4. Check health:
   curl http://127.0.0.1:38420/healthz

Health semantics
- status=ok means the process is up.
- ready=true means Gemini is configured for the primary model-backed path.
- ready=false does not stop the service from running; it means the service is degraded to deterministic fallback.
- details explicitly report:
  - process_up
  - gemini_configured
  - content_service_reachable
  - integrated_mode_available
  - standalone_mode_available
  - deterministic_fallback_available
  - primary_generation_path

Asynchronous job flow
The following endpoints are true background-job endpoints:
- POST /v1/study-plans
- POST /v1/study-plans/{study_plan_id}/revise
- POST /v1/conversations/{conversation_id}/messages
- POST /v1/practice-sets
- POST /v1/practice-sets/{practice_set_id}/revise

Common API flow
1. POST a generation or revision request.
2. Receive {"job_id": "..."} immediately.
3. Poll GET /v1/jobs/{job_id} until the status is one of:
   - succeeded
   - failed
   - needs_user_input
4. If succeeded, use result_id to read the artifact.

Alias-tolerant request normalization
In addition to canonical shapes, the service accepts these app-shell compatibility aliases:
- grounded creation:
  - included_material_ids minus excluded_material_ids can derive material_ids
  - focused_material_ids is treated as advisory unless an explicit focus-only flag is present
- study-plan creation:
  - student_context.known -> student_context.prior_knowledge
- study-plan revision:
  - feedback_note -> instruction_text
  - locked_sections expands to the full corresponding prior-plan section ids
  - missing instruction_text is safely synthesized
- conversation message:
  - text -> message_text
  - response_style=direct_answer -> standard
- practice-set creation:
  - difficulty -> difficulty_profile
  - answer_key -> include_answer_key
  - rubric -> include_rubrics
- practice-set revision:
  - selected_question_ids -> target_question_ids
  - missing instruction_text is safely synthesized

Standalone example request using evidence_bundle
The file sample_data/standalone_evidence_bundle.json is a canonical evidence bundle that can be posted directly.

python - <<'PY'
import json
from pathlib import Path
import requests

bundle = json.loads(Path('sample_data/standalone_evidence_bundle.json').read_text())
payload = {
    "workspace_id": bundle["workspace_id"],
    "material_ids": None,
    "evidence_bundle": bundle,
    "topic_text": "regularization and validation",
    "time_budget_minutes": 90,
    "grounding_mode": "lecture_with_fallback",
    "student_context": {
        "prior_knowledge": "basic algebra",
        "weak_areas": "validation",
        "goals": "prepare for the midterm"
    },
    "include_annotations": True
}
resp = requests.post('http://127.0.0.1:38420/v1/study-plans', json=payload, timeout=30)
print(resp.json())
PY

Then poll the job:
curl http://127.0.0.1:38420/v1/jobs/<job_id>

Standalone template_mimic example
Use sample_data/standalone_template_mimic_bundle.json and set:
- generation_mode=template_mimic
- template_material_id=mat_template_midterm_1

Implemented endpoints
- GET /healthz
- GET /manifest
- GET /v1/jobs/{job_id}
- POST /v1/study-plans
- GET /v1/study-plans?workspace_id=...
- GET /v1/study-plans/{study_plan_id}
- POST /v1/study-plans/{study_plan_id}/revise
- POST /v1/conversations
- GET /v1/conversations?workspace_id=...
- GET /v1/conversations/{conversation_id}
- POST /v1/conversations/{conversation_id}/messages
- POST /v1/conversations/{conversation_id}/clear
- POST /v1/practice-sets
- GET /v1/practice-sets?workspace_id=...
- GET /v1/practice-sets/{practice_set_id}
- POST /v1/practice-sets/{practice_set_id}/revise

Persistence model
LOCAL_DATA_DIR contains JSON files under:
- jobs/
- study_plans/
- conversations/
- practice_sets/

Artifacts remain readable after restart by their IDs. Revision endpoints create new artifacts with parent_* lineage fields instead of overwriting prior versions.

Testing
Run the offline test suite with:
pytest -q

Coverage in the tests
- startup/import behavior in the shared runtime
- clear health + manifest payloads
- real background-job submission for every job endpoint
- canonical request shapes
- legacy alias request shapes
- shared-root startup and LOCAL_DATA_DIR resolution in the integrated layout
- Team 2 evidence-bundle fetch via /v1/evidence-bundles and /v1/retrieval/bundle
- integrated practice generation robustness for short_answer, long_answer, and template_mimic
- integrated conversation flow with polling, read, clear, and reuse
- study-plan generation and revision lock semantics
- practice generation and revision lock semantics
- strict-mode honesty behavior
- standalone mode without a live content service
- restart persistence
- Gemini primary-path usage with citation validation/fallback protection

No Docker, external database, or deployment infrastructure is required.
