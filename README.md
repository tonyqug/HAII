# Study Helper MVP

Study Helper MVP is a local-first AI study buddy for students. The final shipped product is intentionally narrow and centered on two user-visible workflows:

- generating grounded practice tests from uploaded course materials
- asking grounded follow-up questions about those materials

The app is general-purpose for student study workflows, but our demo and pilot use introductory machine learning materials as the concrete example.

## Final visible functionality

- Create a workspace for a course or exam.
- Upload lecture slides or notes as PDF, PPTX, or pasted text.
- See material import and processing progress in the UI.
- Generate a practice test from the uploaded materials after reviewing the request settings.
- Revise a practice test by keeping strong questions and regenerating weaker ones.
- Use the Ask feature to ask grounded questions about the current materials.
- Inspect citations and open the exact source slide/page used for grounding.
- Review prior practice sets and chats in the workspace history.

The final UI does not use the earlier study-plan concept. The final demonstrated workflow is the narrower Practice + Ask experience above.

## Repository structure

- `01_app_shell/`: browser UI, workspace flow, upload progress, practice/ask tabs, history view, and integrated launcher
- `02_content_service/`: material import, preview generation, retrieval, citation resolution, and source viewing support
- `03_learning_service/`: grounded practice generation, grounded Ask conversations, revision flow, and background jobs
- `local_data/`: local runtime storage created or reused by the app

## Open-source code and libraries used

This repository does not vendor a separate starter application or copied third-party app codebase. The project code in `01_app_shell`, `02_content_service`, and `03_learning_service` was implemented for this project.

We do rely on open-source libraries and APIs, including:

- `FastAPI` and `Uvicorn` for the local web services
- `HTTPX` and `requests` for service-to-service communication
- `PyMuPDF`, `python-pptx`, `Pillow`, and `reportlab` for document parsing and preview generation
- `python-multipart` for file upload handling
- `SQLite` plus local JSON files for persistence
- the `Gemini API` for the primary generation path

## What we changed and what we implemented

All application code in this repository is project code rather than imported starter app code.

Main work implemented in this project:

- the multi-service local architecture for the app shell, content service, and learning service
- workspace creation and persistent local state
- material upload, processing, and status feedback in the UI
- grounded practice-test generation with visible request settings and human approval before generation
- selective practice revision so users can keep good questions and regenerate weaker ones
- grounded Ask conversations tied to the uploaded materials
- citation chips, source viewer, and artifact history so outputs stay inspectable
- degraded-mode handling and background-job polling so the UI stays honest about system state

Compared with earlier versions of the project, we deliberately narrowed the final product to the two workflows that were most reliable and easiest to evaluate end to end: Practice and Ask.

## How to run

1. Create a root `.env` file:

   ```env
   GEMINI_API_KEY=your_key_here
   APP_SHELL_PORT=38400
   CONTENT_SERVICE_PORT=38410
   LEARNING_SERVICE_PORT=38420
   CONTENT_SERVICE_URL=http://127.0.0.1:38410
   LEARNING_SERVICE_URL=http://127.0.0.1:38420
   LOCAL_DATA_DIR=./local_data
   AUTO_OPEN_BROWSER=true
   APP_SHELL_MODE=auto
   ```

2. Install dependencies:

   ```powershell
   python -m pip install -r "01_app_shell/requirements.txt"
   python -m pip install -r "02_content_service/requirements.txt"
   python -m pip install -r "03_learning_service/requirements.txt"
   ```

3. Start the integrated app from the project root:

   ```powershell
   python 01_app_shell/run_local.py
   ```

4. Open [http://127.0.0.1:38400](http://127.0.0.1:38400)

## Notes

- Keep the sibling folder names exactly as `01_app_shell`, `02_content_service`, and `03_learning_service` because the launcher expects that layout.
- The app shell auto-starts the sibling services when needed.
- PDF and pasted-text imports work directly. PPTX support is included; slide preview rendering is best when LibreOffice (`soffice`) is available locally.
- No Docker, Node, or external database is required for local use.
