# Study Helper Localhost App

Minimal local setup:

1. Add your Gemini API key to the root `.env`:

   `GEMINI_API_KEY=your_key_here`
   and the rest of the env

APP_SHELL_PORT=38400
CONTENT_SERVICE_PORT=38410
LEARNING_SERVICE_PORT=38420
CONTENT_SERVICE_URL=http://127.0.0.1:38410
LEARNING_SERVICE_URL=http://127.0.0.1:38420
LOCAL_DATA_DIR=./local_data
AUTO_OPEN_BROWSER=true
APP_SHELL_MODE=auto

2. Install dependencies:

   ```powershell
   python -m pip install -r "01_app_shell/requirements.txt"
   python -m pip install -r "02_content_service/requirements.txt"
   python -m pip install -r "03_learning_service/requirements.txt"
   ```

3. Run the integrated app from the project root:

   ```powershell
   python 01_app_shell/run_local.py
   ```

4. Open:

   [http://127.0.0.1:38400](http://127.0.0.1:38400)

Notes:
- Keep the sibling folder names exactly as:
  - `01_app_shell`
  - `02_content_service`
  - `03_learning_service`
- The launcher auto-starts sibling services when needed.
