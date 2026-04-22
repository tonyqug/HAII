from __future__ import annotations

import logging
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from app_shell.main import create_app  # noqa: E402
from app_shell.config import AppConfig  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    config = AppConfig.load(THIS_DIR)
    app = create_app()

    if config.auto_open_browser and not config.testing:
        threading.Timer(1.0, lambda: webbrowser.open(config.ui_base_url)).start()

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
