from __future__ import annotations

import logging
import uvicorn

from learning_service.app import create_app
from learning_service.config import Settings


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    settings = Settings.from_env()
    settings.local_data_dir.mkdir(parents=True, exist_ok=True)
    uvicorn.run(
        create_app(settings),
        host="127.0.0.1",
        port=settings.port,
        log_level="info",
    )
