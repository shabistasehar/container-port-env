"""
FastAPI app for Container Port Environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

from openenv.core.env_server import create_web_interface_app
import uvicorn

from models import ContainerAction, ContainerObservation
from server.environment import ContainerYardEnvironment

app = create_web_interface_app(
    ContainerYardEnvironment,
    ContainerAction,
    ContainerObservation,
    env_name="container-port-env",
)


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
