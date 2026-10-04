"""Run the Black Box API: ``python -m server`` (same as ``make dev-api`` without reload)."""

import os

import uvicorn

uvicorn.run(
    "server.app:app",
    host=os.environ.get("HOST", "127.0.0.1"),
    port=int(os.environ.get("PORT", "8000")),
)
