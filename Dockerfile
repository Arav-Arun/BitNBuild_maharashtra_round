FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
COPY blackbox ./blackbox
COPY agents ./agents
COPY server ./server
RUN uv sync --locked --no-dev --extra ml --extra server
# The dataset is git-ignored and ~770 MB raw. Point DATA_URL at a .tar.gz of data/
# (see README Deployment) to bake it into the image; without it the API starts degraded.
ARG DATA_URL=""
COPY scripts/fetch_data.py ./scripts/fetch_data.py
RUN if [ -n "$DATA_URL" ]; then .venv/bin/python scripts/fetch_data.py "$DATA_URL" /app/data; fi
ENV DATA_DIR=/app/data MODE=recorded PORT=8000
EXPOSE 8000
# Shell form so Render's $PORT is expanded.
CMD uv run --no-sync uvicorn server.app:app --host 0.0.0.0 --port ${PORT}
