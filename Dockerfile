FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
COPY blackbox ./blackbox
COPY agents ./agents
COPY server ./server
RUN uv sync --locked --no-dev --extra ml --extra server
ENV DATA_DIR=/app/data MODE=recorded PORT=8000
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
