FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Dependency layer: resolved from the lockfile, cached until the lock/manifest changes.
COPY pyproject.toml uv.lock README.md ./
COPY nexus/mcp nexus/mcp
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --all-extras --group examples

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-extras --group examples

ENV TEMPORAL_CONFIG_FILE=docker/temporal.docker.toml \
    TEMPORAL_PROFILE=default \
    LARGE_PAYLOAD_DRIVER=local \
    LARGE_PAYLOAD_DIR=/large-payloads
