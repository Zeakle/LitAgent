FROM python:3.11-slim
LABEL litagent.sandbox=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g \
    @modelcontextprotocol/server-filesystem \
    @modelcontextprotocol/server-memory

ENV npm_config_cache=/tmp/.npm \
    UV_CACHE_DIR=/tmp/.uv \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /bin/bash sandbox
USER sandbox
WORKDIR /workspace
ENTRYPOINT []