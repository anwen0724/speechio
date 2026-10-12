FROM node:24-bookworm-slim AS frontend
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm AS wheel
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY --from=frontend /app/src/realtime_asr/server/static/ ./src/realtime_asr/server/static/
RUN uv build --wheel --out-dir /dist

FROM python:3.12-slim-bookworm AS runtime
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=wheel /dist/ /dist/
COPY packaging/wake-onnx-excludes.txt /app/wake-onnx-excludes.txt
RUN uv venv /opt/venv && uv pip install --python /opt/venv/bin/python --excludes /app/wake-onnx-excludes.txt "/dist/realtime_asr-0.1.0-py3-none-any.whl[server,whisper,nemotron,audio,vad,wake-openwakeword,wake-porcupine,models]" && rm -rf /dist
RUN useradd --uid 10001 --create-home asr
COPY packaging/healthcheck.py /app/healthcheck.py
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ASR_HEALTH_URL=https://127.0.0.1:8000/api/v1/ready ASR_HEALTH_CA=/run/asr-tls/server.crt
USER asr
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 CMD ["python", "/app/healthcheck.py"]
ENTRYPOINT ["realtime-asr", "serve"]
CMD ["--model", "/models/whisper-tiny", "--vad-model", "/models/vad/silero-v6.onnx", "--host", "0.0.0.0", "--ssl-certfile", "/run/asr-tls/server.crt", "--ssl-keyfile", "/run/asr-tls/server.key"]
