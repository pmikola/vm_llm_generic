FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      python3-venv \
      python3-pip \
      python3-dev \
      curl && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir \
      torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 && \
    /opt/venv/bin/pip install --no-cache-dir \
      bitsandbytes \
      fastapi uvicorn[standard] pydantic transformers accelerate sentencepiece

ENV PATH=/opt/venv/bin:$PATH

COPY app.py /workspace/app.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD curl -sf http://127.0.0.1:8000/health || exit 1


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
