FROM ubuntu:22.04
WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126 && \
    pip3 install --no-cache-dir --upgrade pip vllm

COPY model.safetensors /workspace/model.safetensors

CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "/workspace/model.safetensors", "--port", "8000", "--host", "0.0.0.0"]
