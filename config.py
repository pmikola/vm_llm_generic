# ───────── config ────────────────────────────────────────────────────────────
import os
import pathlib

MODEL_HF            = "Qwen/Qwen2.5-72B-Instruct"
HOST                = os.getenv("OVH_HOST", "54.38.138.9")
USER                = os.getenv("OVH_USER", "ubuntu")
KEY                 = os.getenv("OVH_KEY",  r"C:\Users\Msi\.ssh\ssh_test_rsa")
LOCAL_CTX           = pathlib.Path("llm")
REMOTE              = f"/home/{USER}/llm"
HOST_MODEL_DIR      = f"{REMOTE}/model"
CONTAINER_MODEL_DIR = "/workspace/model"
IMAGE_TAG           = "llm:latest"
CONTAINER           = "llm"
PORT                = 8000
PORT2UPLOAD         = PORT+1
RETRIES_PER_FILE    = 5
CHUNK_SIZE          = 4 * 1024 * 1024
SHOW_MODE           = "sum"   # or "max"
MODEL_NAME          = "model.safetensors"
MODEL_DIR           = "llm/model"
MODEL_URL           = URL = f"http://{HOST}:{PORT2UPLOAD}/upload?path={CONTAINER_MODEL_DIR}/{MODEL_NAME}"
CHUNK               = 4 * 1024 * 1024
MAX_THREADS         = 20
# ──────────────────────────────────────────────────────────────────────────────