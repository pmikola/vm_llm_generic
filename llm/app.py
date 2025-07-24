import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)

app = FastAPI()

MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/model")

tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
last_error: str | None = None


class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(128, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=0.8, le=2.0)


def _model_files_exist(path: str) -> bool:
    needed = ["config.json", "tokenizer.json", "model.safetensors"]
    return all(os.path.isfile(os.path.join(path, f)) for f in needed)


@app.on_event("startup")
def _load_model_once():
    global tokenizer, model, last_error

    if not os.path.isdir(MODEL_DIR) or not _model_files_exist(MODEL_DIR):
        last_error = f"Model directory '{MODEL_DIR}' missing or incomplete."
        print("[startup] " + last_error)
        return

    try:
        print(f"[startup] Loading model from {MODEL_DIR} on {device} ...")

        tok = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            local_files_only=True,
        )

        cfg = AutoConfig.from_pretrained(
            MODEL_DIR, trust_remote_code=True, local_files_only=True
        )

        # pick a dtype
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            config=cfg,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=True,
        )

        if not torch.cuda.is_available():
            mdl = mdl.to(device)

        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

        tokenizer = tok
        model = mdl.eval()
        last_error = None

        print("[startup] Model loaded OK.")
    except Exception as e:
        last_error = f"{type(e).__name__}: {e}"
        tokenizer = None
        model = None
        print("[startup] FAILED to load model:", last_error)


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_dir": MODEL_DIR,
        "device": device,
        "last_error": last_error,
    }


@app.post("/generate")
@torch.inference_mode()
def generate(req: GenRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {last_error}")

    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)

    gen_out = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        do_sample=req.temperature > 0.0,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = gen_out[0, prompt_len:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "prompt": req.prompt,
        "completion": completion,
        "full_text": req.prompt + completion,
    }
