import os
import pathlib
import threading

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

app = FastAPI()

MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/model")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer: AutoTokenizer | None = None
model: AutoModelForCausalLM | None = None
last_error: str | None = None

stop_flag = threading.Event()

class StopOnFlag(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return stop_flag.is_set()


class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(300, ge=1, le=4096)
    temperature: float = Field(0.2, ge=0.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=0.8, le=2.0)


@app.on_event("startup")
def load_model():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if torch.cuda.is_available():
        qc = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            device_map="auto",
            quantization_config=qc,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.float32, trust_remote_code=True
        )
    model.eval()

@app.get("/health")
def health():
    code = 200 if model is not None else 503
    return JSONResponse(
        status_code=code,
        content={
            "status":       "ok" if model is not None else "degraded",
            "model_loaded": model is not None,
            "model_dir":    MODEL_DIR,
            "device":       str(DEVICE),
            "last_error":   last_error,
        },
    )

@app.post("/generate")
@torch.inference_mode()
def generate(req: GenRequest):
    global stop_flag

    if model is None or tokenizer is None:
        raise HTTPException(503, detail=f"Model not loaded: {last_error}")

    stop_flag.clear()

    inputs = tokenizer(req.prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    criteria = StoppingCriteriaList([StopOnFlag()])

    # start generation
    gen = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        do_sample=(req.temperature > 0.0),
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        stopping_criteria=criteria,
    )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = gen[0, prompt_len:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "prompt":     req.prompt,
        "completion": completion,
        "full_text":  req.prompt + completion,
    }
@app.post("/stop")
def stop_generation():
    stop_flag.set()
    return {"status": "stopping requested"}


