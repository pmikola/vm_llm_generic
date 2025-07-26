import torch, os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
app = FastAPI()
MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/model")
tokenizer = model = last_error = None
device = "cuda" if torch.cuda.is_available() else "cpu"
class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(300, ge=1, le=4096)
    temperature: float = Field(0.2, ge=0.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=0.8, le=2.0)
def _ok(path):
    import pathlib, re
    p = pathlib.Path(path)
    if list(p.glob("*.index.json")):
        return True
    return any(re.match(r"model-\d{5}-of-\d{5}\.safetensors", f.name)
               for f in p.glob("*.safetensors"))@app.on_event("startup")

def _load():
    global tokenizer, model, last_error
    if not _ok(MODEL_DIR):
        last_error=f"bad MODEL_DIR {MODEL_DIR}"
        return
    try:
        bnb=BitsAndBytesConfig(load_in_4bit=True,
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True)
        tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR,trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token=tokenizer.eos_token
        model=AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,quantization_config=bnb,device_map="auto",trust_remote_code=True)
        model.eval(); last_error=None
    except Exception as e:
        last_error=str(e); tokenizer=model=None
@app.get("/health")
def health():
    return {"status":"ok" if model else "degraded","last_error":last_error}
@app.post("/generate")
@torch.inference_mode()
def generate(req:GenRequest):
    if model is None: raise HTTPException(status_code=503,detail=last_error)
    inputs=tokenizer(req.prompt,return_tensors="pt").to(model.device)
    out=model.generate(**inputs,max_new_tokens=req.max_tokens,
                       do_sample=req.temperature>0,temperature=req.temperature,
                       top_p=req.top_p,repetition_penalty=req.repetition_penalty,
                       eos_token_id=tokenizer.eos_token_id,
                       pad_token_id=tokenizer.pad_token_id)
    new=out[0,inputs["input_ids"].shape[1]:]
    return{"prompt":req.prompt,
           "completion":tokenizer.decode(new,skip_special_tokens=True)}
