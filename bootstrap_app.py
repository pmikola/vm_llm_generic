import os
from fastapi import FastAPI, File, UploadFile
import shutil, subprocess
MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/model")

app = FastAPI()
@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    dest_path = f"{MODEL_DIR}/model.safetensors"
    with open(dest_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)
    return {"status": "received"}
