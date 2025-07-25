import os
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import PlainTextResponse

app = FastAPI()
BASE_DIR = "/workspace/model"

@app.on_event("startup")
def ensure_dir():
    os.makedirs(BASE_DIR, exist_ok=True)

@app.put("/upload", response_class=PlainTextResponse)
async def upload_chunk(
    request: Request,
    path: str = Query(...),
):
    full_path = os.path.join(BASE_DIR, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    cr = request.headers.get("Content-Range")
    body = await request.body()

    try:
        mode = "r+b" if cr else "wb"
        with open(full_path, mode) as f:
            if cr:
                # parse "bytes X-Y/TOTAL"
                start = int(cr.split(" ")[1].split("/")[0].split("-")[0])
                f.seek(start)
            f.write(body)
    except Exception as e:
        raise HTTPException(500, f"Disk write error: {e}")
    return "OK"
