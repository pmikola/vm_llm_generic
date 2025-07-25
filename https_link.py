import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from config import *

# genEntry
session = requests.Session()
retry = Retry(
    total=RETRIES_PER_FILE,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["PUT"],
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

def upload_chunk(start: int, end: int, total: int, pbar: tqdm):
    headers = {"Content-Range": f"bytes {start}-{end}/{total}"}
    with open(MODEL_DIR, "rb") as f:
        f.seek(start)
        data = f.read(end - start + 1)

    resp = session.put(MODEL_URL, data=data, headers=headers, verify=False)
    resp.raise_for_status()
    pbar.update(len(data))

def upload_model():
    if not os.path.isfile(MODEL_DIR):
        raise FileNotFoundError(f"Model file not found: {MODEL_DIR}")

    total_size = os.path.getsize(MODEL_DIR)
    ranges = [
        (s, min(s + CHUNK - 1, total_size - 1))
        for s in range(0, total_size, CHUNK)
    ]

    with ThreadPoolExecutor(MAX_THREADS) as executor, \
         tqdm(total=total_size, unit="B", unit_scale=True, desc="Uploading") as pbar:

        futures = [
            executor.submit(upload_chunk, start, end, total_size, pbar)
            for start, end in ranges
        ]
        for fut in as_completed(futures):
            fut.result()

    print("✅ upload complete")
# genFin