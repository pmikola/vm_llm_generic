#!/usr/bin/env python3
import requests
from pathlib import Path
from argparse import ArgumentParser
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from huggingface_hub import HfApi
from config import MODEL_HF

def sync_repo(repo_id: str, local_dir: Path, token: Optional[str], repo_type: str, workers: int):
    api = HfApi(token=token)
    files = api.repo_info(repo_id, repo_type=repo_type).siblings
    to_download = []
    total_bytes = 0
    for f in files:
        size = f.lfs["size"] if f.lfs else (f.size or 0)
        dest = local_dir / f.rfilename
        if dest.exists() and dest.stat().st_size == size:
            print(f"✓ {f.rfilename} (skipped)")
        else:
            to_download.append((f, size))
            total_bytes += size
    local_dir.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(total=total_bytes, unit='B', unit_scale=True, desc="Downloading")
    def download(f, size):
        url = f"https://huggingface.co/{repo_id}/resolve/main/{f.rfilename}"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        dest = local_dir / f.rfilename
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, headers=headers, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as fd:
                for chunk in r.iter_content(1024*1024):
                    if chunk:
                        fd.write(chunk)
                        pbar.update(len(chunk))
        return f.rfilename
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download, f, size): f for f, size in to_download}
        for future in as_completed(futures):
            f = futures[future]
            try:
                future.result()
                print(f"↓ {f.rfilename} (downloaded)")
            except:
                print(f"✗ {f.rfilename} (error)")
    pbar.close()

def download_model():
    p = ArgumentParser()
    p.add_argument("repo", nargs="?", default=MODEL_HF)
    p.add_argument("-t", "--token", default=None)
    p.add_argument("--type", choices=["model", "dataset", "space"], default="model")
    p.add_argument("-w", "--workers", type=int, default=5)
    args = p.parse_args()
    target = Path(__file__).parent.resolve() / "model"
    sync_repo(args.repo, target, args.token, args.type, args.workers)

if __name__ == "__main__":
    download_model()
