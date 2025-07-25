import os
import subprocess
import pathlib
import hashlib
from config import *
from https_link import upload_model

# genEntry
REMOTE_UPLOAD_SRV = f"{REMOTE}/upload_model_srv"
IMAGE_NAME       = "upload-model-server"
CONTAINER_NAME   = "upload-server"
LOCAL_UPLOAD_SRV = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = LOCAL_UPLOAD_SRV.parent.parent.resolve()
LOCAL_MODEL_PATH = PROJECT_ROOT / MODEL_DIR
REMOTE_MODEL_PATH = pathlib.PurePosixPath(HOST_MODEL_DIR) / MODEL_NAME


def run_local(cmd: str):
    print(f"> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)
    return result.stdout.strip()

def get_local_file_hash(file_path):
    if not file_path.exists():
        return None
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_remote_file_hash(remote_path, user, host, key):
    try:
        cmd = (
            f'ssh -i "{key}" {user}@{host} '
            f'-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
            f'"sha256sum {remote_path} | awk \'{{print $1}}\' || true"'
        )
        remote_hash = run_local(cmd)
        if remote_hash:
            return remote_hash.splitlines()[0] # In case there's extra output
        return None
    except subprocess.CalledProcessError as e:
        print(f"Could not get remote file hash (file might not exist): {e}")
        return None


def send_model():
    print(f"DEBUG: LOCAL_UPLOAD_SRV (script location): {LOCAL_UPLOAD_SRV}")
    print(f"DEBUG: PROJECT_ROOT (derived): {PROJECT_ROOT}")
    print(f"DEBUG: LOCAL_MODEL_PATH (derived from config): {LOCAL_MODEL_PATH}")
    print(f"DEBUG: Does LOCAL_MODEL_PATH exist? {LOCAL_MODEL_PATH.exists()}")

    local_model_hash = get_local_file_hash(LOCAL_MODEL_PATH)
    if not local_model_hash:
        print(f"Error: Local model file not found at {LOCAL_MODEL_PATH}")
        return

    print(f"Local model hash: {local_model_hash}")

    print(f"DEBUG: REMOTE_MODEL_PATH (derived from config): {REMOTE_MODEL_PATH}")
    remote_model_hash = get_remote_file_hash(REMOTE_MODEL_PATH, USER, HOST, KEY)
    print(f"Remote model hash: {remote_model_hash}")

    perform_full_upload = True

    if remote_model_hash and remote_model_hash == local_model_hash:
        print("✅ Model already exists on the remote server and is identical.")
        perform_full_upload = False

    run_local(
        f'ssh -i "{KEY}" {USER}@{HOST} '
        f'-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        f'"mkdir -p {REMOTE_UPLOAD_SRV}"'
    )
    run_local(
        f'scp -i "{KEY}" -r '
        f'-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        f'"{LOCAL_UPLOAD_SRV}/" "{USER}@{HOST}:{REMOTE}/"'
    )

    run_local(
        f'ssh -i "{KEY}" {USER}@{HOST} '
        f'-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        f'"cd {REMOTE_UPLOAD_SRV} && '
        f'docker build -t {IMAGE_NAME} -f Dockerfile ."'
    )

    run_local(
        f'ssh -i "{KEY}" {USER}@{HOST} '
        f'-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        f'"docker rm -f {CONTAINER_NAME} || true && '
        f'docker run -d --name {CONTAINER_NAME} '
        f'-p {PORT2UPLOAD}:8001 '
        f'-v {HOST_MODEL_DIR}:/workspace/model ' 
        f'{IMAGE_NAME}"'
    )

    if perform_full_upload:
        print("Model not found on remote, or hashes differ. Proceeding with upload via HTTP chunks.")
        upload_model()
    else:
        print("Skipping HTTP chunk upload as model is identical.")
    run_local(
        f'ssh -i "{KEY}" {USER}@{HOST} '
        f'-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        f'"docker rm -f {CONTAINER_NAME}"'
    )
    print("✅ Model upload/check complete and upload‐server container removed")
# genFin