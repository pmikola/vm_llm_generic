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

def run_local(cmd: str):
    print(f"> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
        print(f"Error: Command failed with exit code {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)
    return result.stdout.strip() # This will now be safe to call

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
            return remote_hash.splitlines()[0]
        return None
    except subprocess.CalledProcessError as e:
        print(f"Could not get remote file hash (file might not exist): {e}")
        return None


def send_model():
    print(f"DEBUG: LOCAL_UPLOAD_SRV (script location): {LOCAL_UPLOAD_SRV}")
    print(f"DEBUG: PROJECT_ROOT (derived): {PROJECT_ROOT}")
    print(f"DEBUG: Local model path that upload_model will target (from config.MODEL_DIR): {PROJECT_ROOT / MODEL_DIR}")
    print(f"DEBUG: Remote model path that upload_model will target (from config.HOST_MODEL_DIR/MODEL_NAME): {pathlib.PurePosixPath(HOST_MODEL_DIR) / MODEL_NAME}")

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
        f'-p {PORT2UPLOAD}:{PORT2UPLOAD} ' 
        f'-v {HOST_MODEL_DIR}:/workspace/model ' 
        f'{IMAGE_NAME}"'
    )

    print("Initiating model upload via HTTP chunks...")
    try:
        upload_model()
        print("✅ Model uploaded successfully via HTTP chunks.")
    except Exception as e:
        print(f"Error during HTTP model upload: {e}")
        run_local(
            f'ssh -i "{KEY}" {USER}@{HOST} '
            f'-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
            f'"docker rm -f {CONTAINER_NAME}"'
        )
        raise

    run_local(
        f'ssh -i "{KEY}" {USER}@{HOST} '
        f'-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        f'"docker rm -f {CONTAINER_NAME}"'
    )
    print("✅ Model upload process complete and upload-server container removed")
# genFin