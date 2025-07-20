import os, sys, hashlib, pathlib, contextlib, paramiko, docker
import time
from dotenv import load_dotenv
from getpass import getpass
import io

# ssh-keygen -t ed25519 -C "ovh_llm" -f "$env:USERPROFILE\.ssh\ssh_test"
load_dotenv()
# === conf ===========================================================
HOST        = os.getenv("OVH_HOST", "51.161.82.41")
USER        = os.getenv("OVH_USER", "ubuntu")
pkay        = r"C:\Users\Msi\.ssh\ssh_test_rsa"
KEY_PATH    = os.getenv("OVH_KEY", pkay)
LOCAL_CTX   = pathlib.Path("llm")
REMOTE_DIR  = f"/home/{USER}/llm"
IMAGE_TAG   = "llm:latest"
CONTAINER   = "llm"
MODEL_FILE  = "model.safetensors"
PORT        = 8000
TIMEOUT  = 5
# ============================================================================
with open(pkay, "r", encoding="utf-8") as f:
    for line in f:
        print(line, end="")

# === progress vis ===========================================================
# gptgen ####################################################
def make_progress_bar(total_bytes: int, width: int = 40):
    total = max(total_bytes, 1)
    last_pct_drawn = -1
    def _cb(transferred: int, _):
        nonlocal last_pct_drawn
        pct = int(transferred * 100 / total)
        if pct != last_pct_drawn:
            last_pct_drawn = pct
            filled = int(width * pct / 100)
            bar = "█" * filled + "─" * (width - filled)
            sys.stdout.write(
                f"\r   ⬆️  [{bar}] {pct:3d}% "
                f"({transferred/1_048_576:,.1f}/{total/1_048_576:.1f} MB)"
            )
            sys.stdout.flush()
        if transferred >= total:
            sys.stdout.write("\n")
    return _cb

def wait_apt_unlock(ssh, timeout=600):
    import time
    t0 = time.time()
    while True:
        lock = run(ssh, "ls /var/lib/dpkg/lock-frontend || true")
        if not lock:
            return
        if time.time() - t0 > timeout:
            raise TimeoutError("apt lock not released in time")
        print("⌛  unattended-upgrades w toku… czekam 5 s")
        time.sleep(5)
# gptgen ####################################################


def test_ssh_connection():
    print(KEY_PATH)
    print(f"🔑 Próba połączenia do {USER}@{HOST}…")
    private_key_file = KEY_PATH
    try:
        try:
            pkey = paramiko.RSAKey.from_private_key_file(private_key_file)
        except paramiko.PasswordRequiredException:
            pw = getpass(f"Passphrase for {KEY_PATH}: ")
            pkey = paramiko.RSAKey.from_private_key_file(private_key_file, password=pw)
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(HOST, username=USER, pkey=pkey, timeout=TIMEOUT)
    except paramiko.AuthenticationException as auth_err:
        print("auth error:", auth_err)
    except paramiko.SSHException as ssh_err:
        print("SSHException:", ssh_err)
    except Exception as e:
        print("Connection failed:", e)
    else:
        print("✅ Connection Established!")
        stdin, stdout, stderr = client.exec_command("uname -a")
        print("Test command result:", stdout.read().decode().strip())
        client.close()
# gptgen ####################################################


def new_ssh():
    try:
        pkey = paramiko.RSAKey.from_private_key_file(KEY_PATH)
    except paramiko.PasswordRequiredException:
        pkey = paramiko.RSAKey.from_private_key_file(
            KEY_PATH, password=getpass(f"Passphrase for {KEY_PATH}: "))
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, pkey=pkey, timeout=15)
    return ssh


def run(ssh, cmd, sudo=False):
    if sudo and not cmd.startswith("sudo"):
        cmd = "sudo " + cmd
    chan = ssh.get_transport().open_session()
    chan.exec_command(cmd)
    stdout = []
    while not chan.exit_status_ready():
        while chan.recv_ready():
            data = chan.recv(1024).decode()
            sys.stdout.write(data)
            stdout.append(data)
        while chan.recv_stderr_ready():
            sys.stderr.write(chan.recv_stderr(1024).decode())
        time.sleep(0.2)
    stdout.append(chan.recv(4096).decode())
    stderr = chan.recv_stderr(4096).decode()
    exit_code = chan.recv_exit_status()
    if exit_code != 0:
        raise RuntimeError(stderr)
    return "".join(stdout).strip()

# gptgen ####################################################
def ensure_docker(ssh):
    print("🐳  Installing Docker Engine + NVIDIA Container Toolkit…")
    wait_apt_unlock(ssh)
    run(ssh, "apt-get update", sudo=True)
    run(ssh, "apt-get install -y docker.io", sudo=True)

    run(ssh,
        "wget -qO /tmp/cuda-keyring.deb "
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/"
        "x86_64/cuda-keyring_1.1-1_all.deb",                       sudo=True)
    run(ssh, "dpkg -i /tmp/cuda-keyring.deb", sudo=True)           # :contentReference[oaicite:1]{index=1}
    run(ssh, "apt-get update", sudo=True)

    # 3) NVIDIA Container Toolkit + konfiguracja runtime
    run(ssh, "apt-get install -y nvidia-container-toolkit", sudo=True)
    run(ssh, "nvidia-ctk runtime configure --runtime=docker", sudo=True)
    run(ssh, "systemctl restart docker", sudo=True)
    run(ssh, f"usermod -aG docker {USER}", sudo=True)
# gptgen ####################################################

# ---------- upload build‑context ---------------------------------------------
def sha256(path: pathlib.Path, buf=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()

def human(n_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"

def upload_if_changed(sftp, local_file: pathlib.Path, remote_file: str):
    try:
        with sftp.open(remote_file, "rb") as r:
            remote_hash = hashlib.sha256(r.read()).hexdigest()
        if remote_hash == sha256(local_file) or sftp.stat(remote_file).st_size == local_file.stat().st_size:
            print(f" {local_file.name} up to date.")
            return
        print(f"{local_file.name} reloading...")
    except FileNotFoundError:
        print(f"{local_file.name} model not exist -> sending...")

    filesize = local_file.stat().st_size
    with local_file.open("rb") as lf:
        sftp.putfo(lf, remote_file, callback=make_progress_bar(filesize))
    print("Done.")

def upload_context(ssh):
    print(f"Upload build‑context → {REMOTE_DIR}")
    sftp = ssh.open_sftp()
    with contextlib.suppress(IOError):
        sftp.mkdir(REMOTE_DIR)

    upload_if_changed(
        sftp,
        LOCAL_CTX / MODEL_FILE,
        f"{REMOTE_DIR}/{MODEL_FILE}"
    )
    for item in LOCAL_CTX.iterdir():
        if item.name == MODEL_FILE:
            continue
        sftp.put(str(item), f"{REMOTE_DIR}/{item.name}")
    sftp.close()

# ---------- docker‑ SSH --------------------------------------------
def docker_cli():
    host_keys_path = os.path.expanduser("~/.ssh/ssh_test_rsa")
    pathlib.Path(host_keys_path).touch(exist_ok=True)

def build_and_run(client):
    print("Building remote image…")
    img, logs = client.images.build(
        path=REMOTE_DIR, tag=IMAGE_TAG, platform="linux/amd64", quiet=False)
    for chunk in logs:
        if "stream" in chunk:
            sys.stdout.write(chunk["stream"])

    print("Restarting container…")
    with contextlib.suppress(docker.errors.NotFound):
        client.containers.get(CONTAINER).remove(force=True)

    client.containers.run(
        IMAGE_TAG,
        name=CONTAINER,
        ports={f"{PORT}/tcp": PORT},
        device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        detach=True,
        restart_policy={"Name": "unless-stopped"}
    )
# ---------- main -------------------------------------------------------------
if __name__ == "__main__":
    test_ssh_connection()
    ssh = new_ssh()
    try:
        ensure_docker(ssh)
        upload_context(ssh)
    finally:
        ssh.close()
    # TODO : problem with build and run (cli)
    # TODO : problem with build context in llm folder on the cloud side (freezing - uploading anyway the model without progress bar?)
    build_and_run(docker_cli())
    print(f"\n LLM API ready:  http://{HOST}:{PORT}/v1/chat/completions\n")
