#!/usr/bin/env python3
"""deploy.py — push local llm/ context to an OVH VM, build & run the GPU container."""

import os, sys, time, contextlib, pathlib, socket, posixpath
import paramiko
from dotenv import load_dotenv
from math import log2

load_dotenv()

# ───────── config ────────────────────────────────────────────────────────────
HOST                = os.getenv("OVH_HOST", "51.79.26.54")
USER                = os.getenv("OVH_USER", "ubuntu")
KEY                 = os.getenv("OVH_KEY",  r"C:\Users\Msi\.ssh\ssh_test_rsa")
LOCAL_CTX           = pathlib.Path("llm")
REMOTE              = f"/home/{USER}/llm"
HOST_MODEL_DIR      = f"{REMOTE}/model"
CONTAINER_MODEL_DIR = "/workspace/model"
IMAGE_TAG           = "llm:latest"
CONTAINER           = "llm"
PORT                = 8000
RETRIES_PER_FILE    = 5
CHUNK_SIZE          = 4 * 1024 * 1024
SHOW_MODE           = "sum"   # or "max"
# ──────────────────────────────────────────────────────────────────────────────

def open_ssh() -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST,
                username=USER,
                pkey=paramiko.RSAKey.from_private_key_file(KEY),
                timeout=15)
    ssh.get_transport().set_keepalive(30)
    return ssh

def run(ssh, cmd, *, sudo=False, stream=False, timeout=None):
    if sudo and not cmd.lstrip().startswith("sudo"):
        cmd = "sudo -n " + cmd
    chan = ssh.get_transport().open_session()
    chan.settimeout(timeout)
    chan.get_pty()
    chan.exec_command(cmd)
    out_chunks, err_chunks = [], []
    def drain():
        while chan.recv_ready():
            d = chan.recv(4096).decode(errors="replace")
            out_chunks.append(d)
            if stream: sys.stdout.write(d); sys.stdout.flush()
        while chan.recv_stderr_ready():
            d = chan.recv_stderr(4096).decode(errors="replace")
            err_chunks.append(d)
            if stream: sys.stderr.write(d); sys.stderr.flush()
    while not chan.exit_status_ready():
        drain(); time.sleep(0.05)
    drain()
    code = chan.recv_exit_status()
    out = "".join(out_chunks)
    err = "".join(err_chunks)
    if code != 0:
        raise RuntimeError(f"Command failed (exit {code}): {cmd}\n\nSTDOUT:\n{out}\n\nSTDERR:\n{err}")
    return out

def fmt_bytes(b):
    units = ["B","KB","MB","GB","TB"]
    if b == 0: return "0 B"
    i = min(int(log2(b)/10), len(units)-1)
    return f"{b/(1024**i):,.1f} {units[i]}"

def safe_put_file(ssh, local: pathlib.Path, remote: str, show_progress: bool, total_for_bar: int, bump=None):
    size = local.stat().st_size
    sent = 0
    sftp = ssh.open_sftp()
    try:
        sent = sftp.stat(remote).st_size
    except FileNotFoundError:
        sent = 0
    attempt = 0
    while sent < size:
        attempt += 1
        if attempt > RETRIES_PER_FILE:
            raise RuntimeError(f"Failed after {RETRIES_PER_FILE} retries: {local}")
        try:
            parent = posixpath.dirname(remote)
            try:
                sftp.stat(parent)
            except FileNotFoundError:
                run(ssh, f"mkdir -p {parent} && chown -R {USER}:{USER} {parent}", sudo=True, stream=True)
                sftp = ssh.open_sftp()
            with local.open("rb") as f, sftp.file(remote, "ab") as rf:
                f.seek(sent)
                rf.set_pipelined(True)
                while sent < size:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk: break
                    rf.write(chunk); sent += len(chunk)
                    if bump: bump(len(chunk))
                    # genEntry
                    if show_progress:
                        pct = int(sent * 100 / size)
                        bar = "█"*(pct//2) + "─"*(50-pct//2)
                        sys.stdout.write(f"\r→ {local.name} [{bar}] {pct:3d}% ({fmt_bytes(sent)}/{fmt_bytes(size)})")
                        sys.stdout.flush()
                    # genFin
            break
        except Exception as e:
            sftp.close()
            ssh.close()
            ssh = open_ssh()
            sftp = ssh.open_sftp()
    if show_progress:
        sys.stdout.write("\n"); sys.stdout.flush()

def ensure_remote_dir_writable(ssh, path):
    run(ssh, f"mkdir -p {path}", sudo=True, stream=True)
    run(ssh, f"chown -R {USER}:{USER} {path}", sudo=True, stream=True)

# genEntry
def upload_ctx(ssh):
    print(">> Uploading context…")
    ensure_remote_dir_writable(ssh, REMOTE)
    ensure_remote_dir_writable(ssh, f"{REMOTE}/model")
    sftp = ssh.open_sftp()
    with contextlib.suppress(OSError):
        sftp.mkdir(REMOTE)
    files = [(p, p.stat().st_size) for p in LOCAL_CTX.rglob("*") if p.is_file()]
    total = sum(sz for _, sz in files)
    largest = max((sz for _, sz in files), default=0)
    grand = total if SHOW_MODE=="sum" else largest
    done = 0
    def draw_bar():
        pct = int(done*100/grand)
        bar = "█"*(pct//2)+"─"*(50-pct//2)
        sys.stdout.write(f"\r⬆️  [{bar}] {pct:3d}% ({fmt_bytes(done)}/{fmt_bytes(grand)})")
        sys.stdout.flush()
        if done>=grand: sys.stdout.write("\n")
    def bump(d):
        nonlocal done
        done = min(done+d, grand); draw_bar()
    for p, sz in files:
        rel = p.relative_to(LOCAL_CTX).as_posix()
        rpath = posixpath.join(REMOTE, rel)
        try:
            if sftp.stat(rpath).st_size == sz:
                if SHOW_MODE=="sum": bump(sz)
                continue
        except FileNotFoundError:
            pass
        print(f"\n→ {rel} ({fmt_bytes(sz)})")
        if sz > 64*1024*1024:
            safe_put_file(ssh, p, rpath, True, grand, bump if SHOW_MODE=="sum" else None)
        else:
            last = 0
            def cb(tx, _): nonlocal last; d=tx-last; last=tx; bump(d) if SHOW_MODE=="sum" or sz==largest else None
            sftp.put(str(p), rpath, callback=cb, confirm=False)
    sftp.close()
# genFin


def ensure_docker_with_nvidia(ssh):
    bsh = lambda c: run(ssh, f"DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a bash -c \"{c}\"", sudo=True,
                        stream=True)
    #bsh = lambda c: run(ssh, f"bash -c \"{c}\"", sudo=True,stream=True)
    # Note: Clear
    bsh("rm -f /etc/apt/sources.list.d/*docker*.list* /etc/apt/keyrings/docker.asc /etc/apt/keyrings/docker.gpg")

    # Note: Docker
    bsh("apt-get upgrade -y")
    bsh("apt-get update -y")
    bsh("apt-get install -y --no-install-recommends ca-certificates curl gnupg lsb-release")
    bsh("apt-get install curl")
    bsh("install -m 0755 -d /etc/apt/keyrings")
    bsh("curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc")
    bsh("chmod 0644 /etc/apt/keyrings/docker.asc")
        #bsh("chmod a+r /etc/apt/keyrings/docker.asc")
    bsh("echo \"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] "
        "https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME:-$VERSION_CODENAME}) stable\" "
        ">> /etc/apt/sources.list.d/docker.list")
    bsh("apt-get update -y")
    bsh("apt install -y apt-transport-https ca-certificates curl software-properties-common")
    bsh("curl -fsSL https://download.docker.com/linux/ubuntu/gpg")
    bsh(r'''
        wget -qO - https://package.perforce.com/perforce.pubkey \
          | gpg --batch --yes --dearmor -o /usr/share/keyrings/perforce-archive-keyring.gpg
        ''')
    bsh(r'''
        echo "deb [signed-by=/usr/share/keyrings/perforce-archive-keyring.gpg] \
        http://package.perforce.com/apt/ubuntu focal release" \
          | tee /etc/apt/sources.list.d/perforce.list > /dev/null
        ''')
    bsh("apt-get update -y")
    bsh("apt-get install -y docker.io")
    bsh("systemctl enable docker")
    bsh("systemctl start docker")
    bsh("sleep 2")
    bsh("docker --version")
    bsh("docker run --rm hello-world")

#     # Note: NVIDIA Container Toolkit
    bsh("curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg")
    bsh("curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | "
        "sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' "
        "| tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null")
    bsh("apt-get update -y")
    bsh("apt-get install -y nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1")
    bsh("nvidia-ctk runtime configure --runtime=docker")

    # Note restart & permissions
    bsh("systemctl restart docker")
    bsh(f"usermod -aG docker {USER}")
    print("\nDocker + NVIDIA runtime ready")

def reboot_and_wait(ssh):
    try: run(ssh, "reboot", sudo=True, stream=True)
    except: pass
    ssh.close()
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            sock = socket.create_connection((HOST, 22), timeout=3); sock.close()
            n = open_ssh(); time.sleep(3); return n
        except: time.sleep(5)
    raise TimeoutError("VM did not reboot in time")

def ensure_nvidia_driver(ssh):
    try:
        run(ssh, "nvidia-smi -L", sudo=True, timeout=10)
        print("NVIDIA driver already present.")
        return ssh
    except Exception:
        print("No working NVIDIA driver found. Installing… (will reboot once)")
        bsh = lambda c: run(ssh, f"DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a bash -lc \"{c}\"", sudo=True,stream=True)
        #bsh = lambda c: run(ssh, f"bash -lc \"{c}\"", sudo=True, stream=True)
        bsh("apt-get update -y && apt-get install -y ubuntu-drivers-common linux-headers-$(uname -r)")
        bsh("ubuntu-drivers autoinstall || true")
        ssh = reboot_and_wait(ssh)
        run(ssh, "nvidia-smi", sudo=True, stream=True)
        return ssh

def configure_noninteractive(ssh):
    run(ssh, "echo 'NEEDRESTART_MODE=a' | sudo tee -a /etc/needrestart/needrestart.conf", sudo=True, stream=True)
    run(ssh, "echo 'DPkg::options { \"--force-confdef\"; };' | sudo tee /etc/apt/apt.conf.d/50forceconfdef", sudo=True, stream=True)

# genEntry
def gpu_sanity_check(ssh):
    run(ssh, "docker run --rm --gpus all nvidia/cuda:12.6.2-runtime-ubuntu22.04 nvidia-smi", sudo=True, stream=True)
# genFin

def open_port(ssh, PORT):
    run(ssh, f"ufw allow {PORT}/tcp || true", sudo=True, stream=True)
    run(ssh, "ufw reload || true", sudo=True, stream=True)

def docker_container_exists(ssh):
    out = run(ssh, f"docker ps -a --format '{{{{.Names}}}}' | grep -x {CONTAINER} || true", sudo=True)
    return out.strip() == CONTAINER

def wait_for_health(ssh, container, timeout_s=900, interval_s=2):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status = run(
                ssh,
                f"sudo -n docker inspect -f '{{{{.State.Health.Status}}}}' {container}",
                stream=False
            ).strip()
            print("health:", status)
            if status == "healthy":
                return True
            if status == "unhealthy":
                break
        except Exception as e:
            print("inspect error:", e)
        time.sleep(interval_s)
    return False


def build_and_run(ssh) -> None:
    bsh = lambda c: run(ssh, f"sudo -n bash -lc \"{c}\"", stream=True)

    print("Stopping & removing old container (if it exists)…")
    bsh(f"docker rm -f {CONTAINER} || true")

    print("Building new image…")
    bsh(f"cd {REMOTE} && docker build -t {IMAGE_TAG} .")

    print("Starting fresh container…")
    bsh(
        f"docker run -d --name {CONTAINER} "
        f"--gpus all -p {PORT}:{PORT} "
        f"-v {HOST_MODEL_DIR}:{CONTAINER_MODEL_DIR}:ro "
        f"-e MODEL_DIR={CONTAINER_MODEL_DIR} "
        f"--restart unless-stopped {IMAGE_TAG}"
    )

    open_port(ssh, PORT)

    print("Waiting for container to pass healthcheck…")
    if not wait_for_health(ssh, CONTAINER, timeout_s=900, interval_s=3):
        logs = run(ssh, f"sudo -n docker logs --tail 200 {CONTAINER}", stream=False)
        print("\n--- last 200 log lines ---\n", logs)
        raise RuntimeError("Container did not pass health check.")
    print("→ container is healthy!")

def debug_health(ssh):
    print("\n### docker ps")
    print(run(ssh, "sudo -n docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"))
    print("\n### docker inspect health")
    print(run(ssh, f"sudo -n docker inspect --format='{{{{json .State.Health}}}}' {CONTAINER} || true"))
    print("\n### /health from inside the container")
    try:
        print(run(ssh, f"sudo -n docker exec {CONTAINER} curl -s -v http://127.0.0.1:{PORT}/health", stream=False))
    except Exception as e:
        print("curl failed:", e)
    print("\n### last 200 log lines")
    print(run(ssh, f"sudo -n docker logs --tail 200 {CONTAINER} || true"))
    print("\n### check model dir exists in the container")
    print(run(ssh, f"sudo -n docker exec {CONTAINER} bash -lc 'ls -lah /workspace/model || true'"))


if __name__ == "__main__":
    ssh = open_ssh()
    print(run(ssh, "uname -a"))
    ssh = ensure_nvidia_driver(ssh)
    ensure_docker_with_nvidia(ssh)
    upload_ctx(ssh)
    build_and_run(ssh)
    gpu_sanity_check(ssh)
    debug_health(ssh)
    ssh.close()
    print(f"\nServer is ✅ Ready! → http://{HOST}:{PORT}\n") # gptGen