#!/usr/bin/env python3
"""deploy.py — push local llm/ context to an OVH VM, build & run the GPU container."""

import os, sys, time, tarfile, io, contextlib, pathlib
import paramiko, docker
from dotenv import load_dotenv
import os, posixpath
from math import log2

load_dotenv()

# ───────── config ────────────────────────────────────────────────────────────
HOST      = os.getenv("OVH_HOST", "51.79.31.174")
USER      = os.getenv("OVH_USER", "ubuntu")
KEY       = os.getenv("OVH_KEY",  r"C:\Users\Msi\.ssh\ssh_test_rsa")
LOCAL_CTX = pathlib.Path("llm")
REMOTE    = f"/home/{USER}/llm"
HOST_MODEL_DIR     = f"{REMOTE}/model"
CONTAINER_MODEL_DIR = "/workspace/model"
IMAGE_TAG = "llm:latest"
CONTAINER = "llm"
PORT      = 8000
RETRIES_PER_FILE = 5
CHUNK_SIZE = 4 * 1024 * 1024
SHOW_MODE = "sum"   #Note: or "max"
# ───────── config ────────────────────────────────────────────────────────────

def human_mb(b):
    return f"{b/1_048_576:,.1f} MB"

def open_ssh() -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER,
                pkey=paramiko.RSAKey.from_private_key_file(KEY), timeout=15)
    ssh.get_transport().set_keepalive(30)
    return ssh

# gptgen ##############
def safe_put_file(ssh, local_path: pathlib.Path, remote_path: str,
                  show_progress: bool, total_for_bar: int | None,
                  bump=None):
    size = local_path.stat().st_size
    attempt = 0
    sent = 0

    def open_sftp():
        return ssh.open_sftp()

    sftp = open_sftp()

    try:
        st = sftp.stat(remote_path)
        sent = st.st_size
    except FileNotFoundError:
        sent = 0

    while sent < size:
        attempt += 1
        if attempt > RETRIES_PER_FILE:
            raise RuntimeError(f"Giving up after {RETRIES_PER_FILE} retries uploading {local_path}")

        try:
            parent = posixpath.dirname(remote_path)
            try:
                sftp.stat(parent)
            except FileNotFoundError:
                run(ssh, f"sudo -n mkdir -p {parent} && sudo -n chown -R {USER}:{USER} {parent}", stream=True)
                sftp = open_sftp()

            with local_path.open("rb") as lf:
                lf.seek(sent)
                with sftp.file(remote_path, "ab") as rf:
                    rf.set_pipelined(True)
                    while sent < size:
                        chunk = lf.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        rf.write(chunk)
                        sent += len(chunk)
                        if bump:
                            bump(len(chunk))
                        if show_progress:
                            pct = int(sent * 100 / size)
                            bar = "█" * (pct // 2) + "─" * (50 - pct // 2)
                            sys.stdout.write(
                                f"\r→ {local_path.name} [{bar}] {pct:3d}% ({fmt_bytes(sent)}/{fmt_bytes(size)})"
                            )
                            sys.stdout.flush()
            break
        except (EOFError, OSError, socket.error, paramiko.SSHException) as e:
            try:
                sftp.close()
            except Exception:
                pass
            try:
                ssh.close()
            except Exception:
                pass
            ssh = open_ssh()
            sftp = open_sftp()
            try:
                sent = sftp.stat(remote_path).st_size
            except FileNotFoundError:
                sent = 0
            print(f"\n[retry {attempt}/{RETRIES_PER_FILE}] resuming {local_path.name} @ {fmt_bytes(sent)} due to: {e}")

    if show_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()

def run(ssh, cmd, *, sudo=False, stream=False, timeout=None):
    if sudo and not cmd.lstrip().startswith("sudo"):
        cmd = "sudo -n " + cmd
    chan = ssh.get_transport().open_session()
    chan.settimeout(timeout)
    chan.get_pty()
    chan.exec_command(cmd)
    stdout_chunks = []
    stderr_chunks = []
    def drain():
        while chan.recv_ready():
            data = chan.recv(4096).decode(errors="replace")
            stdout_chunks.append(data)
            if stream:
                sys.stdout.write(data)
                sys.stdout.flush()
        while chan.recv_stderr_ready():
            data = chan.recv_stderr(4096).decode(errors="replace")
            stderr_chunks.append(data)
            if stream:
                sys.stderr.write(data)
                sys.stderr.flush()
    while not chan.exit_status_ready():
        drain()
        time.sleep(0.05)
    drain()
    exit_code = chan.recv_exit_status()
    out = "".join(stdout_chunks)
    err = "".join(stderr_chunks)

    if exit_code != 0:
        raise RuntimeError(
            f"Command failed (exit {exit_code}): {cmd}\n\n"
            f"--- STDOUT ---\n{out}\n"
            f"--- STDERR ---\n{err}"
        )
    return out


def ensure_docker_with_nvidia(ssh):
    bsh = lambda c: run(ssh, f"bash -c \"{c}\"", sudo=True,stream=True)
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


def fmt_bytes(b):
    units = ["B", "KB", "MB", "GB", "TB"]
    if b == 0:
        return "0 B"
    i = min(int(log2(b) / 10), len(units) - 1)
    return f"{b / (1024 ** i):,.1f} {units[i]}"

def ensure_remote_dir_writable(ssh, path: str, user: str):
    run(ssh, f"sudo -n mkdir -p {path}", stream=True)
    run(ssh, f"sudo -n chown -R {user}:{user} {path}", stream=True)

def upload_ctx(ssh):
    # TODO: Need to send models via https not sftp
    print(">> Uploading context (SFTP)")
    ensure_remote_dir_writable(ssh, REMOTE, USER)
    ensure_remote_dir_writable(ssh, f"{REMOTE}/model", USER)
    sftp = ssh.open_sftp()

    with contextlib.suppress(OSError):
        sftp.mkdir(REMOTE)

    files = [p for p in LOCAL_CTX.rglob("*") if p.is_file()]
    manifest = [(p, p.stat().st_size) for p in files]
    total_bytes   = sum(sz for _, sz in manifest)
    largest_bytes = max((sz for _, sz in manifest), default=0)

    print("Files to sync:")
    for p, sz in sorted(manifest, key=lambda x: x[1], reverse=True):
        rel = str(p.relative_to(LOCAL_CTX))
        print(f"  {rel:<40} {fmt_bytes(sz)}")
    print()

    grand_total = total_bytes if SHOW_MODE == "sum" else largest_bytes
    if grand_total == 0:
        print("Nothing to upload.")
        sftp.close()
        return

    done = 0
    last_bar = -1
    # gptgen ########
    def draw_bar():
        nonlocal last_bar
        pct = int(done * 100 / grand_total)
        if pct == last_bar:
            return
        last_bar = pct
        bar = "█" * (pct // 2) + "─" * (50 - pct // 2)
        sys.stdout.write(
            f"\r⬆️  [{bar}] {pct:3d}% ({fmt_bytes(done)}/{fmt_bytes(grand_total)})"
        )
        sys.stdout.flush()
        if done >= grand_total:
            sys.stdout.write("\n")

    def bump(delta):
        nonlocal done
        done += delta
        if done > grand_total:
            done = grand_total
        draw_bar()

    def ensure_dirs(path):
        parts = path.split("/")
        cur = ""
        for part in parts:
            if not part:
                continue
            cur = cur + "/" + part
            with contextlib.suppress(OSError):
                sftp.mkdir(cur)

    for p, size in manifest:
        rel_str = p.relative_to(LOCAL_CTX).as_posix()
        remote = posixpath.join(REMOTE, rel_str)

        try:
            if sftp.stat(remote).st_size == size:
                print(f"  {rel_str} (skip, {fmt_bytes(size)})")
                if SHOW_MODE == "sum":
                    bump(size)
                continue
        except FileNotFoundError:
            pass

        print(f"\n→ {rel_str} ({fmt_bytes(size)})")

        if size > 64 * 1024 * 1024:
            safe_put_file(ssh, p, remote, show_progress=True,
                          total_for_bar=grand_total, bump=bump if SHOW_MODE == "sum" else None)
        else:
            tries = 0
            while True:
                tries += 1
                try:
                    last = 0

                    def cb(tx, total):
                        nonlocal last
                        delta = tx - last
                        last = tx
                        if SHOW_MODE == "sum":
                            bump(delta)
                        elif size == largest_bytes:
                            bump(delta)

                    sftp.put(str(p), remote, callback=cb, confirm=False)
                    break
                except (EOFError, OSError, socket.error, paramiko.SSHException) as e:
                    if tries >= RETRIES_PER_FILE:
                        raise
                    print(f"\n[retry {tries}/{RETRIES_PER_FILE}] {rel_str} due to: {e}")
                    try:
                        sftp.close()
                    except Exception:
                        pass
                    ssh = open_ssh()
                    sftp = ssh.open_sftp()

    sftp.close()
    if SHOW_MODE == "sum" and done < grand_total:
        bump(grand_total - done)
    print("\n✓ Upload complete")

def tar_context(folder):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for p in folder.rglob("*"):
            tar.add(p, arcname=p.relative_to(folder))
    buf.seek(0)
    return buf

# gptgen ######
def render_build_line(line, layers):
    if 'stream' in line:
        sys.stdout.write(line['stream'])
        sys.stdout.flush()
        return

    if 'status' in line and 'id' in line:
        _id = line['id']
        status = line['status']
        detail = line.get('progressDetail') or {}
        cur, tot = detail.get('current'), detail.get('total')

        if tot and tot > 0:
            pct = int(cur * 100 / tot)
            layers[_id] = f"{status} {pct:3d}% ({cur/1e6:.1f}/{tot/1e6:.1f} MB)"
        else:
            layers[_id] = status

        sys.stdout.write("\r")
        for k, v in list(layers.items())[-6:]:
            sys.stdout.write(f"{k[:12]}: {v} | ")
        sys.stdout.flush()
        return

    if 'aux' in line and isinstance(line['aux'], dict) and 'ID' in line['aux']:
        sys.stdout.write(f"\nBuilt image: {line['aux']['ID']}\n")
        sys.stdout.flush()
        return

    if 'error' in line:
        raise RuntimeError(line.get('errorDetail', {}).get('message', line['error']))

import socket
def reboot_and_wait(ssh, host, port=22, timeout=600, poll_every=3):
    try:
        run(ssh, "sudo -n reboot || sudo reboot", stream=True)
    except Exception:
        pass
    try:
        ssh.close()
    except Exception:
        pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=3)
            sock.close()
            new = open_ssh()
            time.sleep(3)
            return new
        except Exception:
            time.sleep(poll_every)
    raise TimeoutError("VM did not come back after reboot")

def ensure_nvidia_driver(ssh):
    try:
        run(ssh, "nvidia-smi -L", sudo=True, timeout=10)
        print("NVIDIA driver already present.")
        return ssh
    except Exception:
        print("No working NVIDIA driver found. Installing… (will reboot once)")
        bsh = lambda c: run(ssh, f"bash -lc \"{c}\"", sudo=True, stream=True)
        bsh("apt-get update -y && apt-get install -y ubuntu-drivers-common linux-headers-$(uname -r)")
        bsh("ubuntu-drivers autoinstall || true")
        ssh = reboot_and_wait(ssh, HOST)
        run(ssh, "nvidia-smi", sudo=True, stream=True)
        return ssh

# gptgen ######
def gpu_sanity_check(ssh):
    bsh = lambda c: run(ssh, f"sudo -n bash -lc \"{c}\"", stream=True)
    print("Sanity check: docker can see the GPU…")
    bsh("docker run --rm --gpus all nvidia/cuda:12.6.2-runtime-ubuntu22.04 nvidia-smi")

def open_port(ssh, port):
    try:
        run(ssh, f"which ufw >/dev/null 2>&1 && sudo -n ufw allow {port}/tcp || true", stream=True)
        run(ssh, "which ufw >/dev/null 2>&1 && sudo -n ufw reload || true", stream=True)
    except Exception:
        pass

# gptgen ######
def check_container_health(ssh, port, container, path="/health", tries=120, sleep_s=2):
    for i in range(tries):
        code = run(
            ssh,
            f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}{path}",
            sudo=True
        ).strip()
        if code == "200":
            print("Health check OK")
            return True
        time.sleep(sleep_s)

    logs = run(ssh, f"sudo -n docker logs --tail 200 {container} || true", stream=False)
    print("\n--- last 200 container log lines ---\n", logs)
    return False

# gptgen ######
def wait_for_docker_health(ssh, container, timeout_s=600, interval_s=2):
    import time
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status = run(
                ssh,
                f"sudo -n docker inspect -f '{{{{.State.Health.Status}}}}' {container}",
                stream=False
            ).strip()
            if status == "healthy":
                print("Docker healthcheck: healthy")
                return True
            else:
                print("Docker healthcheck:", status)
        except Exception as e:
            print("Error reading docker health status:", e)
        time.sleep(interval_s)
    return False

def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", r"'\''") + "'"

# gptgen ######
def discover_model_dir(ssh, container: str) -> str | None:
    required = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    # Note: From HuggingFace Example TinyLama1.1B

    def has_all_files(path: str) -> bool:
        for f in required:
            try:
                run(ssh, f"sudo -n docker exec {container} test -f {path}/{f}", stream=False)
            except Exception:
                return False
        return True

    candidates = [
        "/model",
        "/workspace/model",
        "/workspace",
        "/data/model",
        "/data",
    ]
    for c in candidates:
        if has_all_files(c):
            return c

    try:
        find_cmd = (
            "find / -maxdepth 5 -type d \\( -name model -o -name '*llama*' -o -name '*tiny*' -o -name '*hf*' \\) "
            "2>/dev/null | head -n 100"
        )
        out = run(
            ssh,
            f"sudo -n docker exec {container} bash -lc { _shell_quote(find_cmd) }",
            stream=False
        ).strip()
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            if has_all_files(line):
                return line
    except Exception as e:
        print("discover_model_dir: find failed:", e)

    return None

def list_dir_tree(ssh, container: str, root: str, depth: int = 2, max_lines: int = 300) -> None:
    root_q = _shell_quote(root)
    cmd = (
        f"sudo -n docker exec {container} bash -lc "
        f"\"set -e; "
        f"if [ -d {root_q} ]; then "
        f"  find {root_q} -maxdepth {depth} -printf '%y %p\\n' 2>/dev/null | head -n {max_lines}; "
        f"else "
        f"  echo 'Path {root} does not exist'; "
        f"fi\""
    )
    try:
        out = run(ssh, cmd, stream=False)
        print(out.strip())
    except Exception as e:
        print(f"(list_dir_tree) failed for {root}: {e}")

def build_and_run(ssh) -> None:
    print("\n### docker inspect .Mounts")
    print(run(ssh, f"sudo -n docker inspect -f '{{{{json .Mounts}}}}' {CONTAINER}", stream=False))

    print("\n### host dir listing")
    print(run(ssh, f"sudo -n bash -lc 'ls -lah {HOST_MODEL_DIR}'", stream=False))

    print("\n### container dir listing (mounted path)")
    print(run(ssh, f"sudo -n docker exec {CONTAINER} bash -lc 'ls -lah {CONTAINER_MODEL_DIR}'", stream=False))
    print(">> Building & running (on the VM, no second upload)")
    bsh = lambda c: run(ssh, f"sudo -n bash -lc \"{c}\"", stream=True)

    bsh(f"cd {REMOTE} && DOCKER_BUILDKIT=1 docker build --progress=plain -t {IMAGE_TAG} .")
    bsh(f"docker rm -f {CONTAINER} >/dev/null 2>&1 || true")
    bsh(
        f"docker run -d --name {CONTAINER} "
        f"--gpus all -p {PORT}:{PORT} "
        f"-v {HOST_MODEL_DIR}:{CONTAINER_MODEL_DIR}:ro "
        f"-e MODEL_DIR={CONTAINER_MODEL_DIR} "
        f"--restart unless-stopped {IMAGE_TAG}"
    )
    discovered = discover_model_dir(ssh, CONTAINER)
    if not discovered:
        print("\n>>> Diagnostics (couldn’t find model automatically)")
        list_dir_tree(ssh, CONTAINER, "/model", depth=3)
        list_dir_tree(ssh, CONTAINER, "/workspace", depth=3)
        logs = run(ssh, f"sudo -n docker logs --tail 200 {CONTAINER} || true")
        print("\n--- last 200 container log lines ---\n", logs)
        raise RuntimeError("No directory containing a full HF model was found inside the container.")

    print(f"\n✅  Model directory discovered: {discovered}")

    if discovered != CONTAINER_MODEL_DIR:
        print(
            "⚠️  NOTE: The container was started with MODEL_DIR=/model,\n"
            f"    but the files are under {discovered}.\n"
            "    You may want to adjust the -v mount or set MODEL_DIR accordingly."
        )

    print("\n### Contents of the discovered model directory")
    list_dir_tree(ssh, CONTAINER, discovered, depth=2)

    open_port(ssh, PORT)

    print("\n>> Waiting for health-check …")
    if not wait_for_docker_health(ssh, CONTAINER, timeout_s=600, interval_s=2):
        if not check_container_health(ssh, PORT, CONTAINER, path="/health"):
            logs = run(ssh, f"sudo -n docker logs --tail 200 {CONTAINER} || true")
            print("\n--- last 200 container log lines ---\n", logs)
            raise RuntimeError(" Container did not pass health check.")

    print("🎉  Health check OK – the service should be live!")

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
    ensure_docker_with_nvidia(ssh)
    ssh = ensure_nvidia_driver(ssh)
    ensure_docker_with_nvidia(ssh)
    upload_ctx(ssh)
    build_and_run(ssh)
    debug_health(ssh)
    ssh.close()
    print(f"\nServer is ✅ Ready! → http://{HOST}:{PORT}\n")