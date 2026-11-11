"""Persistent video/image GPU allocation, with video priority and graceful draining."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import tempfile
import time
import urllib.request

from .config import (BASE_PORT, COMFY_DIR, IMAGE_SERVER, LOG_DIR, MAX_GPU_COUNT,
                     MODEL_DIR, OUTPUT_DIR, PROJECT_DIR, ensure_model_paths_config)

STATE_PATH = PROJECT_DIR / ".runtime/gpu_pools.json"
ROLES = ("video", "image")


def empty_state():
    return {"version": 0, "requested": {"video": 0, "image": 0}, "workers": [], "error": "", "retry_after": {}}


def read_state(path=None):
    path = Path(path or STATE_PATH)
    if not path.exists():
        return empty_state()
    state = json.loads(path.read_text())
    if (not isinstance(state, dict) or type(state.get("version")) is not int
            or not isinstance(state.get("workers"), list)
            or not isinstance(state.get("requested"), dict)
            or any(type(state["requested"].get(role)) is not int
                   or not 0 <= state["requested"][role] <= MAX_GPU_COUNT for role in ROLES)):
        raise ValueError("Invalid GPU pool state; refusing to change workers")
    seen = set()
    for worker in state["workers"]:
        if (not isinstance(worker, dict) or worker.get("role") not in ROLES
                or type(worker.get("gpu")) is not int or worker["gpu"] < 0
                or worker.get("status") not in ("starting", "ready", "draining")):
            raise ValueError("Invalid GPU worker record")
        expected = new_worker(worker["role"], worker["gpu"])
        if any(worker.get(key) != expected[key] for key in ("session", "server", "log")):
            raise ValueError("GPU worker record does not match the configured runtime")
        if worker["gpu"] in seen or not isinstance(worker.get("started"), (int, float)):
            raise ValueError("Duplicate GPU or invalid worker start time")
        seen.add(worker["gpu"])
    if not isinstance(state.get("retry_after"), dict):
        state["retry_after"] = {}
    return state


@contextmanager
def allocation_lock(path=None):
    path = Path(path or STATE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as stream:
        os.chmod(stream.name, 0o600)
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def save_state(state, path=None):
    path = Path(path or STATE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(state, stream, ensure_ascii=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def set_requested(video=None, image=None, expected_version=None):
    with allocation_lock():
        state = read_state()
        if expected_version is not None and expected_version != state["version"]:
            raise ValueError("资源设置已被修改，请刷新面板后重试。")
        for role, count in (("video", video), ("image", image)):
            if count is not None:
                if type(count) is not int or not 0 <= count <= MAX_GPU_COUNT:
                    raise ValueError(f"{role} GPU count must be 0–{MAX_GPU_COUNT}")
                state["requested"][role] = count
                state["retry_after"][role] = 0
        state["version"] += 1
        state["error"] = ""
        save_state(state)
        return state


def servers(role, include_draining=False):
    state = read_state()
    if not include_draining and state["requested"][role] == 0:
        return []
    statuses = ("ready", "draining") if include_draining else ("ready",)
    return [w["server"] for w in state["workers"] if w["role"] == role and w["status"] in statuses
            and (include_draining or not w.get("retire"))]


def plan(requested, workers, free):
    """Keep existing video GPUs first; use free GPUs before reclaiming image GPUs."""
    video = [w["gpu"] for w in workers if w["role"] == "video"]
    image = [w["gpu"] for w in workers if w["role"] == "image"]
    free = sorted(set(free) - set(video) - set(image))
    # reconcile orders busy workers first; reclaim idle image workers first.
    order = list(dict.fromkeys(video + free + list(reversed(image))))
    video_target = order[:requested["video"]]
    remaining = [gpu for gpu in image + free + video if gpu not in video_target]
    image_target = list(dict.fromkeys(remaining))[:requested["image"]]
    return {**dict.fromkeys(video_target, "video"), **dict.fromkeys(image_target, "image")}


def new_worker(role, gpu):
    port = (BASE_PORT if role == "video" else int(IMAGE_SERVER.rsplit(":", 1)[1])) + gpu
    return dict(role=role, gpu=gpu, session=f"minimax-h3-pool-{role}-gpu{gpu}",
                server=f"http://127.0.0.1:{port}", log=str(LOG_DIR / f"minimax-h3-pool-{role}-gpu{gpu}.log"),
                status="starting", started=time.time())


class Runtime:
    def run(self, *args, **kwargs):
        return subprocess.run(args, check=True, capture_output=True, text=True, timeout=10, **kwargs).stdout

    def free_gpus(self):
        rows = self.run("nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits")
        candidates = [int(parts[0]) for row in rows.splitlines()
                      if len(parts := row.split(",")) == 3 and int(parts[1]) < 512 and int(parts[2]) < 10]
        return [gpu for gpu in candidates if not any(line.strip().isdigit() for line in self.run(
            "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits").splitlines())]

    def exists(self, worker):
        return subprocess.run(["tmux", "has-session", "-t", "=" + worker["session"]],
                              capture_output=True, timeout=5).returncode == 0

    def request(self, worker, endpoint):
        with urllib.request.urlopen(worker["server"] + endpoint, timeout=1) as response:
            return json.load(response)

    def ready(self, worker):
        try:
            self.request(worker, "/system_stats")
            return True
        except (OSError, ValueError):
            return False

    def busy(self, worker):
        try:
            queue = self.request(worker, "/queue")
            if any(not isinstance(queue.get(key), list) for key in ("queue_running", "queue_pending")):
                return True
            return bool(queue.get("queue_running") or queue.get("queue_pending"))
        except (OSError, ValueError):
            # A failed status request must never cause an active job to be killed.
            return worker["status"] != "starting"

    def stop(self, worker):
        self.run("tmux", "kill-session", "-t", "=" + worker["session"])

    def launch(self, role, gpu):
        comfy = Path(os.environ.get("MINIMAX_IMAGE_COMFY_DIR", PROJECT_DIR / ".runtime/ComfyUI-image"))
        python = os.environ.get("MINIMAX_IMAGE_PYTHON", os.environ.get("COMFY_PYTHON", "python3"))
        if not (comfy / "main.py").is_file():
            raise RuntimeError("Set up MINIMAX_IMAGE_COMFY_DIR first")
        if role == "video" and not (COMFY_DIR / "custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py").is_file():
            raise RuntimeError("Missing video Turbo custom node")
        models = [("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")]
        models += ([("vae", "minimax_h3_image_vae_fp16.safetensors")] if role == "image" else
                   [("diffusion_models", "minimax_h3_fl2va_pruned_int8_convrot.safetensors")] +
                   [("vae", "minimax_h3_video_vae_fp16.safetensors"), ("vae", "minimax_h3_audio_vae_fp32.safetensors")])
        if role == "image":
            from .image_jobs import QUANTS
            if not any((MODEL_DIR / "diffusion_models" / name).is_file() for name in QUANTS.values()):
                raise RuntimeError("Missing image DiT weights; download W4A8 or INT8")
        for category, name in models:
            if not (MODEL_DIR / category / name).is_file():
                raise RuntimeError(f"Missing {role} model: {name}")
        worker = new_worker(role, gpu)
        port = worker["server"].rsplit(":", 1)[1]
        session, logfile = worker["session"], Path(worker["log"])
        try:
            connection = socket.create_connection(("127.0.0.1", int(port)), timeout=0.2)
        except OSError:
            pass
        else:
            connection.close()
            raise RuntimeError(f"Worker port {port} is already in use")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (COMFY_DIR / "input").mkdir(parents=True, exist_ok=True)
        command = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", "PYTORCH_ALLOC_CONF=expandable_segments:True",
                   python, str(comfy / "main.py"), "--listen", "127.0.0.1", "--port", str(port),
                   "--lowvram", "--reserve-vram", os.environ.get("MINIMAX_IMAGE_RESERVE_VRAM", "3"),
                   "--input-directory", str(COMFY_DIR / "input"), "--output-directory", str(OUTPUT_DIR),
                   "--extra-model-paths-config", str(ensure_model_paths_config()),
                   "--disable-auto-launch", "--disable-all-custom-nodes", "--database-url", "sqlite:///:memory:"]
        if role == "video":
            command += ["--base-directory", str(COMFY_DIR), "--whitelist-custom-nodes", "ComfyUI-MiniMax-H3-Turbo"]
        self.run("tmux", "new-session", "-d", "-s", session, "-n", "worker",
                 "exec " + shlex.join(command) + " > " + shlex.quote(str(logfile)) + " 2>&1")
        return worker


def reconcile(runtime=None):
    runtime = runtime or Runtime()
    with allocation_lock():
        state = read_state()
        workers = []
        for worker in state["workers"]:
            if not runtime.exists(worker):
                state["error"] = f"{worker['role']} GPU {worker['gpu']} worker exited; inspect its log."
                state["retry_after"][worker["role"]] = time.time() + 60
                continue
            if worker["status"] == "starting" and runtime.ready(worker):
                worker["status"] = "ready"
            if worker["status"] == "starting" and time.time() - worker["started"] > 180:
                runtime.stop(worker)
                state["error"] = f"{worker['role']} GPU {worker['gpu']} startup timed out"
                state["retry_after"][worker["role"]] = time.time() + 60
                continue
            workers.append(worker)
        # Keep busy workers when reducing a pool, so fewer jobs need to drain.
        workers.sort(key=lambda w: (w["role"], not runtime.busy(w), w["gpu"]))
        targets = plan(state["requested"], workers, runtime.free_gpus())
        kept = []
        for worker in workers:
            if targets.get(worker["gpu"]) != worker["role"] or worker.get("retire"):
                busy = runtime.busy(worker)
                worker["status"] = "draining"
                if not busy:
                    runtime.stop(worker)
                    continue
            elif worker["status"] == "draining" and runtime.ready(worker):
                worker["status"] = "ready"
            kept.append(worker)
        occupied = {w["gpu"] for w in kept}
        free = runtime.free_gpus()
        for gpu, role in targets.items():
            if gpu in occupied or gpu not in free or time.time() < state["retry_after"].get(role, 0):
                continue
            worker = new_worker(role, gpu)
            kept.append(worker)
            state["workers"] = kept
            save_state(state)  # Journal ownership before spawning the external process.
            try:
                kept[-1] = runtime.launch(role, gpu)
            except Exception as error:
                if not runtime.exists(worker):
                    kept.remove(worker)
                state["error"] = str(error)
                state["retry_after"][role] = time.time() + 60
        state["workers"] = kept
        save_state(state)
        return state


def restart_pool(role):
    if role not in ROLES:
        raise ValueError("Unknown worker pool")
    with allocation_lock():
        state = read_state()
        for worker in state["workers"]:
            if worker["role"] == role:
                worker["retire"] = True
        save_state(state)
    return reconcile()


def describe(state):
    lines = ["GPU 分配（全局设置，视频优先）"]
    for role, label in (("video", "视频"), ("image", "图像")):
        workers = [w for w in state["workers"] if w["role"] == role]
        ready = [w for w in workers if w["status"] == "ready" and not w.get("retire")]
        cards = "、".join(str(w["gpu"]) + ("（退出中）" if w["status"] == "draining" else "（启动中）" if w["status"] == "starting" else "") for w in workers) or "无"
        lines.append(f"{label}：目标 {state['requested'][role]} 张，可接任务 {len(ready)} 张；GPU {cards}")
    lines.append("0 张即关闭该类服务。卡不足先满足视频；已有任务完成后再释放卡。")
    if state.get("error"):
        lines.append("最近错误：" + state["error"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "set", "reconcile", "restart"))
    parser.add_argument("--video", type=int)
    parser.add_argument("--image", type=int)
    parser.add_argument("--role", choices=ROLES)
    args = parser.parse_args()
    if args.action == "set":
        set_requested(args.video, args.image)
    if args.action == "restart":
        if not args.role:
            parser.error("restart needs --role")
        state = restart_pool(args.role)
    else:
        state = read_state() if args.action == "status" else reconcile()
    print(describe(state))


if __name__ == "__main__":
    main()
