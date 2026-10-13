"""Run a prepared attention comparison on one idle GPU and owned workers only."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

from minimax_h3 import jobs


VARIANTS = ("sage_on", "sage_off")
MODEL_FILES = (
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae/minimax_h3_video_vae_fp16.safetensors",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_plan(path: Path) -> tuple[dict, dict[str, list[dict]]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or plan.get("variable") != "MINIMAX_USE_SAGE_ATTENTION":
        raise ValueError("Unsupported attention comparison plan")
    variants = {}
    for name in VARIANTS:
        entry = plan["variants"][name]
        expected = "1" if name == "sage_on" else "0"
        if entry["environment"] != {"MINIMAX_USE_SAGE_ATTENTION": expected}:
            raise ValueError(f"Unexpected worker settings for {name}")
        source = (path.parent / entry["jobs"]).resolve()
        source.relative_to(path.parent.resolve())
        if file_hash(source) != plan["jobs_sha256"]:
            raise ValueError(f"Task file changed: {name}")
        batch = jobs.load_jobs(source)
        if len(batch) != plan["job_count_per_variant"] or len(batch) != len(plan["pairs"]):
            raise ValueError("Task count does not match the plan")
        for index, (job, pair) in enumerate(zip(batch, plan["pairs"]), 1):
            if pair["index"] != index or job["seed"] != pair["seed"]:
                raise ValueError("Task pairing does not match the plan")
            image = plan["inputs"][pair["scene"] - 1]
            if file_hash(Path(job["image_path"])) != image["image_sha256"]:
                raise ValueError("Input image changed since preparation")
        variants[name] = batch
    if variants["sage_on"] != variants["sage_off"]:
        raise ValueError("The variants must use identical jobs")
    return plan, variants


def gpu_query(gpu: int, fields: str) -> str:
    return subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True, timeout=10,
    ).strip()


def require_idle_gpu(gpu: int) -> None:
    processes = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True, timeout=10,
    ).strip()
    if processes:
        raise RuntimeError(f"GPU {gpu} has existing compute processes; refusing to use it")


class MemorySampler:
    """Sample device memory, not PyTorch allocator memory, at one-second intervals."""

    def __init__(self, gpu: int):
        self.gpu = gpu
        self.samples = []
        self.errors = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        while not self.stop.is_set():
            try:
                memory = int(gpu_query(self.gpu, "memory.used"))
                self.samples.append({"timestamp": time.time(), "memory_mib": memory})
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self.errors.append(str(error))
            self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=12)


@contextmanager
def owned_worker(command: list[str], cwd: Path, env: dict, log_path: Path,
                 server: str, startup_timeout: float):
    port = int(server.rsplit(":", 1)[1])
    with socket.socket() as probe:
        # Match aiohttp's reuse behavior; TIME_WAIT is not an active listener.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    process = None
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Worker exited with {process.returncode}; see {log_path}")
                try:
                    jobs.request_json(f"{server}/system_stats", timeout=1)
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise TimeoutError(f"Worker startup timed out; see {log_path}")
            yield process
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)


def wait_job(record: dict, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = jobs.poll_job(record)
        if result is not None:
            return result
        time.sleep(1)
    raise TimeoutError(f"Generation timed out: {record['prompt_id']}")


def attention_evidence(log_path: Path) -> dict:
    log = log_path.read_text(encoding="utf-8", errors="replace").lower()
    lines = [line.strip() for line in log.splitlines()]
    return {
        "sage_selected": any(line.endswith("using sage attention") for line in lines),
        "pytorch_selected": any(line.endswith("using pytorch attention") for line in lines),
        "fallback_log_count": log.count("using pytorch attention instead"),
        "note": "Log counts do not quantify per-layer coverage or prove no unlogged fallback.",
    }


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def completed_variant(path: Path, batch: list[dict]) -> bool:
    if not path.exists():
        return False
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (len(rows) != len(batch) or any(row.get("index") != index or
            row.get("seed") != job["seed"] or row.get("status") != "completed"
            for index, (row, job) in enumerate(zip(rows, batch), 1))):
        raise ValueError("Cannot resume a partially completed variant; use a new experiment directory")
    if any(not row.get("outputs") or any(not Path(output).is_file() for output in row["outputs"])
           for row in rows):
        raise ValueError("Completed variant outputs are missing")
    return True


def run(args) -> None:
    plan, variants = load_plan(args.plan)
    require_idle_gpu(args.gpu)
    previous = None
    if args.resume:
        previous = json.loads((args.output_dir / "provenance.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    model_paths = args.output_dir / "models.yaml"
    provenance = {
        "plan_sha256": file_hash(args.plan), "gpu": gpu_query(args.gpu, "name,uuid,driver_version"),
        "comfy_commit": subprocess.check_output(["git", "-C", str(args.comfy_dir), "rev-parse", "HEAD"], text=True).strip(),
        "turbo_commit": subprocess.check_output(["git", "-C", str(args.comfy_dir / "custom_nodes/ComfyUI-MiniMax-H3-Turbo"), "rev-parse", "HEAD"], text=True).strip(),
        "models": {}, "variants": {}, "warmups": args.warmups, "reserve_vram": args.reserve_vram,
    }
    if previous is None:
        write_json(args.output_dir / "provenance.json", provenance)
    for relative in MODEL_FILES:
        print(f"Hashing {relative}", flush=True)
        provenance["models"][relative] = file_hash(args.model_dir / relative)
    provenance["models"][jobs.TURBO_LORA_NAME] = file_hash(
        args.comfy_dir / "models/loras" / jobs.TURBO_LORA_NAME
    )
    version_code = (
        "import sys,json,importlib.metadata as m; "
        "print(json.dumps({'python':sys.version,'packages':{k:m.version(k) for k in "
        "['torch','sageattention','comfy-kitchen','comfy-aimdo','aiohttp','Pillow']}}))"
    )
    provenance["runtime"] = json.loads(subprocess.check_output([str(args.comfy_python), "-c", version_code], text=True))
    if previous is not None:
        for key in ("plan_sha256", "gpu", "comfy_commit", "turbo_commit", "models", "runtime", "warmups", "reserve_vram"):
            if previous[key] != provenance[key]:
                raise ValueError(f"Cannot resume with changed provenance: {key}")
        provenance["variants"] = previous["variants"]
    write_json(args.output_dir / "provenance.json", provenance)
    model_paths.write_text(
        "minimax_h3:\n  base_path: " + json.dumps(str(args.model_dir)) +
        "\n  diffusion_models: diffusion_models\n  text_encoders: text_encoders\n  vae: vae\n",
        encoding="utf-8",
    )

    jobs.COMFY_DIR = args.comfy_dir
    server = f"http://127.0.0.1:{args.port}"
    jobs.SERVERS[:] = [server]
    for name in VARIANTS:
        directory = args.output_dir / name
        if args.resume and completed_variant(directory / "results.jsonl", variants[name]):
            provenance["variants"][name]["attention"] = attention_evidence(directory / "worker.log")
            write_json(args.output_dir / "provenance.json", provenance)
            print(f"Keeping completed variant: {name}", flush=True)
            continue
        require_idle_gpu(args.gpu)
        directory.mkdir(exist_ok=args.resume)
        for folder in ("output", "temp", "user"):
            (directory / folder).mkdir(exist_ok=args.resume)
        # stage_image returns paths relative to COMFY_DIR/input.
        # Use a dedicated staging root inside the isolated ComfyUI checkout.
        jobs.INPUT_DIR = args.comfy_dir / "input/minimax_h3_attention_ab" / name
        jobs.OUTPUT_DIR = directory / "output"
        command = [str(args.comfy_python), "main.py", "--listen", "127.0.0.1", "--port", str(args.port),
                   "--output-directory", str(jobs.OUTPUT_DIR), "--temp-directory", str(directory / "temp"),
                   "--user-directory", str(directory / "user"), "--extra-model-paths-config", str(model_paths),
                   "--reserve-vram", str(args.reserve_vram), "--disable-auto-launch", "--disable-all-custom-nodes",
                   "--whitelist-custom-nodes", "ComfyUI-MiniMax-H3-Turbo", "--database-url", "sqlite:///:memory:"]
        if name == "sage_on":
            command.append("--use-sage-attention")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
                   PYTORCH_ALLOC_CONF="expandable_segments:True", PYTHONDONTWRITEBYTECODE="1")
        for key in ("TG_BOT_TOKEN", "TG_ALLOWLIST"):
            env.pop(key, None)
        records = []
        log_path = directory / "worker.log"
        provenance["variants"][name] = {"command": command, "gpu_index": args.gpu}
        write_json(args.output_dir / "provenance.json", provenance)
        print(f"Starting {name} on GPU {args.gpu}", flush=True)
        with owned_worker(command, args.comfy_dir, env, log_path, server, args.startup_timeout):
            issues = jobs.check_turbo_support([server]) if any(j.get("accelerated") for j in variants[name]) else []
            if issues:
                raise RuntimeError("; ".join(issues))
            warmup = dict(variants[name][0], seconds=0.9, seed=20261013)
            warmup_records = []
            for index in range(args.warmups):
                print(f"Warmup {index + 1}/{args.warmups}: {name}", flush=True)
                record = wait_job(jobs.submit_job(warmup, 0), args.job_timeout)
                warmup_records.append(record)
                write_json(directory / "warmups.json", {"jobs": warmup_records})
                if record["status"] != "completed":
                    write_json(directory / "warmup_error.json", record)
                    raise RuntimeError(f"Warmup failed; see {directory}")
            for index, job in enumerate(variants[name], 1):
                print(f"Running {name} {index}/{len(variants[name])}: seed={job['seed']} seconds={job['seconds']}", flush=True)
                with MemorySampler(args.gpu) as memory:
                    record = wait_job(jobs.submit_job(job, index), args.job_timeout)
                record["observed_device_memory_peak_mib"] = max((sample["memory_mib"] for sample in memory.samples), default=None)
                record["memory_samples"] = memory.samples
                record["memory_sampling_errors"] = memory.errors
                records.append(record)
                (directory / "results.jsonl").write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
                )
                print(f"Completed {name} {index}: {record['status']} render={record.get('render_seconds')}s", flush=True)
                if record["status"] != "completed":
                    raise RuntimeError(f"Generation failed; see {directory / 'results.jsonl'}")
        provenance["variants"][name]["attention"] = attention_evidence(log_path)
        write_json(args.output_dir / "provenance.json", provenance)
        print(f"Stopped owned worker: {name}", flush=True)
    print(f"Comparison complete: {args.output_dir}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--comfy-dir", type=Path, required=True)
    parser.add_argument("--comfy-python", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, default=18290)
    parser.add_argument("--reserve-vram", type=float, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Keep completed variants after verifying identical provenance")
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--job-timeout", type=float, default=1800)
    args = parser.parse_args()
    for name in ("plan", "comfy_dir", "model_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    # Preserve the venv interpreter path: resolving its symlink changes sys.prefix.
    args.comfy_python = args.comfy_python.absolute()
    if args.gpu < 0 or not 1024 <= args.port <= 65535 or args.warmups < 1:
        parser.error("Use a nonnegative GPU, an unprivileged port and at least one warmup")
    if args.startup_timeout <= 0 or args.job_timeout <= 0 or args.reserve_vram < 0:
        parser.error("Timeouts must be positive and reserve VRAM must be nonnegative")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
