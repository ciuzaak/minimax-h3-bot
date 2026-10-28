#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import secrets
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from .config import BASE_PORT, COMFY_DIR, MODEL_DIR, I2V_TURBO_BACKEND, REFERENCE_TURBO_BACKEND, LOG_DIR, MAX_GPU_COUNT, OUTPUT_DIR, VIDEO_VAE
from .assets import ASSETS, verify_asset
from .profiles import (NATIVE_TURBO_LORA_NAME, NATIVE_REFERENCE_LORA_NAME, NATIVE_VIDEO_SHIFT,
                       NATIVE_AUDIO_SHIFT, VIDEO_VAES, RESOLUTION_PROFILES, resolution_for_size)


INPUT_DIR = COMFY_DIR / "input/minimax_h3_service"
SERVERS = []
POSSIBLE_SERVERS = [f"http://127.0.0.1:{port}" for port in range(BASE_PORT, BASE_PORT + MAX_GPU_COUNT)]
NO_MUSIC_FIELD = "non_diegetic_music: N/A"
TURBO_LORA_NAME = "h3/minimax_h3_turbo_v4_step600_ema.safetensors"
TURBO_STEPS = 8
TURBO_NODE_TYPES = ("MiniMaxH3TurboLoRA", "MiniMaxH3TurboSampler")
REFERENCE_MODEL_NAME = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
REFERENCE_TURBO_LORA_NAME = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
REFERENCE_TURBO_STEPS = 4
MAX_REFERENCE_IMAGES = 9
PROMPT_COUNT_PATTERN = re.compile(r"^(.*?)(?<![A-Za-z0-9_])[xX]([1-9][0-9]*)\s*$", re.DOTALL)


def extract_prompt_count(prompt):
    """Return a prompt and an optional trailing positive batch count."""
    prompt = prompt.strip()
    match = PROMPT_COUNT_PATTERN.match(prompt)
    if not match:
        return prompt, None
    clean_prompt = match.group(1).rstrip()
    if not clean_prompt:
        return prompt, None
    return clean_prompt, int(match.group(2))


def request_json(url, data=None, timeout=30):
    body = None if data is None else json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from error


def job_image_paths(job):
    task = job.get("task", "i2v")
    if task not in ("i2v", "reference"):
        raise ValueError("task must be i2v or reference")
    paths = job.get("image_paths", [job["image_path"]] if job.get("image_path") else [])
    if not isinstance(paths, (list, tuple)) or not paths or not all(isinstance(p, str) and p for p in paths):
        raise ValueError("image_paths must be a non-empty list of image paths")
    if len(paths) > (MAX_REFERENCE_IMAGES if task == "reference" else 1):
        raise ValueError("reference supports up to 9 images; i2v requires exactly one image")
    if job.get("ref_image_size", "match") not in ("match", "max"):
        raise ValueError("ref_image_size must be match or max")
    return tuple(paths)


def job_profile(job):
    """Validate recipe choices before staging files or contacting a worker."""
    reference = job.get("task") == "reference"
    backend = job.get("turbo_backend", REFERENCE_TURBO_BACKEND if reference else I2V_TURBO_BACKEND)
    if backend not in ("legacy", "native", "disabled"):
        raise ValueError("turbo_backend must be legacy, native or disabled")
    if backend == "disabled" and job.get("accelerated", False):
        raise ValueError("Turbo is disabled; use standard mode or an explicit supported backend")
    vae = job.get("video_vae", VIDEO_VAE)
    if not isinstance(vae, str) or vae not in VIDEO_VAES:
        raise ValueError("video_vae must be fp16 or int8")
    resolution = job.get("resolution", "balanced")
    if not isinstance(resolution, str) or resolution not in RESOLUTION_PROFILES:
        raise ValueError("resolution must be balanced or native")
    return backend, vae, resolution


def load_jobs(path):
    jobs = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Line {line_number}: invalid JSON: {error}") from error
            if not isinstance(job, dict):
                raise ValueError(f"Line {line_number}: job must be a JSON object")
            missing = [key for key in ("prompt", "seconds") if key not in job]
            if missing:
                raise ValueError(f"Line {line_number}: missing fields: {', '.join(missing)}")
            paths = []
            for value in job_image_paths(job):
                image_path = Path(value).expanduser().resolve()
                if not image_path.is_file():
                    raise ValueError(f"Line {line_number}: image does not exist: {image_path}")
                paths.append(str(image_path))
            if not isinstance(job["prompt"], str) or not job["prompt"].strip():
                raise ValueError(f"Line {line_number}: prompt must be a non-empty string")
            job["image_path"] = paths[0]
            if job.get("task") == "reference" or "image_paths" in job:
                job["image_paths"] = paths
            if job.get("seed") in (None, ""):
                job["seed"] = secrets.randbelow(2 ** 64)
            else:
                job["seed"] = int(job["seed"])
            if not 0 <= job["seed"] < 2 ** 64:
                raise ValueError(f"Line {line_number}: seed must be between 0 and {2 ** 64 - 1}")
            job["seconds"] = float(job["seconds"])
            if not 0.2 <= job["seconds"] <= 15.0:
                raise ValueError(f"Line {line_number}: seconds must be between 0.2 and 15")
            job["steps"] = int(job.get("steps", 20))
            if not 1 <= job["steps"] <= 100:
                raise ValueError(f"Line {line_number}: steps must be between 1 and 100")
            if not isinstance(job.get("accelerated", False), bool):
                raise ValueError(f"Line {line_number}: accelerated must be true or false")
            job["accelerated"] = job.get("accelerated", False)
            job_profile(job)
            jobs.append(job)
    if not jobs:
        raise ValueError("The JSONL file contains no jobs")
    return jobs


def image_resolution(image_path, requested_width=None, requested_height=None, profile="balanced"):
    if requested_width is not None or requested_height is not None:
        if requested_width is None or requested_height is None:
            raise ValueError("width and height must be provided together")
        width, height = int(requested_width), int(requested_height)
    else:
        from PIL import Image

        with Image.open(image_path) as image:
            source_width, source_height = image.size
        width, height = resolution_for_size(source_width, source_height, profile)
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("width and height must be positive multiples of 32")
    return width, height


def staged_image_path(image_path):
    source = Path(image_path)
    digest = hashlib.sha256(str(source).encode("utf-8") + str(source.stat().st_mtime_ns).encode()).hexdigest()[:12]
    suffix = source.suffix.lower() or ".png"
    return INPUT_DIR / f"{digest}_{source.stem}{suffix}"


def stage_image(image_path):
    source = Path(image_path)
    destination = staged_image_path(source)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_mtime_ns < source.stat().st_mtime_ns:
        shutil.copy2(source, destination)
    return destination.relative_to(COMFY_DIR / "input").as_posix()


def discard_staged_image(image_path):
    try:
        staged_image_path(image_path).unlink(missing_ok=True)
    except OSError:
        pass


def frame_count(seconds):
    frames = max(5, round(seconds * 24))
    while frames % 17 != 5:
        frames += 1
    return frames


def output_prefix():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"video/minimax_h3_service/{timestamp}"


def enforce_no_background_music(prompt):
    """Force H3's non-diegetic music field off while preserving other audio cues."""
    lines = [
        line for line in prompt.rstrip().splitlines()
        if not line.lstrip().lower().startswith("non_diegetic_music:")
    ]
    return "\n".join(lines).rstrip() + f"\n\n{NO_MUSIC_FIELD}"


def build_prompt(job, index):
    image_paths = job_image_paths(job)
    backend, video_vae, resolution = job_profile(job)
    staged_images = [stage_image(path) for path in image_paths]
    reference = job.get("task") == "reference"
    width, height = image_resolution(image_paths[0], job.get("width"), job.get("height"), resolution)
    length = frame_count(job["seconds"])
    prefix = output_prefix()
    conditioned_prompt = enforce_no_background_music(job["prompt"])
    accelerated = bool(job.get("accelerated", False))
    native_turbo = accelerated and backend == "native"
    legacy_turbo = accelerated and not reference and backend == "legacy"
    if native_turbo:
        steps, sampler, scheduler = 8, "euler", "simple"
    elif accelerated:
        steps = REFERENCE_TURBO_STEPS if reference else TURBO_STEPS
        sampler, scheduler = ("minimax_h3_turbo" if legacy_turbo else "res_multistep"), "simple"
    else:
        steps = int(job.get("steps", 20))
        sampler, scheduler = job.get("sampler", "res_multistep"), job.get("scheduler", "simple")
    model = ["30", 0] if native_turbo else ["16", 0] if accelerated else ["2", 0]
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": staged_images[0]}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": REFERENCE_MODEL_NAME if reference else "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAES[video_vae]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "6": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": conditioned_prompt, "width": width, "height": height, "length": length, "first_frame": ["1", 0]}},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": job["seed"]}},
        "8": {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["6", 0]}},
        "9": {"class_type": "MiniMaxH3TurboSampler", "inputs": {}} if legacy_turbo else {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}},
        "10": {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": scheduler, "steps": steps, "denoise": 1.0}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["7", 0], "guider": ["8", 0], "sampler": ["9", 0], "sigmas": ["10", 0], "latent_image": ["6", 1]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "13": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["5", 0]}},
        "14": {"class_type": "CreateVideo", "inputs": {"images": ["12", 0], "fps": 24.0, "audio": ["13", 0], "bit_depth": 8}},
        "15": {"class_type": "SaveVideo", "inputs": {"video": ["14", 0], "filename_prefix": prefix, "format": "mp4", "codec": "h264"}},
    }
    if reference:
        inputs = {"clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0],
                  "prompt": conditioned_prompt, "width": width, "height": height,
                  "length": length, "ref_image_size": job.get("ref_image_size", "match")}
        for ordinal, image in enumerate(staged_images):
            node_id = "1" if ordinal == 0 else str(16 + ordinal)
            prompt[node_id] = {"class_type": "LoadImage", "inputs": {"image": image}}
            inputs[f"ref_images.ref_image_{ordinal}"] = [node_id, 0]
        prompt["6"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": inputs}
    if accelerated:
        prompt["16"] = {
            "class_type": "MiniMaxH3TurboLoRA",
            "inputs": {"model": ["2", 0], "lora_name": TURBO_LORA_NAME, "strength": 1.0, "low_vram": False},
        }
        if not legacy_turbo:
            lora_name = NATIVE_REFERENCE_LORA_NAME if reference else NATIVE_TURBO_LORA_NAME
            if not native_turbo:
                lora_name = REFERENCE_TURBO_LORA_NAME
            prompt["16"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                "model": ["2", 0], "lora_name": lora_name, "strength_model": 1.0}}
    if native_turbo:
        prompt["30"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {
            "model": ["16", 0], "shift_video": NATIVE_VIDEO_SHIFT, "shift_audio": NATIVE_AUDIO_SHIFT}}
    metadata = {
        "seed": job["seed"],
        "width": width,
        "height": height,
        "frames": length,
        "actual_seconds": length / 24,
        "output_prefix": prefix,
        "accelerated": accelerated,
        "steps": steps,
        "sampler": sampler,
        "scheduler": scheduler,
        "task": "reference" if reference else "i2v",
        "reference_count": len(image_paths) if reference else 0,
        "turbo_backend": backend if accelerated else None,
        "video_vae": video_vae,
        "resolution": resolution,
        "diffusion_model": prompt["2"]["inputs"]["unet_name"],
        "lora": prompt["16"]["inputs"]["lora_name"] if accelerated else None,
        "shift_video": NATIVE_VIDEO_SHIFT if native_turbo else 12.0,
        "shift_audio": NATIVE_AUDIO_SHIFT if native_turbo else 3.0,
        "experimental": native_turbo or video_vae == "int8",
        "lora_sha256": ASSETS["native-reference-turbo" if reference else "native-turbo"][1] if native_turbo else None,
        "video_vae_sha256": ASSETS["int8-vae"][1] if video_vae == "int8" else None,
    }
    return prompt, metadata


def sampler_progress(server):
    """Return the latest sampler percentage for a worker from its ComfyUI log."""
    port = int(server.rsplit(":", 1)[1])
    gpu = port - BASE_PORT
    log_path = LOG_DIR / f"minimax-h3-service-gpu{gpu}.log"
    try:
        with log_path.open("rb") as stream:
            stream.seek(max(0, log_path.stat().st_size - 2 * 1024 * 1024))
            data = stream.read()
    except OSError:
        return None
    prompt_start = data.rfind(b"got prompt")
    if prompt_start < 0:
        return None
    data = data[prompt_start:]
    starts = list(re.finditer(rb"(?<!\d)0%\|", data))
    if not starts:
        return None
    percentages = re.findall(rb"(\d{1,3})%\|", data[starts[-1].start():])
    return min(int(percentages[-1]), 100) if percentages else None


def server_schedule_key(server):
    """Prefer fewer jobs ahead, then the running worker closest to completion."""
    queue = request_json(f"{server}/queue")
    running = len(queue.get("queue_running", []))
    pending = len(queue.get("queue_pending", []))
    jobs_ahead = running + pending
    progress = sampler_progress(server) if running else 100
    if progress is None:
        progress = 0
    return jobs_ahead, -progress, server


def select_server(servers):
    return min(servers, key=server_schedule_key)


def check_servers():
    available = []
    for server in POSSIBLE_SERVERS:
        try:
            request_json(f"{server}/system_stats", timeout=0.5)
        except Exception:
            continue
        available.append(server)
    if not available:
        raise RuntimeError(
            f"No MiniMax H3 service endpoints are available on ports "
            f"{BASE_PORT}-{BASE_PORT + MAX_GPU_COUNT - 1}"
        )
    return available


def check_turbo_support(servers=None, backend=None):
    """Return worker capability problems that would make Turbo jobs fail."""
    servers = list(servers or check_servers())
    backend = I2V_TURBO_BACKEND if backend is None else backend
    if backend == "disabled":
        return ["I2V Turbo 已禁用"]
    if backend not in ("legacy", "native"):
        return ["未知的 I2V Turbo backend"]
    node_types = TURBO_NODE_TYPES if backend == "legacy" else ("LoraLoaderModelOnly", "KSamplerSelect", "MiniMaxH3SigmaShift")
    lora_name = TURBO_LORA_NAME if backend == "legacy" else NATIVE_TURBO_LORA_NAME
    issues = verify_asset("native-turbo", MODEL_DIR, COMFY_DIR) if backend == "native" else []
    for server in servers:
        try:
            for node_type in node_types:
                info = request_json(f"{server}/object_info/{node_type}", timeout=3)
                if node_type not in info:
                    issues.append(f"{server} 缺少节点 {node_type}")
            loras = request_json(f"{server}/models/loras", timeout=3)
            if lora_name not in loras:
                issues.append(f"{server} 找不到 LoRA {lora_name}")
        except Exception as error:
            issues.append(f"{server} 预检失败：{error}")
    return issues


def check_reference_support(servers=None, accelerated=False, backend=None):
    backend = REFERENCE_TURBO_BACKEND if backend is None else backend
    if accelerated and backend not in ("legacy", "native"):
        return ["Ref2VA Turbo 已禁用或 backend 无效"]
    lora_name = NATIVE_REFERENCE_LORA_NAME if backend == "native" else REFERENCE_TURBO_LORA_NAME
    issues = verify_asset("native-reference-turbo", MODEL_DIR, COMFY_DIR) if accelerated and backend == "native" else []
    for server in list(servers or check_servers()):
        try:
            info = request_json(f"{server}/object_info/MiniMaxH3ReferenceToVideo", timeout=3)
            if "MiniMaxH3ReferenceToVideo" not in info:
                issues.append(f"{server} 缺少多图参考节点")
            models = request_json(f"{server}/models/diffusion_models", timeout=3)
            if REFERENCE_MODEL_NAME not in models:
                issues.append(f"{server} 找不到 Ref2VA 模型")
            if accelerated:
                if backend == "native":
                    info = request_json(f"{server}/object_info/MiniMaxH3SigmaShift", timeout=3)
                    if "MiniMaxH3SigmaShift" not in info:
                        issues.append(f"{server} 缺少 Ref2VA 768p shift 节点")
                info = request_json(f"{server}/object_info/LoraLoaderModelOnly", timeout=3)
                if "LoraLoaderModelOnly" not in info:
                    issues.append(f"{server} 缺少 Ref2VA Turbo 加载节点")
                loras = request_json(f"{server}/models/loras", timeout=3)
                if lora_name not in loras:
                    issues.append(f"{server} 找不到 Ref2VA {'768p 8' if backend == 'native' else '4'} 步 LoRA {lora_name}")
        except Exception as error:
            issues.append(f"{server} Ref2VA 预检失败：{error}")
    return issues


def check_video_vae_support(server, video_vae):
    """Fail before enqueueing an INT8 VAE on an old or incomplete worker."""
    if video_vae == "fp16":
        return []
    issues = verify_asset("int8-vae", MODEL_DIR, COMFY_DIR)
    if issues:
        return issues
    try:
        stats = request_json(f"{server}/system_stats", timeout=3)
        version = stats.get("system", {}).get("comfyui_version", "")
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
        if match is None or tuple(map(int, match.groups())) < (0, 36, 0):
            return [f"{server} INT8 video VAE 需要 ComfyUI v0.36.0+，当前版本：{version or '未知'}"]
        models = request_json(f"{server}/models/vae", timeout=3)
        if VIDEO_VAES[video_vae] not in models:
            return [f"{server} 找不到 INT8 video VAE"]
    except Exception as error:
        return [f"{server} INT8 video VAE 预检失败：{error}"]
    return []


def check_job_support(server, prompt, metadata):
    """Check the selected graph only; missing optional recipes cannot block it."""
    issues = []
    if metadata["turbo_backend"] == "native":
        issues.extend(verify_asset("native-reference-turbo" if metadata["task"] == "reference" else "native-turbo", MODEL_DIR, COMFY_DIR))
    stats = request_json(f"{server}/system_stats", timeout=3)
    version = stats.get("system", {}).get("comfyui_version", "")
    metadata["comfyui_version"] = version
    if metadata["turbo_backend"] == "native":
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
        if match is None or tuple(map(int, match.groups())) < (0, 31, 0):
            issues.append(f"{server} 768p Turbo 需要 ComfyUI v0.31.0+，当前版本：{version or '未知'}")
    for node_type in sorted({node["class_type"] for node in prompt.values()}):
        info = request_json(f"{server}/object_info/{node_type}", timeout=3)
        if node_type not in info:
            issues.append(f"{server} 缺少所选配方节点 {node_type}")
    required_models = {
        "diffusion_models": [prompt["2"]["inputs"]["unet_name"]],
        "text_encoders": [prompt["3"]["inputs"]["clip_name"]],
        "vae": [prompt["4"]["inputs"]["vae_name"], prompt["5"]["inputs"]["vae_name"]],
    }
    if metadata["accelerated"]:
        required_models["loras"] = [metadata["lora"]]
    for folder, names in required_models.items():
        available = request_json(f"{server}/models/{folder}", timeout=3)
        issues.extend(f"{server} 找不到 {folder}/{name}" for name in names if name not in available)
    return issues


def submit_job(job, index):
    prompt, metadata = build_prompt(job, index)
    server = select_server(SERVERS)
    vae_issues = check_video_vae_support(server, metadata["video_vae"])
    if vae_issues:
        raise RuntimeError("；".join(vae_issues))
    issues = check_job_support(server, prompt, metadata)
    if issues:
        raise RuntimeError("所选配方不可用：" + "；".join(issues))
    submitted_at_ms = round(time.time() * 1000)
    response = request_json(f"{server}/prompt", {"prompt": prompt, "client_id": str(uuid.uuid4()), "extra_data": {"minimax_h3_job": job}})
    if "prompt_id" not in response:
        raise RuntimeError(f"Submission rejected by {server}: {response}")
    return {
        "index": index,
        "server": server,
        "prompt_id": response["prompt_id"],
        "status": "queued",
        "submitted_at_ms": submitted_at_ms,
        **metadata,
    }


def poll_job(record):
    prompt_id = record["prompt_id"]
    server = record["server"]
    history = request_json(f"{server}/history/{prompt_id}", timeout=30)
    if prompt_id not in history:
        return None
    item = history[prompt_id]
    status = item.get("status", {})
    if not status.get("completed", False):
        if status.get("status_str") == "error":
            record["status"] = "error"
            record["messages"] = status.get("messages", [])
            return record
        return None
    outputs = []
    for node_output in item.get("outputs", {}).values():
        for value in node_output.values():
            if not isinstance(value, list):
                continue
            for file_info in value:
                if not isinstance(file_info, dict) or "filename" not in file_info:
                    continue
                subfolder = file_info.get("subfolder", "")
                path = OUTPUT_DIR / subfolder / file_info["filename"]
                outputs.append(str(path.resolve()))
    record["status"] = "completed"
    record["outputs"] = outputs
    event_times = {}
    for message in status.get("messages", []):
        if not isinstance(message, (list, tuple)) or len(message) != 2:
            continue
        event, detail = message
        if isinstance(detail, dict) and detail.get("timestamp") is not None:
            event_times[event] = detail["timestamp"]
    started_at = event_times.get("execution_start")
    completed_at = event_times.get("execution_success")
    submitted_at = record.get("submitted_at_ms")
    if started_at is not None and submitted_at is not None:
        record["queue_seconds"] = max(0.0, (started_at - submitted_at) / 1000)
    if completed_at is not None and started_at is not None:
        record["render_seconds"] = max(0.0, (completed_at - started_at) / 1000)
    if completed_at is not None and submitted_at is not None:
        record["total_seconds"] = max(0.0, (completed_at - submitted_at) / 1000)
    return record


def wait_for_job(record):
    while True:
        result = poll_job(record)
        if result is not None:
            return result
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description="Submit MiniMax H3 image-to-video jobs from JSONL to the multi-GPU service.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--no-wait", action="store_true", help="Return after all jobs are queued instead of waiting for completion.")
    parser.add_argument("--results", type=Path, help="Results JSONL path (default: INPUT.results.jsonl).")
    args = parser.parse_args()

    jobs = load_jobs(args.jsonl.resolve())
    SERVERS.extend(check_servers())
    print(f"Discovered {len(SERVERS)} worker(s): {', '.join(SERVERS)}", flush=True)
    records = []
    for index, job in enumerate(jobs, 1):
        record = submit_job(job, index)
        records.append(record)
        print(f"[{index}/{len(jobs)}] queued {record['prompt_id']} on {record['server']} (seed {record['seed']}, {record['width']}x{record['height']}, {record['actual_seconds']:.3f}s)", flush=True)

    if not args.no_wait:
        with ThreadPoolExecutor(max_workers=len(SERVERS)) as executor:
            futures = {executor.submit(wait_for_job, record): record for record in records}
            completed = []
            for future in as_completed(futures):
                record = future.result()
                completed.append(record)
                print(f"[{record['index']}/{len(jobs)}] {record['status']}: {', '.join(record.get('outputs', [])) or record['prompt_id']}", flush=True)
            records = sorted(completed, key=lambda item: item["index"])

    results_path = args.results or args.jsonl.with_suffix(".results.jsonl")
    with results_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Results: {results_path.resolve()}")
    if any(record["status"] == "error" for record in records):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
