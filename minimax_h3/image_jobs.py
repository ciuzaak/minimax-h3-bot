"""Native H3-Image jobs, isolated from the video workers and their dependencies."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import secrets
import sys
import uuid

from . import jobs
from .config import IMAGE_SERVER, MANAGED_POOLS

SERVER = IMAGE_SERVER
QUANTS = {
    "int8": "minimax_h3_image_pruned_int8_convrot.safetensors",
    "w4a8": "minimax_h3_image_pruned_w4a8.safetensors",
}
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = "minimax_h3_image_vae_fp16.safetensors"
PROFILES = {"preview": ("预览 4MP", 2048), "balanced": ("均衡 8MP", 2880), "native": ("原生 16MP", 4096)}
RATIOS = {"1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "16:9": (16, 9), "9:16": (9, 16)}
MAX_IMAGES = 16


def canvas(profile="native", ratio="1:1"):
    if profile not in PROFILES or (isinstance(ratio, str) and ratio not in RATIOS):
        raise ValueError("Unknown image profile or aspect ratio")
    edge = PROFILES[profile][1]
    w, h = RATIOS[ratio] if isinstance(ratio, str) else ratio
    return round(edge * math.sqrt(w / h) / 32) * 32, round(edge * math.sqrt(h / w) / 32) * 32


def source_size(path):
    """Use display dimensions, matching ComfyUI LoadImage's EXIF orientation."""
    from PIL import Image
    with Image.open(path) as image:
        width, height = image.size
        if image.getexif().get(274, 1) in (5, 6, 7, 8):
            width, height = height, width
    return width, height


def normalize_job(job):
    job = dict(job)
    if not isinstance(job.get("prompt"), str) or not job["prompt"].strip():
        raise ValueError("prompt must be a non-empty string")
    paths = job.get("image_paths", [])
    if not isinstance(paths, (list, tuple)) or len(paths) > MAX_IMAGES:
        raise ValueError("H3-Image supports up to 16 reference images")
    paths = [str(Path(p).expanduser().resolve()) for p in paths]
    if any(not Path(p).is_file() for p in paths):
        raise ValueError("Reference image does not exist")
    if any(not 1 <= int(n) <= len(paths) for n in re.findall(r"<Picture\s+([0-9]+)>", job["prompt"])):
        raise ValueError("Prompt references a picture that was not supplied")
    profile, ratio, quant = job.get("image_profile", "native"), job.get("image_ratio", "1:1"), job.get("image_quant", "w4a8")
    generation_width, generation_height = canvas(profile, ratio)
    edit_size = job.get("image_edit_size", "input")
    if edit_size not in ("input", "canvas"):
        raise ValueError("image_edit_size must be input or canvas")
    width, height = generation_width, generation_height
    if paths and edit_size == "input":
        width, height = source_size(paths[0])
        generation_width, generation_height = canvas(profile, (width, height))
    output_width, output_height = job.get("output_width"), job.get("output_height")
    if output_width is not None or output_height is not None:
        if any(type(n) is not int or not 1 <= n <= 16384 for n in (output_width, output_height)):
            raise ValueError("output_width and output_height must both be integers between 1 and 16384")
        width, height = output_width, output_height
        generation_width, generation_height = canvas(profile, (width, height))
    if not 1 <= width <= 16384 or not 1 <= height <= 16384:
        raise ValueError("Output dimensions must be between 1 and 16384")
    if not 32 <= generation_width <= 16384 or not 32 <= generation_height <= 16384:
        raise ValueError("Generation aspect ratio exceeds the supported canvas range")
    if quant not in QUANTS:
        raise ValueError("image_quant must be int8 or w4a8")
    if job.get("steps", 6) != 6 or job.get("cfg", 1) != 1:
        raise ValueError("H3-Image is distilled for exactly 6 steps and CFG 1")
    seed = job.get("seed")
    seed = secrets.randbelow(2 ** 64) if seed is None else int(seed)
    if not 0 <= seed < 2 ** 64:
        raise ValueError("Invalid seed")
    return dict(job, task="image", image_paths=paths, image_profile=profile, image_ratio=ratio,
                image_quant=quant, image_edit_size=edit_size, width=width, height=height,
                generation_width=generation_width, generation_height=generation_height,
                seed=seed, steps=6, cfg=1)


def build_prompt(job, index=1):
    job = normalize_job(job)
    cond = {"clip": ["3", 0], "vae": ["4", 0], "prompt": job["prompt"],
            "width": job["generation_width"], "height": job["generation_height"]}
    model = QUANTS[job["image_quant"]]
    prompt = {
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "6": {"class_type": "MiniMaxH3ImageConditioning", "inputs": cond},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": job["seed"]}},
        "8": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["6", 0]}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "10": {"class_type": "BasicScheduler", "inputs": {"model": ["2", 0], "scheduler": "simple", "steps": 6, "denoise": 1.0}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["7", 0], "guider": ["8", 0], "sampler": ["9", 0], "sigmas": ["10", 0], "latent_image": ["6", 1]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "15": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": "image/minimax_h3/" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")}},
    }
    if (job["width"], job["height"]) != (job["generation_width"], job["generation_height"]):
        prompt["14"] = {"class_type": "ImageScale", "inputs": {"image": ["12", 0], "upscale_method": "lanczos",
                            "width": job["width"], "height": job["height"], "crop": "disabled"}}
        prompt["15"]["inputs"]["images"] = ["14", 0]
    for i, path in enumerate(job["image_paths"], 1):
        node = str(20 + i)
        prompt[node] = {"class_type": "LoadImage", "inputs": {"image": jobs.stage_image(path)}}
        cond[f"images.image_{i}"] = [node, 0]
    return prompt, dict(job, diffusion_model=model, sampler="euler", scheduler="simple", index=index,
                        prompt_enhancement="off", vae_decode="VAEDecode")


def check_servers():
    if not MANAGED_POOLS:
        candidates = [SERVER]
    else:
        from .worker_pools import servers
        candidates = servers("image")
    available = []
    for server in candidates:
        try:
            jobs.request_json(f"{server}/system_stats", timeout=0.5)
            available.append(server)
        except Exception:
            continue
    if not available:
        raise jobs.NoWorkersAvailable("图像服务未分配 GPU 或正在启动，请用 /gpu 分配图像卡数。")
    return available


def check_support(quant="w4a8", server=None):
    issues = []
    try:
        server = server or (jobs.select_server(check_servers()) if MANAGED_POOLS else SERVER)
        jobs.request_json(f"{server}/system_stats", timeout=3)
        info = jobs.request_json(f"{server}/object_info/MiniMaxH3ImageConditioning", timeout=3)
        if "MiniMaxH3ImageConditioning" not in info:
            issues.append("Image worker 缺少 MiniMaxH3ImageConditioning，需要 ComfyUI 0.45+")
        for category, name in (("diffusion_models", QUANTS[quant]), ("text_encoders", CLIP), ("vae", VAE)):
            if name not in jobs.request_json(f"{server}/models/{category}", timeout=3):
                issues.append(f"Image worker 缺少 {name}")
    except Exception as error:
        issues.append(f"图像服务不可用，请运行 bin/minimax-h3-image-service start：{error}")
    return issues


def submit_job(job, index=1):
    if MANAGED_POOLS:
        from .worker_pools import allocation_lock
        with allocation_lock():
            return _submit_job(job, index)
    return _submit_job(job, index)


def _submit_job(job, index):
    prompt, metadata = build_prompt(job, index)
    server = jobs.select_server(check_servers()) if MANAGED_POOLS else SERVER
    issues = check_support(metadata["image_quant"], server)
    if issues:
        raise RuntimeError("；".join(issues))
    import time
    submitted_at_ms = round(time.time() * 1000)
    response = jobs.request_json(f"{server}/prompt", {"prompt": prompt, "client_id": str(uuid.uuid4()),
                                "extra_data": {"minimax_h3_recipe": metadata}})
    if "prompt_id" not in response:
        raise RuntimeError(f"Image submission rejected: {response}")
    return dict(metadata, server=server, prompt_id=response["prompt_id"], status="queued", submitted_at_ms=submitted_at_ms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--profile", choices=PROFILES, default="native")
    parser.add_argument("--ratio", choices=RATIOS, default="1:1")
    parser.add_argument("--quant", choices=QUANTS, default="w4a8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--edit-size", choices=("input", "canvas"), default="input",
                        help="Editing output follows image 1 by default; canvas uses profile/ratio")
    parser.add_argument("--output-width", type=int)
    parser.add_argument("--output-height", type=int)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    record = submit_job(dict(prompt=args.prompt, image_profile=args.profile, image_ratio=args.ratio,
                             image_quant=args.quant, seed=args.seed, image_paths=args.image,
                             image_edit_size=args.edit_size, output_width=args.output_width, output_height=args.output_height))
    print(f"Queued {record['prompt_id']} {record['width']}x{record['height']}", flush=True)
    record = jobs.wait_for_job(record)
    serialized = json.dumps(record, ensure_ascii=False)
    if args.results:
        args.results.write_text(serialized + "\n", encoding="utf-8")
    print(serialized, flush=True)
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
