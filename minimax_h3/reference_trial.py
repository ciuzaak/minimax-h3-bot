"""Ref2VA eight-step trial only; legacy recipes have no new asset dependency."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import re

LORA_NAME = "minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors"
EXPECTED_HASH = "e56863a7524e0300729cbb7b64cc8b548d90698c1abbf7d83898b0122fbe1e81"
# LightX2V release30f485ed,2026-10-20. Older author table documents768p family6/3,
# not this v1.1 file separately:8/Euler/simple/6,3 remains an explicit experiment.


def mode_for_job(job):
    mode = job.get("reference_mode")
    if job.get("task", "i2v") != "reference":
        if mode is not None:
            raise ValueError("reference_mode requires task=reference")
        return None
    if mode is None:
        return "fast" if job.get("accelerated", False) else "standard"
    if mode not in ("fast", "trial8", "standard"):
        raise ValueError("reference_mode must be fast, trial8 or standard")
    return mode


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=16)
def _hash(path, size, mtime, ctime, inode):
    return file_sha256(path)


def verify_weights(model_dir, comfy_dir):
    found = False
    issues = []
    for path in dict.fromkeys((Path(model_dir) / "loras" / LORA_NAME,
                               Path(comfy_dir) / "models/loras" / LORA_NAME)):
        try:
            if not path.is_file():
                continue
            found = True
            stat = path.stat()
            if _hash(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino) != EXPECTED_HASH:
                issues.append(f"Ref8 SHA-256 mismatch: {path}")
        except OSError as error:
            issues.append(f"Cannot verify Ref8 LoRA {path}: {error}")
    if not found:
        issues.append("Missing Ref8 LoRA; run python -m tools.download_reference_trial")
    return issues


def check_worker(server, request):
    issues = []
    version = request(server + "/system_stats", timeout=3).get("system", {}).get("comfyui_version", "")
    match = re.match(r"v?(\d+)\.(\d+)\.(\d+)", version)
    if not match or tuple(map(int, match.groups())) < (0, 31, 0):
        issues.append(f"{server} Ref8 requires ComfyUI v0.31.0+")
    nodes = ("LoadImage", "UNETLoader", "CLIPLoader", "VAELoader", "MiniMaxH3ReferenceToVideo",
             "RandomNoise", "BasicGuider", "KSamplerSelect", "BasicScheduler", "SamplerCustomAdvanced",
             "VAEDecode", "VAEDecodeAudio", "CreateVideo", "SaveVideo", "LoraLoaderModelOnly", "MiniMaxH3SigmaShift")
    for node in nodes:
        if node not in request(f"{server}/object_info/{node}", timeout=3):
            issues.append(f"{server} missing Ref8 node {node}")
    for category, names in (("text_encoders", ("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",)),
                             ("vae", ("minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors"))):
        available = request(f"{server}/models/{category}", timeout=3)
        issues.extend(f"{server} missing {name}" for name in names if name not in available)
    return issues
