from hashlib import sha256
import os
from pathlib import Path

from modelscope import snapshot_download

from minimax_h3.config import MODEL_DIR


MODEL_ID = "Comfy-Org/MiniMax-H3"
FILES = {
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors": "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
    "vae/minimax_h3_audio_vae_fp32.safetensors": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
    "vae/minimax_h3_video_vae_fp16.safetensors": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


print(f"Downloading I2V files to {MODEL_DIR}", flush=True)
snapshot_download(
    MODEL_ID,
    revision="master",
    local_dir=str(MODEL_DIR),
    allow_file_pattern=list(FILES),
    max_workers=int(os.environ.get("MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS", "4")),
)

print("Download complete; verifying SHA-256 checksums", flush=True)
for relative_path, expected_hash in FILES.items():
    path = MODEL_DIR / relative_path
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"Checksum mismatch: {relative_path}")
    print(f"OK  {relative_path}", flush=True)

print("I2V_MODELS_READY", flush=True)
