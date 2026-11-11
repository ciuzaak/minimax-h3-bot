"""Download H3-Image components from ModelScope; reuse the video text encoder."""

import argparse
from hashlib import sha256
import os

from minimax_h3.config import MODEL_DIR

MODEL_ID = "Comfy-Org/MiniMax-H3-Image"
REVISION = "b8ebf14be96f02e5c58960620a101de71b0bd9f3"
FILES = {
    "int8": ("diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors", "83eb140f1ff95c068f8a6886e6d3f3d0d4ab2e07bb2fe6d2ef992a51146ee622"),
    "w4a8": ("diffusion_models/minimax_h3_image_pruned_w4a8.safetensors", "517cc19b85348386f4df7c806dce80d839d0899e7cdaa3e1dbb09de964381dbb"),
    "vae": ("vae/minimax_h3_image_vae_fp16.safetensors", "d80e9c2e5cfc9844869961ff5a58c9a9e52182df6bfb30b26018dd7f348721af"),
    "clip": ("text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6"),
}


def file_hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", choices=("int8", "w4a8", "both"), default="w4a8")
    args = parser.parse_args()
    from modelscope import snapshot_download
    keys = ["int8", "w4a8"] if args.quant == "both" else [args.quant]
    for key in [*keys, "vae", "clip"]:
        relative, expected = FILES[key]
        path = MODEL_DIR / relative
        if path.is_file() and file_hash(path) == expected:
            print(f"REUSE {relative}", flush=True)
            continue
        snapshot_download(MODEL_ID, revision=REVISION, local_dir=str(MODEL_DIR),
                          allow_file_pattern=[relative],
                          max_workers=int(os.environ.get("MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS", "4")))
        if file_hash(path) != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}")
        print(f"OK {relative}", flush=True)
    print("IMAGE_MODELS_READY", flush=True)


if __name__ == "__main__":
    main()
