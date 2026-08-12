from hashlib import sha256
import os

from modelscope import snapshot_download

from minimax_h3.config import MODEL_DIR


MODEL_ID = "Comfy-Org/MiniMax-H3"
RELATIVE_PATH = "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
EXPECTED_HASH = "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779"


print(f"Downloading Ref2VA file to {MODEL_DIR}", flush=True)
snapshot_download(
    MODEL_ID,
    revision="master",
    local_dir=str(MODEL_DIR),
    allow_file_pattern=[RELATIVE_PATH],
    max_workers=int(os.environ.get("MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS", "4")),
)

print("Download complete; verifying SHA-256 checksum", flush=True)
digest = sha256()
path = MODEL_DIR / RELATIVE_PATH
with path.open("rb") as file:
    for chunk in iter(lambda: file.read(16 * 1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != EXPECTED_HASH:
    raise RuntimeError(f"Checksum mismatch: {RELATIVE_PATH}")

print(f"OK  {RELATIVE_PATH}", flush=True)
print("REF2VA_MODEL_READY", flush=True)
