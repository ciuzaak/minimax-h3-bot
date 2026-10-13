"""Download and verify the native Ref2VA 4-step LoRA without changing services."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import tempfile

from minimax_h3.config import COMFY_DIR
from minimax_h3.jobs import REFERENCE_TURBO_LORA_NAME


EXPECTED_HASH = "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loras-dir", type=Path, default=COMFY_DIR / "models/loras")
    args = parser.parse_args()
    from modelscope import snapshot_download

    destination = args.loras_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination, prefix=".ref2va-") as temporary:
        snapshot_download("Comfy-Org/MiniMax-H3", revision="master", local_dir=temporary,
                          allow_file_pattern=[f"loras/{REFERENCE_TURBO_LORA_NAME}"],
                          max_workers=int(os.environ.get("MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS", "4")))
        source = Path(temporary) / "loras" / REFERENCE_TURBO_LORA_NAME
        digest = sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != EXPECTED_HASH:
            raise RuntimeError("Checksum mismatch for the Ref2VA Turbo LoRA")
        source.replace(destination / REFERENCE_TURBO_LORA_NAME)
    print(f"REF2VA_TURBO_READY {destination / REFERENCE_TURBO_LORA_NAME}")


if __name__ == "__main__":
    main()
