"""Download only the optional Ref2VA eight-step trial LoRA from ModelScope."""

import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path

from minimax_h3.config import MODEL_DIR
from minimax_h3.reference_trial import LORA_NAME, EXPECTED_HASH, file_sha256


@contextmanager
def direct_connection():
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    saved = {key: os.environ.pop(key) for key in keys if key in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def download_asset(model_dir):
    name = "ref8"
    relative, expected = f"loras/{LORA_NAME}", EXPECTED_HASH
    repository, remote_path = "lightx2v/Minimax-h3-Turbo", LORA_NAME
    destination = Path(model_dir) / relative
    if destination.is_file() and file_sha256(destination) == expected:
        print(f"REUSED {relative}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep SDK range parts across cancellation/retry. Each version has its own cache.
    temporary = Path(model_dir) / ".downloads" / f"{name}-{expected[:16]}"
    temporary.mkdir(parents=True, exist_ok=True)
    with (temporary / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"A download for {name} is already active") from error
        with direct_connection():
            from modelscope import snapshot_download

            snapshot_download(repository, revision="master", local_dir=str(temporary),
                              allow_file_pattern=[remote_path],
                              max_workers=int(os.environ.get("MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS", "4")))
        source = Path(temporary) / remote_path
        if file_sha256(source) != expected:
            source.replace(source.with_name(source.name + ".invalid"))
            raise RuntimeError(f"Checksum mismatch: {relative}")
        source.replace(destination)
    print(f"VERIFIED {relative}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    download_asset(args.model_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
