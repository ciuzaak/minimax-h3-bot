"""Pinned optional assets and stat-keyed verification of shared search paths."""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from .profiles import NATIVE_TURBO_LORA_NAME, NATIVE_REFERENCE_LORA_NAME, VIDEO_VAES


# Comfy-Org e5eb578a89295337b8ff433a035929ce0279e0b6; LightX2V 30f485ed874aa44c8183348ad361c5eabad318da.
ASSETS = {
    "native-turbo": (f"loras/{NATIVE_TURBO_LORA_NAME}", "fb4c00361c94bad47f878d5e096c6cab2f45063b0968bd5756dcb7da6936f53e"),
    "native-reference-turbo": (f"loras/{NATIVE_REFERENCE_LORA_NAME}", "e56863a7524e0300729cbb7b64cc8b548d90698c1abbf7d83898b0122fbe1e81"),
    "int8-vae": (f"vae/{VIDEO_VAES['int8']}", "52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e"),
}
LIGHTX_ASSETS = {"native-turbo", "native-reference-turbo"}


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=64)
def _cached_hash(path, size, mtime_ns, ctime_ns, inode):
    return file_sha256(path)


def verify_asset(name, model_dir, comfy_dir):
    relative, expected = ASSETS[name]
    issues = []
    found = False
    for path in dict.fromkeys((Path(model_dir) / relative, Path(comfy_dir) / "models" / relative)):
        try:
            if not path.is_file():
                continue
            found = True
            stat = path.stat()
            actual = _cached_hash(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
            if actual != expected:
                issues.append(f"Checksum mismatch: {path}")
        except OSError as error:
            issues.append(f"Cannot verify {path}: {error}")
    if not found:
        issues.append(f"Missing verified {relative}; run python -m tools.download_upgrades {name}")
    return issues
