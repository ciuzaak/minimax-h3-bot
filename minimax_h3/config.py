"""Shared, environment-driven configuration for the MiniMax H3 scripts."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_env(path: Path | None = None) -> None:
    """Load simple KEY=VALUE entries without requiring python-dotenv."""
    env_path = path or Path(os.environ.get("MINIMAX_ENV_FILE", PROJECT_DIR / ".env"))
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if not match:
            raise ValueError(f"Invalid .env entry: {raw_line!r}")
        key, value = match.groups()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_env()


def env_path(name: str, default: Path | str) -> Path:
    value = os.environ.get(name, str(default))
    return Path(os.path.expandvars(value)).expanduser().resolve()


def env_executable(name: str, default: str) -> str:
    value = os.path.expandvars(os.environ.get(name, default))
    if os.sep in value:
        return str(Path(value).expanduser().resolve())
    return value


COMFY_DIR = env_path("COMFY_DIR", PROJECT_DIR.parent / "ComfyUI")
MODEL_DIR = env_path("MINIMAX_MODEL_DIR", PROJECT_DIR / "models" / "MiniMax-H3")
OUTPUT_DIR = env_path("MINIMAX_OUTPUT_DIR", PROJECT_DIR / "output")
LOG_DIR = env_path("MINIMAX_LOG_DIR", PROJECT_DIR / "logs")
BOT_INPUT_DIR = env_path("MINIMAX_BOT_INPUT_DIR", PROJECT_DIR / "bot_input")
MODEL_PATHS_CONFIG = env_path(
    "MINIMAX_MODEL_PATHS_CONFIG", PROJECT_DIR / ".runtime" / "extra_model_paths.yaml"
)
COMFY_PYTHON = env_executable("COMFY_PYTHON", sys.executable)
MODELSCOPE_PYTHON = env_executable("MODELSCOPE_PYTHON", sys.executable)
BASE_PORT = int(os.environ.get("MINIMAX_BASE_PORT", "8190"))
MAX_GPU_COUNT = int(os.environ.get("MINIMAX_MAX_GPU_COUNT", "8"))
I2V_TURBO_BACKEND = os.environ.get("MINIMAX_I2V_TURBO_BACKEND", "legacy")
REFERENCE_TURBO_BACKEND = os.environ.get("MINIMAX_REFERENCE_TURBO_BACKEND", "legacy")
VIDEO_VAE = os.environ.get("MINIMAX_VIDEO_VAE", "fp16")
if I2V_TURBO_BACKEND not in ("legacy", "native", "disabled"):
    raise ValueError("MINIMAX_I2V_TURBO_BACKEND must be legacy, native or disabled")
if VIDEO_VAE not in ("fp16", "int8"):
    raise ValueError("MINIMAX_VIDEO_VAE must be fp16 or int8")
if REFERENCE_TURBO_BACKEND not in ("legacy", "native", "disabled"):
    raise ValueError("MINIMAX_REFERENCE_TURBO_BACKEND must be legacy, native or disabled")


def ensure_model_paths_config() -> Path:
    """Create the ComfyUI extra-model-path file from MINIMAX_MODEL_DIR."""
    MODEL_PATHS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "minimax_h3:\n"
        f"  base_path: {json.dumps(str(MODEL_DIR).replace(os.sep, '/'))}\n"
        "  diffusion_models: diffusion_models\n"
        "  text_encoders: text_encoders\n"
        "  vae: vae\n"
        "  loras: loras\n"
    )
    if not MODEL_PATHS_CONFIG.exists() or MODEL_PATHS_CONFIG.read_text(encoding="utf-8") != content:
        MODEL_PATHS_CONFIG.write_text(content, encoding="utf-8")
    return MODEL_PATHS_CONFIG
