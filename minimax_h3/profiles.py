"""Explicit recipes: keep adapter, sampler, and schedule in the same family."""

import math


NATIVE_TURBO_LORA_NAME = "minimax_h3_fl2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors"
NATIVE_REFERENCE_LORA_NAME = "minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors"
# The author's 768p family uses 6/3 and its ComfyUI examples use Euler.
# v1.1 is opt-in: the release's older README table does not document it separately.
NATIVE_VIDEO_SHIFT = 6.0
NATIVE_AUDIO_SHIFT = 3.0
VIDEO_VAES = {
    "fp16": "minimax_h3_video_vae_fp16.safetensors",
    "int8": "minimax_h3_video_vae_int8_convrot.safetensors",
}
RESOLUTION_PROFILES = {"balanced": (576, 576 * 1024), "native": (768, 768 * 1344)}


def resolution_for_size(width, height, profile="balanced"):
    if profile not in RESOLUTION_PROFILES:
        raise ValueError("resolution must be balanced or native")
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    short_edge, max_pixels = RESOLUTION_PROFILES[profile]
    scale = min(short_edge / min(width, height), math.sqrt(max_pixels / (width * height)))
    return max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)
