import gc
import os
import sys
from minimax_h3.config import COMFY_DIR, ensure_model_paths_config


CONFIG_PATH = ensure_model_paths_config()

os.chdir(COMFY_DIR)
sys.path.insert(0, str(COMFY_DIR))

import torch

import comfy.model_management
import comfy.sd
import folder_paths
import utils.extra_config
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
from nodes import VAELoader


utils.extra_config.load_extra_path_config(str(CONFIG_PATH))

diffusion_path = folder_paths.get_full_path_or_raise(
    "diffusion_models", "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
)
text_encoder_path = folder_paths.get_full_path_or_raise(
    "text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
)

print("Loading MiniMax H3 I2V diffusion model", flush=True)
model = comfy.sd.load_diffusion_model(diffusion_path)
print(f"Diffusion model detected as {type(model.model).__name__}", flush=True)

print("Loading MiniMax H3 text encoder", flush=True)
clip = comfy.sd.load_clip(
    [text_encoder_path],
    embedding_directory=folder_paths.get_folder_paths("embeddings"),
    clip_type=comfy.sd.CLIPType.MINIMAX,
)

print("Loading MiniMax H3 video and audio VAEs", flush=True)
video_vae = VAELoader().load_vae("minimax_h3_video_vae_fp16.safetensors")[0]
audio_vae = VAELoader().load_vae("minimax_h3_audio_vae_fp32.safetensors")[0]
audio_vae.throw_exception_if_invalid()

print("Running a 64x64 first-frame conditioning smoke test", flush=True)
first_frame = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
result = MiniMaxH3ImageToVideo.execute(
    clip=clip,
    vae=video_vae,
    prompt="A static test frame based on <Picture 1>. Audio: silence.",
    width=64,
    height=64,
    length=5,
    first_frame=first_frame,
)
if result is None:
    raise RuntimeError("MiniMaxH3ImageToVideo returned no result")

print("I2V_COMPONENT_SMOKE_TEST_OK", flush=True)
del result, first_frame, audio_vae, video_vae, clip, model
gc.collect()
comfy.model_management.soft_empty_cache()
