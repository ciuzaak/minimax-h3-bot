import gc
import os
import sys
from minimax_h3.config import COMFY_DIR, OUTPUT_DIR, ensure_model_paths_config


OUTPUT_PATH = OUTPUT_DIR / "MiniMax_H3_I2V_smoke_test.mp4"
WIDTH = int(os.environ.get("MINIMAX_WIDTH", "608"))
HEIGHT = int(os.environ.get("MINIMAX_HEIGHT", "352"))
LENGTH = int(os.environ.get("MINIMAX_LENGTH", "22"))
STEPS = int(os.environ.get("MINIMAX_STEPS", "6"))
SEED = int(os.environ.get("MINIMAX_SEED", "20260805"))

os.chdir(COMFY_DIR)
sys.path.insert(0, str(COMFY_DIR))

import torch

import comfy.options

comfy.options.enable_args_parsing()

import comfy_aimdo.control
from comfy.cli_args import args, enables_dynamic_vram

if enables_dynamic_vram():
    simple_vram_headroom = None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3)
    comfy_aimdo.control.init(simple_vram_headroom=simple_vram_headroom)

import comfy.model_management
import folder_paths
import utils.extra_config
from comfy_api.latest._util.video_types import VideoContainer
from comfy_extras.nodes_audio import VAEDecodeAudio
from comfy_extras.nodes_custom_sampler import (
    BasicGuider,
    BasicScheduler,
    KSamplerSelect,
    RandomNoise,
    SamplerCustomAdvanced,
)
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
from comfy_extras.nodes_video import CreateVideo
from nodes import CLIPLoader, UNETLoader, VAEDecode, VAELoader


if enables_dynamic_vram():
    comfy_aimdo.control.init_devices(
        (device.index, int(args.vram_headroom * 1024 ** 3))
        for device in comfy.model_management.get_all_torch_devices()
    )

print(
    f"VRAM mode={comfy.model_management.vram_state.name}, "
    f"reserve={comfy.model_management.extra_reserved_memory() / 1024 ** 3:.1f} GiB, "
    f"dynamic={enables_dynamic_vram()}",
    flush=True,
)


utils.extra_config.load_extra_path_config(str(ensure_model_paths_config()))
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading diffusion model", flush=True)
model = UNETLoader().load_unet(
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "default"
)[0]
print("Loading text encoder", flush=True)
clip = CLIPLoader().load_clip(
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "minimax", "default"
)[0]
print("Loading video and audio VAEs", flush=True)
video_vae = VAELoader().load_vae("minimax_h3_video_vae_fp16.safetensors")[0]
audio_vae = VAELoader().load_vae("minimax_h3_audio_vae_fp32.safetensors")[0]

# A deterministic synthetic first frame: blue-to-orange horizontal gradient.
x = torch.linspace(0.0, 1.0, WIDTH).view(1, 1, WIDTH, 1)
first_frame = torch.cat((x, 0.25 + 0.25 * (1.0 - x), 1.0 - x), dim=-1)
first_frame = first_frame.expand(1, HEIGHT, WIDTH, 3).contiguous()

print(f"Encoding first frame at {WIDTH}x{HEIGHT}, {LENGTH} frames", flush=True)
positive, latent = MiniMaxH3ImageToVideo.execute(
    clip=clip,
    vae=video_vae,
    prompt=(
        "The abstract blue and orange gradient from <Picture 1> slowly flows "
        "from left to right like soft silk, with a gentle cinematic shimmer. "
        "Audio: a quiet airy whoosh. No text, no logo."
    ),
    width=WIDTH,
    height=HEIGHT,
    length=LENGTH,
    first_frame=first_frame,
)

noise = RandomNoise.execute(SEED)[0]
guider = BasicGuider.execute(model, positive)[0]
sampler = KSamplerSelect.execute("res_multistep")[0]
sigmas = BasicScheduler.execute(model, "simple", STEPS, 1.0)[0]

print(f"Sampling with res_multistep for {STEPS} steps", flush=True)
samples = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[0]

del latent, positive, sigmas, sampler, guider, noise, clip, model
comfy.model_management.unload_all_models()
gc.collect()
comfy.model_management.soft_empty_cache()

print("Decoding video", flush=True)
images = VAEDecode().decode(video_vae, samples)[0]
comfy.model_management.unload_all_models()
gc.collect()
comfy.model_management.soft_empty_cache()

print("Decoding audio", flush=True)
audio = VAEDecodeAudio.execute(audio_vae, samples)[0]
images = images.detach()
audio = {**audio, "waveform": audio["waveform"].detach()}

video = CreateVideo.execute(images=images, fps=24.0, audio=audio, bit_depth=8)[0]
video.save_to(str(OUTPUT_PATH), format=VideoContainer.MP4, codec="h264", crf=28)

if not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0:
    raise RuntimeError("Smoke-test video was not created")
print(f"I2V_FULL_SMOKE_TEST_OK {OUTPUT_PATH} {OUTPUT_PATH.stat().st_size} bytes", flush=True)

del video, audio, images, samples
del first_frame, audio_vae, video_vae
gc.collect()
comfy.model_management.soft_empty_cache()
