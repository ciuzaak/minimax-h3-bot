"""CPU-only preflight for opt-in assets; never load a model or change ComfyUI."""

import ast
import subprocess

from minimax_h3.config import COMFY_DIR, I2V_TURBO_BACKEND, REFERENCE_TURBO_BACKEND, MODEL_DIR, VIDEO_VAE
from minimax_h3.assets import verify_asset


def runtime_version():
    version_path = COMFY_DIR / "comfyui_version.py"
    if version_path.is_file():
        for node in ast.parse(version_path.read_text()).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets):
                value = ast.literal_eval(node.value)
                if isinstance(value, str):
                    try:
                        return tuple(int(p) for p in value.split(".")[:3])
                    except ValueError:
                        pass
    return (0, 0, 0)


def validate_core_runtime():
    # Release check also works for shallow tagged checkouts where Git ancestry is incomplete.
    if runtime_version() >= (0, 31, 0):
        return
    result = subprocess.run(["git", "-C", str(COMFY_DIR), "merge-base", "--is-ancestor", "93cb5ed", "HEAD"],
                            capture_output=True)
    if result.returncode:
        raise RuntimeError("ComfyUI requires the H3 audio fix 93cb5ed or v0.31.0+")


def validate_assets():
    names = []
    if I2V_TURBO_BACKEND == "native":
        names.append("native-turbo")
    if REFERENCE_TURBO_BACKEND == "native":
        names.append("native-reference-turbo")
    if VIDEO_VAE == "int8":
        if runtime_version() < (0, 36, 0):
            raise RuntimeError("INT8 video VAE requires ComfyUI v0.36.0+; use fp16 on older installs")
        names.append("int8-vae")
    for name in names:
        issues = verify_asset(name, MODEL_DIR, COMFY_DIR)
        if issues and name == "int8-vae":
            raise RuntimeError("; ".join(issues))
        for issue in issues:
            print(f"TURBO_UNAVAILABLE {issue}; standard mode remains available; selected Turbo requests will be rejected")
    print(f"PROFILE_READY backend={I2V_TURBO_BACKEND} video_vae={VIDEO_VAE}")


if __name__ == "__main__":
    validate_core_runtime()
    validate_assets()
