import subprocess
import time

from minimax_h3.config import (
    COMFY_PYTHON,
    LOG_DIR,
    MODELSCOPE_PYTHON,
    PROJECT_DIR,
)

I2V_LOG = LOG_DIR / "minimax-h3-i2v.log"


def i2v_ready() -> bool:
    if not I2V_LOG.exists():
        return False
    with I2V_LOG.open("rb") as log:
        log.seek(max(0, log.seek(0, 2) - 128 * 1024))
        return b"I2V_MODELS_READY" in log.read()


print("Waiting for the I2V download and checksum verification", flush=True)
while not i2v_ready():
    time.sleep(60)

print("I2V files are verified; starting the ComfyUI component smoke test", flush=True)
subprocess.run(
    [str(COMFY_PYTHON), "-m", "tools.validate_i2v"],
    check=True,
    cwd=PROJECT_DIR,
)

print("I2V smoke test passed; starting Ref2VA download", flush=True)
subprocess.run(
    [str(MODELSCOPE_PYTHON), "-m", "tools.download_ref2va"],
    check=True,
    cwd=PROJECT_DIR,
)
print("MINIMAX_H3_COMPLETE_COMBINATION_READY", flush=True)
