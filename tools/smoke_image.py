"""Run native H3-Image profiles and a real edit; record sampled GPU memory."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from minimax_h3 import image_jobs, jobs

PROMPT = ("A detailed studio photograph of a single red ceramic mug and one yellow lemon on a wooden table. "
          "The mug is on the left and the lemon is on the right, soft daylight from a window, realistic ceramic glaze "
          "and wood grain, clean neutral background, centered still life, no lettering or logos.")


def run_case(job, gpu, timeout):
    record = image_jobs.submit_job(job)
    deadline = time.monotonic() + timeout
    peak = 0
    while True:
        measurement = subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=10, check=True)
        peak = max(peak, int(measurement.stdout.strip()))
        result = jobs.poll_job(record)
        if result is not None:
            result["sampled_peak_gpu_mib"] = peak
            result["physical_gpu"] = gpu
            result["memory_sample_interval_seconds"] = 1
            if result["status"] == "completed":
                from PIL import Image
                images = result.get("outputs", [])
                if not images:
                    raise RuntimeError("Completed image job has no output")
                for path in images:
                    with Image.open(path) as image:
                        if image.size != (record["width"], record["height"]) or image.format != "PNG":
                            raise RuntimeError(f"Unexpected image output: {path}")
            return result
        if time.monotonic() > deadline:
            # Stop only this test's request, so a timed-out test cannot block later runs.
            jobs.request_json(f"{image_jobs.SERVER}/queue", {"delete": [record["prompt_id"]]})
            jobs.request_json(f"{image_jobs.SERVER}/interrupt", {"prompt_id": record["prompt_id"]})
            return dict(record, status="error", error="smoke test timed out", sampled_peak_gpu_mib=peak)
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU used by the image worker")
    parser.add_argument("--quant", choices=image_jobs.QUANTS, default="w4a8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profiles", nargs="+", choices=image_jobs.PROFILES, default=list(image_jobs.PROFILES))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--results", type=Path, default=Path(".runtime/image-smoke.json"))
    args = parser.parse_args()
    results = []
    source = None
    for profile in args.profiles:
        result = run_case(dict(prompt=PROMPT, seed=args.seed, image_profile=profile, image_quant=args.quant), args.gpu, args.timeout)
        results.append(result)
        if result["status"] == "completed" and source is None:
            source = result["outputs"][0]
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if source:
        result = run_case(dict(prompt="Edit <Picture 1>: change the red ceramic mug to blue. Preserve the lemon, table, composition, camera angle and lighting.",
                               seed=args.seed, image_profile="preview", image_quant=args.quant, image_paths=[source]), args.gpu, args.timeout)
        results.append(result)
        args.results.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if source and all(result["status"] == "completed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
