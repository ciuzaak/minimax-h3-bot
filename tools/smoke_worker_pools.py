"""Temporarily validate separate GPU pools and restore the test allocation to 0/0."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path(".runtime/pools-live-smoke.json"))
    parser.add_argument("--video-port", type=int, default=8390)
    parser.add_argument("--image-port", type=int, default=8490)
    args = parser.parse_args()
    os.environ.update(MINIMAX_MANAGED_POOLS="1", MINIMAX_SHARED_WORKER="0",
                      MINIMAX_BASE_PORT=str(args.video_port), MINIMAX_IMAGE_PORT=str(args.image_port))
    from PIL import Image
    from minimax_h3 import image_jobs, jobs, worker_pools as pools

    # Use separate state and ports; never change the Bot's requested allocation.
    args.results.parent.mkdir(parents=True, exist_ok=True)
    pools.STATE_PATH = args.results.with_name(f"pools-smoke-state-{time.time_ns()}.json").resolve()
    runtime = pools.Runtime()
    results, records, checks = [], [], {}

    def ready(video, image):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            state = pools.reconcile(runtime)
            print("STATE", [(w["role"], w["gpu"], w["status"]) for w in state["workers"]], flush=True)
            if len(pools.servers("video")) == video and len(pools.servers("image")) == image:
                return state
            time.sleep(3)
        raise TimeoutError(pools.describe(state))

    def pid(worker):
        return runtime.run("tmux", "display-message", "-p", "-t", "=" + worker["session"], "#{pane_pid}").strip()

    try:
        if len(runtime.free_gpus()) < 2:
            raise RuntimeError("This smoke test needs two idle GPUs")
        pools.set_requested(1, 1)
        state = ready(1, 1)
        video = next(w for w in state["workers"] if w["role"] == "video")
        image = next(w for w in state["workers"] if w["role"] == "image")
        before = pid(video)
        checks["separate_cards"] = video["gpu"] != image["gpu"]
        source = str(args.image.resolve())
        records.append(jobs.submit_job(dict(task="i2v", image_path=source,
            prompt="A red ceramic mug and yellow lemon on a wooden table. Slow camera push in, soft daylight, no speech.",
            seconds=5, width=576, height=576, seed=721, accelerated=True), 1))
        records.append(image_jobs.submit_job(dict(
            prompt="Edit <Picture 1>: change the ceramic mug from red to blue. Preserve the lemon, framing and lighting.",
            image_paths=[source], image_profile="native", seed=722)))
        pools.set_requested(image=0)
        state = pools.reconcile(runtime)
        checks["video_pid_unchanged_on_image_zero"] = pid(video) == before
        checks["image_draining"] = next(w for w in state["workers"] if w["role"] == "image")["status"] == "draining"
        checks["no_new_image_jobs_when_zero"] = not pools.servers("image")
        pending = list(records)
        deadline = time.monotonic() + 600
        while pending and time.monotonic() < deadline:
            for record in list(pending):
                result = jobs.poll_job(record)
                if result is not None:
                    if result["status"] != "completed":
                        raise RuntimeError(result)
                    pending.remove(record)
                    results.append(result)
                    print("COMPLETE", record["task"], record.get("render_seconds"), flush=True)
            pools.reconcile(runtime)
            if pid(video) != before:
                raise RuntimeError("Image resize changed the video worker process")
            time.sleep(2)
        if pending:
            raise TimeoutError("Smoke jobs timed out")
        ready(1, 0)
        checks["image_released_video_untouched"] = pid(video) == before
        for result in results:
            if result["task"] == "image":
                with Image.open(result["outputs"][0]) as image_file:
                    if image_file.size != (result["width"], result["height"]):
                        raise RuntimeError("Image output size mismatch")
            else:
                probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", result["outputs"][0]], text=True))
                if not {s["codec_type"] for s in probe["streams"]} >= {"audio", "video"}:
                    raise RuntimeError("Missing audio/video stream")
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", result["outputs"][0], "-f", "null", "-"], capture_output=True, check=True)
                result["duration"] = probe["format"]["duration"]
        pools.set_requested(image=1)
        state = ready(1, 1)
        image = next(w for w in state["workers"] if w["role"] == "image")
        image_pid = pid(image)
        pools.set_requested(video=0)
        ready(0, 1)
        checks["video_zero_preserves_image_pid"] = pid(image) == image_pid
        if not all(checks.values()):
            raise RuntimeError(checks)
        print("CHECKS", json.dumps(checks), flush=True)
    finally:
        for record in records:
            try:
                jobs.request_json(record["server"] + "/queue", {"delete": [record["prompt_id"]]})
                jobs.request_json(record["server"] + "/interrupt", {"prompt_id": record["prompt_id"]})
            except Exception:
                pass
        pools.set_requested(0, 0)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = pools.reconcile(runtime)
            if not state["workers"]:
                break
            time.sleep(2)
        args.results.write_text(json.dumps(dict(checks=checks, results=results, final_state=state,
                                               state_path=str(pools.STATE_PATH)), ensure_ascii=False, indent=2) + "\n")
        if state["workers"]:
            raise RuntimeError("Test workers did not drain; inspect the saved state")
    print("POOLS_LIVE_SMOKE_PASSED; test allocation restored to 0/0", flush=True)


if __name__ == "__main__":
    main()
