"""Prepare paired SageAttention jobs without contacting ComfyUI or Telegram."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from minimax_h3.jobs import image_resolution, load_jobs


VARIANTS = {"sage_on": "1", "sage_off": "0"}
DEFAULT_SEEDS = (42, 314159, 271828)


def prepare_jobs(source: Path, destination: Path, seeds: list[int]) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least one seed, without duplicates")
    if any(not 0 <= seed < 2 ** 64 for seed in seeds):
        raise ValueError("Seeds must be between 0 and 2**64 - 1")
    if destination.exists():
        raise ValueError(f"Output directory already exists: {destination}")

    scenes = load_jobs(source)
    jobs = []
    pairs = []
    inputs = []
    for scene_index, scene in enumerate(scenes, 1):
        width, height = image_resolution(
            scene["image_path"], scene.get("width"), scene.get("height")
        )
        image_hash = hashlib.sha256()
        with Path(scene["image_path"]).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                image_hash.update(block)
        inputs.append({"scene": scene_index, "image_sha256": image_hash.hexdigest(),
                       "width": width, "height": height})
        for seed in seeds:
            job = dict(scene, seed=seed)
            jobs.append(job)
            pairs.append({"index": len(jobs), "scene": scene_index, "seed": seed})
    payload = "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs)
    plan = {
        "schema_version": 1,
        "variable": "MINIMAX_USE_SAGE_ATTENTION",
        "jobs_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "job_count_per_variant": len(jobs),
        "inputs": inputs,
        "pairs": pairs,
        "variants": {
            name: {"environment": {"MINIMAX_USE_SAGE_ATTENTION": value},
                   "jobs": f"{name}/jobs.jsonl"}
            for name, value in VARIANTS.items()
        },
        "notes": [
            "These files do not start workers or submit generation jobs.",
            "Attention is a worker startup setting, not a per-job setting.",
            "Use isolated workers; verify each worker's actual attention backend.",
            "Keep model files, software revisions, hardware and all other settings fixed.",
            "Seeds are shared across variants; deterministic output is not guaranteed.",
        ],
    }
    destination.mkdir(parents=True)
    for name in VARIANTS:
        variant_dir = destination / name
        variant_dir.mkdir()
        (variant_dir / "jobs.jsonl").write_text(payload, encoding="utf-8")
    (destination / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Existing JSONL scenes with local images")
    parser.add_argument("--output-dir", required=True, type=Path, help="New output directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    try:
        plan = prepare_jobs(args.jsonl.resolve(), args.output_dir.resolve(), args.seeds)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Prepared {plan['job_count_per_variant']} jobs per variant in {args.output_dir.resolve()}")
    print("No workers were started and no jobs were submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
