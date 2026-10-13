"""Summarize paired render timings; visual quality still needs human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from tools.run_attention_ab import VARIANTS, file_hash, write_json


def summarize(plan_path: Path, result_dir: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    provenance = json.loads((result_dir / "provenance.json").read_text(encoding="utf-8"))
    if provenance["plan_sha256"] != file_hash(plan_path):
        raise ValueError("Results refer to a different plan")
    batches = {}
    for name in VARIANTS:
        rows = {}
        path = result_dir / name / "results.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                index = row["index"]
                if index in rows or not 1 <= index <= plan["job_count_per_variant"]:
                    raise ValueError(f"Unexpected or duplicate result index: {name}/{index}")
                if row["seed"] != plan["pairs"][index - 1]["seed"]:
                    raise ValueError(f"Result seed differs from plan: {name}/{index}")
                rows[index] = row
        batches[name] = rows
    pairs = []
    speedups = []
    for pair in plan["pairs"]:
        index = pair["index"]
        on = batches["sage_on"].get(index)
        off = batches["sage_off"].get(index)
        output = dict(pair, variants={})
        if on and off:
            for field in ("width", "height", "frames", "steps", "sampler", "scheduler", "accelerated"):
                if on.get(field) != off.get(field):
                    raise ValueError(f"Generation settings differ for pair {index}: {field}")
        for name, row in (("sage_on", on), ("sage_off", off)):
            output["variants"][name] = None if row is None else {
                key: row.get(key) for key in
                ("status", "render_seconds", "queue_seconds", "actual_seconds", "observed_device_memory_peak_mib")
            }
        if on and off and on["status"] == off["status"] == "completed":
            on_time, off_time = on.get("render_seconds"), off.get("render_seconds")
            if on_time is not None and off_time is not None and on_time > 0 and off_time > 0:
                output["sage_speedup_ratio"] = off_time / on_time
                speedups.append(output["sage_speedup_ratio"])
        pairs.append(output)
    return {
        "schema_version": 1,
        "expected_jobs_per_variant": plan["job_count_per_variant"],
        "completed_jobs": {name: sum(row["status"] == "completed" for row in rows.values())
                           for name, rows in batches.items()},
        "paired_timing_count": len(speedups),
        "median_paired_sage_speedup": statistics.median(speedups) if speedups else None,
        "pairs": pairs,
        "limitations": [
            "Render timings cover the whole ComfyUI execution, not sampler-only kernel timing.",
            "Memory is the observed device peak sampled at one-second intervals.",
            "Missing, failed or untimed jobs are excluded from speed ratios.",
            "These numbers do not measure visual quality or establish a better default.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.plan, args.results)
    if args.output:
        write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
