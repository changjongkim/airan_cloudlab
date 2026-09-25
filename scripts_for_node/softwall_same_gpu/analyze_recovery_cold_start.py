#!/usr/bin/env python3
"""Audit the first conventional fallback in external same-request runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.controller:
        data = json.loads(path.read_text(encoding="utf-8"))
        fallbacks = [
            item for item in data["records"]
            if item.get("conventional_gpu_ms") is not None
        ]
        if len(fallbacks) < 2:
            raise SystemExit(f"need at least two fallbacks: {path}")
        first, later = fallbacks[0], fallbacks[1:]
        later_gpu = [item["conventional_gpu_ms"] for item in later]
        row = {
            "controller": str(path),
            "job": data["slurm_job_id"],
            "host": data["host"],
            "releases": data["iterations"],
            "conventional_warmup_units": (
                data.get("warmup", 0)
                if "conventional_warmup_correct" in data else 0
            ) + data.get("noisy_conventional_warmup_units", 0),
            "first_fallback_index": first["index"],
            "first_fallback_gpu_ms": first["conventional_gpu_ms"],
            "first_fallback_response_ms": first["response_ms"],
            "later_fallbacks": len(later),
            "later_gpu_median_ms": statistics.median(later_gpu),
            "later_gpu_max_ms": max(later_gpu),
            "first_over_later_median": (
                first["conventional_gpu_ms"] / statistics.median(later_gpu)
            ),
        }
        rows.append(row)
    report = {
        "schema": "softwall-recovery-cold-start-audit-v1",
        "rows": rows,
        "interpretation": (
            "The first timed conventional fallback is much slower in these runs. "
            "This is consistent with an unprimed conventional path, but causal "
            "attribution requires a frozen warmed-versus-unwarmed experiment."
        ),
    }
    lines = [
        "# External recovery cold-start audit",
        "",
        "| job | first fallback release | first conventional GPU ms | "
        "later median/max GPU ms | first response ms |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['job']} | {row['first_fallback_index']} | "
            f"{row['first_fallback_gpu_ms']:.3f} | "
            f"{row['later_gpu_median_ms']:.3f}/{row['later_gpu_max_ms']:.3f} | "
            f"{row['first_fallback_response_ms']:.3f} |"
        )
    lines.extend(["", report["interpretation"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
