#!/usr/bin/env python3
"""Receiver times of runs: conventional completion per profile and NeuralRx run length.

usage: analyze_times.py JOB TAG [TAG ...]   (reads results/backstop_slot/raw/TAG_*_jJOB_work)
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")


def spread(values) -> str:
    v = np.asarray(values, dtype=float)
    if not v.size:
        return "-"
    return " / ".join(f"{np.percentile(v, q):.2f}" for q in (50, 99, 99.9)) + f" / {v.max():.2f}"


def main() -> None:
    job = sys.argv[1]
    print("| run | profile | conventional done ms (p50 / p99 / p99.9 / max) | NeuralRx runs | NeuralRx run ms (p50 / p99 / p99.9 / max) | AI busy |")
    print("|---|---|---|---|---|---|")
    for tag in sys.argv[2:]:
        for work in sorted(glob.glob(str(ROOT / f"{tag}_*_j{job}_work"))):
            result = json.loads(Path(work[:-5] + ".json").read_text())
            cells = {int(c["cell"]): c["profile"] for c in result["config"]["cells"]}
            skip = int(result["config"].get("skip_periods", 20))
            conv = {}
            for path in glob.glob(work + "/conv*.json"):
                data = json.loads(Path(path).read_text())
                for r in data["records"]:
                    if r[0] >= skip:
                        conv.setdefault(cells[data["cell"]], []).append((r[4] - r[2]) / 1e6)
            busy = []
            for path in glob.glob(work + "/lane*.json"):
                busy += [(r[3] - r[2]) / 1e6 for r in json.loads(Path(path).read_text())["records"] if r[1] >= skip]
            ai = (result["headline"].get("ai_total") or {})
            seconds = int(result["config"]["periods"]) * float(result["config"]["period_ms"]) / 1e3
            share = ai.get("gpu_busy_ms", 0.0) / 1e3 / seconds / result["config"]["num_gpus"]
            name = Path(work).name.split("_j")[0]
            for i, profile in enumerate(sorted(conv)):
                print(f"| {name if not i else ''} | {profile} | {spread(conv[profile])} | "
                      f"{len(busy) if not i else ''} | {spread(busy) if not i else ''} | {100 * share:.0f}% |")


if __name__ == "__main__":
    main()
