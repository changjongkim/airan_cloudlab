#!/usr/bin/env python3
"""AI served and recoveries kept at every recovery deadline of v32_sawtooth.sh.

usage: analyze_deadline_ai.py JOB      (16 cells; runs <tag><seed>n and <tag><seed>r32wd|p30|p70)
Mean over the seeds; recoveries kept are against the run without AI of the same seed.
"""

from __future__ import annotations

import re
import sys

import numpy as np

from analyze_sweep import metrics
from loss_trace import RAW
from predict_deadline_sweep import DEADLINES, MEASURED_BEFORE

CODES = (("wd", "Antiphase"), ("p30", "low priority + 30%"), ("p70", "low priority + 70%"))


def main() -> None:
    job = sys.argv[1]
    print("| recovery deadline | " + " | ".join(name for _, name in CODES) + " | Antiphase / 30% |")
    print("|---|" + "---|" * (len(CODES) + 1))
    for tag, deadline in DEADLINES.items():
        if tag in MEASURED_BEFORE:
            continue
        runs: dict[str, dict[int, dict]] = {}
        for path in sorted(RAW.glob(f"{tag}[0-9]*_c16_*_j{job}.json")):
            m = re.match(rf"^{tag}(\d+)(n|r32([a-z0-9]+))_c16_", path.name)
            if m:
                runs.setdefault(m.group(3) or "n", {})[int(m.group(1))] = metrics(path)
        tokens, cells = {}, []
        for code, _ in CODES:
            seeds = sorted(s for s in runs[code] if s in runs["n"])
            tokens[code] = float(np.mean([runs[code][s]["ai_slo"] for s in seeds]))
            kept = float(np.mean([100.0 * runs[code][s]["recovered_per_s"] / runs["n"][s]["recovered_per_s"] for s in seeds]))
            cells.append(f"{tokens[code] / 1e3:.1f}k, {kept:.1f}%")
        print(f"| {deadline:.2f} ms ({tag}) | " + " | ".join(cells) + f" | {tokens['wd'] / tokens['p30']:.2f}x |")
    print("\ncell: AI served within the time limit (tokens/s), recoveries kept")


if __name__ == "__main__":
    main()
