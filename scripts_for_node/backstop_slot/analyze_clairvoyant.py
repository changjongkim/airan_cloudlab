#!/usr/bin/env python3
"""The rule against a clairvoyant controller that was run (v25_oracle.sh, v27_oracle_more.sh).

usage: analyze_clairvoyant.py OUT.json JOB:TAG:LABEL [JOB:TAG:LABEL ...]      (16 cells; runs TAG<seed>n|r32<code>)
Per condition, mean over the seeds: AI served within the time limit, recoveries kept against the run without AI
of the same seed, and the layer-1 latency at the 99.9th percentile, for
  wd   the rule
  wo1  clairvoyant controller that keeps one neural receiver free beyond the known demand (the reference)
  wo0  clairvoyant controller that keeps none free
and the AI served by the rule as a share of the reference.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from analyze_sweep import metrics
from loss_trace import RAW

CODES = ("wd", "wo1", "wo0", "p100")
NAMES = {"wd": "Antiphase", "wo1": "clairvoyant, one receiver kept free", "wo0": "clairvoyant, none kept free",
         "p100": "low priority, no cap"}


def main() -> None:
    out_path, out = Path(sys.argv[1]), {}
    print("| condition | seeds | policy | AI served (tokens/s) | recoveries kept | L1 p99.9 (ms) | Antiphase / this |")
    print("|---|---|---|---|---|---|---|")
    for spec in sys.argv[2:]:
        job, tag, label = spec.split(":", 2)
        runs: dict[str, dict[int, dict]] = {}
        for path in sorted(RAW.glob(f"{tag}[0-9]*_c16_*_j{job}.json")):
            m = re.match(rf"^{tag}(\d+)(n|r32([a-z0-9]+))_c16_", path.name)
            if m:
                runs.setdefault(m.group(3) or "n", {})[int(m.group(1))] = metrics(path)
        rows = {}
        for code in CODES:
            if code not in runs:
                continue
            seeds = sorted(s for s in runs[code] if s in runs["n"])
            rows[code] = {"seeds": seeds,
                          "tokens": float(np.mean([runs[code][s]["ai_slo"] for s in seeds])),
                          "kept_pct": float(np.mean([100.0 * runs[code][s]["recovered_per_s"] / runs["n"][s]["recovered_per_s"] for s in seeds])),
                          "l1_p999_ms": float(np.mean([runs[code][s]["l1_p999_ms"] for s in seeds]))}
        out[tag] = {"label": label, "job": job, "policies": rows,
                    "l1_p999_ms_no_ai": float(np.mean([m["l1_p999_ms"] for m in runs["n"].values()]))}
        for code, r in rows.items():
            share = "" if code == "wd" else f"{100.0 * rows['wd']['tokens'] / r['tokens']:.0f}%"
            print(f"| {label} | {len(r['seeds'])} | {NAMES[code]} | {r['tokens'] / 1e3:.1f}k | {r['kept_pct']:.1f}% | {r['l1_p999_ms']:.2f} | {share} |")
    out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
