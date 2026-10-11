#!/usr/bin/env python3
"""The value of knowing the failures in advance: the clairvoyant controller with errors in its table (v31_info.sh).

usage: analyze_info.py OUT.json JOB[,JOB] [TAG]      (16 cells; runs TAG<seed>n, TAG<seed>r32wd|wo1|wk<variant>)
Per variant, mean over the seeds: AI served within the time limit, recoveries kept against the run without AI of
the same seed, the layer-1 latency at the 99.9th percentile, and the AI served against the rule (wd).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from analyze_sweep import metrics
from loss_trace import RAW
from oracle_noise import VARIANTS

ROWS = [("wd", "Antiphase (knows nothing of the future)", None)] + \
       [("wk" + v, f"table with {'misses' if v[0] == 'r' else 'false alarms' if v[0] == 'f' else 'both'}", v)
        for v in ("f50", "f25", "f10", "f05")] + \
       [("wo1", "exact table", "exact")] + \
       [("wk" + v, "table with misses" if v != "m" else "table with both", v) for v in ("r90", "r75", "r50", "r25", "r00", "m")]


def main() -> None:
    out_path, jobs = Path(sys.argv[1]), sys.argv[2].split(",")
    tag = sys.argv[3] if len(sys.argv) > 3 else "fk"
    runs: dict[str, dict[int, dict]] = {}
    for job in jobs:
        for path in sorted(RAW.glob(f"{tag}[0-9]*_c16_*_j{job}.json")):
            m = re.match(rf"^{tag}(\d+)(n|r32([a-z0-9]+))_c16_", path.name)
            if m:
                runs.setdefault(m.group(3) or "n", {}).setdefault(int(m.group(1)), metrics(path))
    out = {}
    print("| table of the controller | true candidates in the table | false candidates added | AI served (tokens/s) | against Antiphase | recoveries kept | L1 p99.9 (ms) | seeds |")
    print("|---|---|---|---|---|---|---|---|")
    base = None
    for code, label, variant in ROWS:
        if code not in runs:
            continue
        seeds = sorted(s for s in runs[code] if s in runs["n"])
        tokens = float(np.mean([runs[code][s]["ai_slo"] for s in seeds]))
        kept = float(np.mean([100.0 * runs[code][s]["recovered_per_s"] / runs["n"][s]["recovered_per_s"] for s in seeds]))
        tail = float(np.mean([runs[code][s]["l1_p999_ms"] for s in seeds]))
        if code == "wd":
            base = tokens
        keep, alarm = (1.0, 0.0) if variant == "exact" else VARIANTS.get(variant, (None, None))
        out[code] = {"label": label, "keep": keep, "false_alarm": alarm, "tokens": tokens, "kept_pct": kept,
                     "l1_p999_ms": tail, "seeds": seeds}
        show = lambda v: "–" if v is None else f"{100 * v:.0f}%"
        print(f"| {label} | {show(keep)} | {show(alarm)} | {tokens / 1e3:.1f}k | {tokens / base:.2f}x | {kept:.1f}% | {tail:.2f} | {len(seeds)} |")
    out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
