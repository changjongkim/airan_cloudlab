#!/usr/bin/env python3
"""Step 4 table: partial load, per activity pattern and policy (mean over seeds)."""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_runs import run_row  # noqa: E402

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot")
JOB = sys.argv[1]
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "y"
PATTERN = sys.argv[3] if len(sys.argv) > 3 else r"([bu]\d+)"
LABEL = {"u": "Antiphase"}


def main() -> None:
    refs, groups = {}, {}
    for path in sorted(glob.glob(str(ROOT / "raw" / f"{PREFIX}[0-9]*_j{JOB}.json"))):
        m = re.match(rf"{PREFIX}(\d){PATTERN}(n|m|r(\d+)([a-z]+\d*))_c\d+_", Path(path).name)
        if not m:
            continue
        row = run_row(path)
        seed, act = int(m.group(1)), m.group(2)
        if m.group(3) in ("n", "m"):
            refs.setdefault((act, seed), []).append(row)
        else:
            groups.setdefault((act, int(m.group(4)), m.group(5)), []).append((seed, row))
    out = []
    for (act, seed), rs in sorted(refs.items()):
        print(f"no AI {act} seed {seed}: rescues {[r['rescues'] for r in rs]}, L1 late % {[round(r['l1_late_pct'], 3) for r in rs]}")
    print("| setting | AI load | policy | rescues (vs no AI) min / mean | L1 late % | NRx late | AI tokens in SLO /s | AI GPU busy |")
    print("|---|---|---|---|---|---|---|---|")
    for (act, rate, pol), items in sorted(groups.items()):
        ratios = [r["rescues"] / np.mean([x["rescues"] for x in refs[(act, s)]]) for s, r in items if (act, s) in refs]
        rec = {"setting": act, "ai_rate": rate, "policy": LABEL.get(pol, f"fixed {pol[1:]}% share" if pol.startswith("s") else pol),
               "rescue_min": float(min(ratios)), "rescue_mean": float(np.mean(ratios)),
               "l1_late_pct": float(np.mean([r["l1_late_pct"] for _, r in items])),
               "nrx_late": float(np.mean([r["nrx_late"] for _, r in items])),
               "ai_slo": float(np.mean([r["ai_slo"] for _, r in items])),
               "ai_busy": float(np.mean([r["ai_busy"] for _, r in items])), "seeds": len(items)}
        if any(r.get("primary_tbs") for _, r in items):
            rec["primary_on_time_pct"] = float(np.mean([r["primary_on_time_pct"] for _, r in items]))
        out.append(rec)
        print(f"| {act} | {rate} | {rec['policy']} | {100*rec['rescue_min']:.1f}% / {100*rec['rescue_mean']:.1f}% | "
              f"{rec['l1_late_pct']:.3f} | {rec['nrx_late']:.0f} | {rec['ai_slo']:.0f} | {100*rec['ai_busy']:.0f}% |")
    (ROOT / f"v3_{PREFIX}_j{JOB}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
