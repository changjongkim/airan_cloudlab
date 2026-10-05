#!/usr/bin/env python3
"""Goodput against recoveries kept, over every condition and policy: what one percent of lost
recoveries costs in uplink goodput of the two-user cells.

usage: analyze_goodput_slope.py OUT.json CONDITION=MAC_GOODPUT.json:SWEEP.json [...]
Per condition: goodput gain of the recovery path without AI (against no recovery path), and for
every policy the recoveries kept (sweep file) and the goodput gain (mac_goodput.py file).
Fit per condition: gain(policy) = gain(no AI) - slope * (100 - recoveries kept), over the policies
that have every seed of the condition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    out_path, out = Path(sys.argv[1]), {}
    print("| condition | policies | goodput gain of the recovery path, no AI | goodput points per 1% of lost recoveries | largest gap between two policies |")
    print("|---|---|---|---|---|")
    slopes = []
    for item in sys.argv[2:]:
        cond, files = item.split("=", 1)
        mac_file, sweep_file = files.split(":")
        mac = json.loads(Path(mac_file).read_text())[cond]
        sweep = {r["policy"]: r for r in json.loads(Path(sweep_file).read_text())}
        none = mac["none"]["weak"]["goodput_share"]
        gain = lambda v: 100.0 * (v["weak"]["goodput_share"] / none - 1.0)
        most = max(int(v["seeds"]) for p, v in mac.items() if p not in ("n", "none"))      # policies with every seed only
        points = [(100.0 - sweep[p]["recovered_pct"], gain(v), p) for p, v in mac.items()
                  if p not in ("n", "none") and p in sweep and sweep[p].get("recovered_pct") is not None and int(v["seeds"]) == most]
        base = gain(mac["n"])
        xs = [x for x, _, _ in points]
        ys = [base - y for _, y, _ in points]
        slope = sum(x * y for x, y in zip(xs, ys)) / max(1e-12, sum(x * x for x in xs))     # line through the no-AI point
        gains = [y for _, y, _ in points]
        out[cond] = {"gain_no_ai_pct": base, "slope_points_per_pct": slope,
                     "policies": {p: {"lost_pct": x, "gain_pct": y} for x, y, p in points}}
        slopes.append(slope)
        print(f"| {cond} | {len(points)} | +{base:.2f}% | {slope:.3f} | {max(gains) - min(gains):.2f} points |")
    print(f"\nmean over the conditions: {sum(slopes) / len(slopes):.3f} goodput points per 1% of lost recoveries")
    out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
