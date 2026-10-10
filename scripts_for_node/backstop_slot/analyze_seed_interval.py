#!/usr/bin/env python3
"""Goodput of a policy against the server without AI, with its interval over seeds.

usage: analyze_seed_interval.py OUT.json POLICY LA_CLOSED.json [LA_CLOSED.json ...]
Every file is an output of analyze_la_closed.py for the same condition (other jobs, other seeds).  A run is one
seed of one file; its value is the goodput of POLICY over the goodput of the run without AI of the same seed and
file, minus one.  Printed: mean, standard error and 95% interval (Student t) over the runs, over the seeds (runs of
a seed averaged first), and over the seeds of the last file alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def interval(values) -> dict:
    v = np.asarray(values, dtype=float)
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
    half = float(stats.t.ppf(0.975, len(v) - 1) * se) if len(v) > 1 else float("nan")
    return {"n": int(len(v)), "mean_pct": float(v.mean()), "se_pct": se, "low_pct": float(v.mean() - half),
            "high_pct": float(v.mean() + half), "min_pct": float(v.min()), "max_pct": float(v.max())}


def main() -> None:
    out_path, policy, files = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    runs, by_seed, last = [], {}, []
    for name in files:
        pol = json.loads(Path(name).read_text())["policies"]
        ref = dict(zip(pol["n"]["seeds"], pol["n"]["goodput_by_seed"]))
        for seed, goodput in zip(pol[policy]["seeds"], pol[policy]["goodput_by_seed"]):
            if seed not in ref:
                continue
            value = 100.0 * (goodput / ref[seed] - 1.0)
            runs.append({"file": Path(name).name, "seed": seed, "pct": value})
            by_seed.setdefault(seed, []).append(value)
            if name == files[-1]:
                last.append(value)
    out = {"policy": policy, "runs": runs,
           "over_runs": interval([r["pct"] for r in runs]),
           "over_seeds": interval([np.mean(v) for v in by_seed.values()]),
           "last_file": interval(last)}
    out_path.write_text(json.dumps(out, indent=1))
    print(f"goodput of {policy} against the server without AI\n")
    print("| over | n | mean | standard error | 95% interval | range |")
    print("|---|---|---|---|---|---|")
    for label, key in (("runs", "over_runs"), ("seeds (runs of a seed averaged)", "over_seeds"),
                       (f"seeds of {Path(files[-1]).name}", "last_file")):
        r = out[key]
        print(f"| {label} | {r['n']} | {r['mean_pct']:+.2f}% | {r['se_pct']:.2f} | {r['low_pct']:+.2f} to {r['high_pct']:+.2f}% | "
              f"{r['min_pct']:+.2f} to {r['max_pct']:+.2f}% |")


if __name__ == "__main__":
    main()
