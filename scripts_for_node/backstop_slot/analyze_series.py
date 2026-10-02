#!/usr/bin/env python3
"""Runs whose radio load changes: what every policy does over time and at each load level.

usage: analyze_series.py JOB[,JOB...] TAG CELLS [WINDOW_PERIODS]
Per run (names as in analyze_sweep.py):
  series      per window: active cells per period, AI tokens finished within the time limit
              (counted when they finish), TBs recovered by the recovery deadline, L1-late TBs
  by_level    per load level (active share of the cells, rounded to the configured levels):
              AI tokens per second (requests counted at the level they arrived in), recovered
              TBs, and recovered TBs as a share of the no-AI run of the same seed
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from activity import activity_mask, step_levels  # noqa: E402
from analyze_sweep import ROOT, policy_name  # noqa: E402


def level_of_period(config: dict) -> np.ndarray:
    spec = config.get("activity") or {}
    periods = int(config["periods"])
    if spec.get("mode") == "steps":
        return step_levels(spec, periods)
    if spec.get("mode") == "phased":
        busy = (np.arange(periods) // int(spec["phase_periods"])) % 2 == 0
        return np.where(busy, float(spec.get("phase_high", 1.0)), float(spec["prob"]))
    return np.full(periods, float(spec.get("prob", 1.0)))


def load(path: Path, window: int) -> dict:
    result = json.loads(path.read_text())
    config = result["config"]
    work = Path(str(path)[:-5] + "_work")
    periods, skip = int(config["periods"]), int(config.get("skip_periods", 20))
    period_ns = float(config["period_ms"]) * 1e6
    rescue_ns = float(config["rescue_deadline_ms"]) * 1e6
    deadline_ns = float(config["deadline_ms"]) * 1e6
    level = level_of_period(config)
    mask = activity_mask(config)
    lanes = {}
    for lane in work.glob("lane*.json"):
        for r in json.loads(lane.read_text())["records"]:
            lanes[(r[0], r[1])] = r
    recovered, late, tbs = np.zeros(periods), np.zeros(periods), np.zeros(periods)
    for cell in config["cells"]:
        index, ues = int(cell["cell"]), int(cell.get("num_ue", 1))
        for r in json.loads((work / f"conv{index}.json").read_text())["records"]:
            period, release, done = r[0], r[2], r[4]
            tbs[period] += ues
            late[period] += ues * int(done - release > deadline_ns)
            if cell.get("nrx_gpu") is None:
                continue
            conv_good = r[9] if ues > 1 else r[6]
            ran = lanes.get((index, period))
            if ran is not None and ran[3] - release <= rescue_ns:
                good = ran[7] if ues > 1 else ran[5]
                recovered[period] += bin(good & ~conv_good & ((1 << ues) - 1)).count("1")
    slo_ns = float(config.get("ai", {}).get("slo_ms", 200.0)) * 1e6
    served_at_finish, served_at_arrival = np.zeros(periods), np.zeros(periods)
    for path_ai in work.glob("ai*.json"):
        data = json.loads(path_ai.read_text())
        epoch = int(data["epoch_ns"])
        for arrival, length, _, finished in data.get("requests", []):
            if finished and finished - (epoch + arrival) <= slo_ns:
                served_at_arrival[min(periods - 1, int(arrival // period_ns))] += length
                served_at_finish[min(periods - 1, max(0, int((finished - epoch) // period_ns)))] += length
    keep = np.arange(periods) >= skip
    edges = list(range(0, periods, window))
    seconds = window * period_ns / 1e9
    series = {
        "t_s": [e * period_ns / 1e9 for e in edges],
        "active_cells": [float(mask[:, e:e + window].sum(axis=0).mean()) for e in edges],
        "ai_tokens_per_s": [float(served_at_finish[e:e + window].sum() / seconds) for e in edges],
        "recovered": [float(recovered[e:e + window].sum()) for e in edges],
        "l1_late": [float(late[e:e + window].sum()) for e in edges],
    }
    by_level = {}
    for value in sorted(set(level.tolist())):
        chosen = keep & (level == value)
        by_level[f"{value:g}"] = {
            "seconds": float(chosen.sum() * period_ns / 1e9),
            "ai_tokens_per_s": float(served_at_arrival[chosen].sum() / max(1e-9, chosen.sum() * period_ns / 1e9)),
            "recovered": float(recovered[chosen].sum()),
            "l1_late_pct": float(100 * late[chosen].sum() / max(1, tbs[chosen].sum())),
        }
    return {"series": series, "by_level": by_level,
            "ai_tokens_per_s": float(served_at_arrival[keep].sum() / (keep.sum() * period_ns / 1e9)),
            "recovered": float(recovered[keep].sum()),
            "l1_late_pct": float(100 * late[keep].sum() / max(1, tbs[keep].sum()))}


def main() -> None:
    jobs, tag, cells = sys.argv[1].split(","), sys.argv[2], sys.argv[3]
    window = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    pattern = re.compile(rf"^{re.escape(tag)}(\d)(?:n|r(\d+)([a-z0-9]+))_c{cells}_")
    runs = {}
    for job in jobs:
        for path in sorted((ROOT / "raw").glob(f"{tag}[0-9]*_c{cells}_*_j{job}.json")):
            m = pattern.match(path.name)
            if m:
                runs.setdefault(m.group(3) or "n", {})[int(m.group(1))] = load(path, window)
    refs = runs.get("n", {})
    levels = sorted({k for by_seed in runs.values() for r in by_seed.values() for k in r["by_level"]}, key=float)
    print("| policy | seeds | whole run: AI tokens/s, recovered kept, L1 late | " +
          " | ".join(f"load {float(v):.0%}: AI tokens/s, recovered kept" for v in levels) + " |")
    print("|---|---|---|" + "---|" * len(levels))
    out = {}
    for policy, by_seed in runs.items():
        seeds = sorted(s for s in by_seed if s in refs)
        if not seeds:
            continue
        row = {"name": "No AI" if policy == "n" else policy_name(policy), "seeds": seeds,
               "ai_tokens_per_s": float(np.mean([by_seed[s]["ai_tokens_per_s"] for s in seeds])),
               "recovered_pct": float(np.mean([100 * by_seed[s]["recovered"] / max(1, refs[s]["recovered"]) for s in seeds])),
               "recovered_pct_by_seed": [100 * by_seed[s]["recovered"] / max(1, refs[s]["recovered"]) for s in seeds],
               "ai_by_seed": [by_seed[s]["ai_tokens_per_s"] for s in seeds],
               "l1_late_pct": float(np.mean([by_seed[s]["l1_late_pct"] for s in seeds])),
               "by_level": {}, "series": {str(s): by_seed[s]["series"] for s in seeds}}
        cellsout = []
        for v in levels:
            ai = float(np.mean([by_seed[s]["by_level"][v]["ai_tokens_per_s"] for s in seeds]))
            kept = float(np.mean([100 * by_seed[s]["by_level"][v]["recovered"] / max(1, refs[s]["by_level"][v]["recovered"])
                                  for s in seeds]))
            l1 = float(np.mean([by_seed[s]["by_level"][v]["l1_late_pct"] for s in seeds]))
            row["by_level"][v] = {"ai_tokens_per_s": ai, "recovered_pct": kept, "l1_late_pct": l1}
            cellsout.append(f"{ai / 1e3:.1f}k, {kept:.1f}%")
        out[policy] = row
        print(f"| {row['name']} | {len(seeds)} | {row['ai_tokens_per_s'] / 1e3:.1f}k, {row['recovered_pct']:.1f}%, "
              f"{row['l1_late_pct']:.3f}% | " + " | ".join(cellsout) + " |")
    (ROOT / f"series_{tag}_c{cells}_j{'_'.join(jobs)}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
