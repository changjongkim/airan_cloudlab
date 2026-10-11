#!/usr/bin/env python3
"""Candidates that are not recovered in time as the recovery deadline moves, computed before the runs.

usage: predict_deadline_sweep.py predict OUT.json                     before the runs of v32_sawtooth.sh
       predict_deadline_sweep.py compare OUT.json PRED.json JOB[,JOB]   runs of v32_sawtooth.sh against PRED

The model (loss_model.py) says that a neural receiver run which starts at offset s of its slot and takes S holds
its receiver for h = ceil((s + S - L) / P) slots, with L = recovery deadline - time limit (7.6 ms) and P = 2.5 ms.
As the deadline moves, the lost candidates should therefore fall in steps: for each policy the step is where its
runs go from three slots to two, and from two to one.  The closed form of loss_model.lane_chain has no state for
one-slot runs, so this file uses an event simulation with the measured inputs:
  arrivals  the candidates of every slot of the headline condition (16 cells, four GPUs, four two-user cells; the
            runs without AI of seeds 1-5; the candidates of a cell are the same in every run)
  c, S      the time at which a candidate becomes known and the run length of the neural receiver under the
            policy, drawn independently from the runs of that policy at the deadline of 11.5 ms
A candidate takes the receiver that is free first, starts at max(known, free) if that is not after its latest
start, and is late if it ends after the recovery deadline.  Nothing is measured at the predicted deadlines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from loss_trace import trace
from predict_new_conditions import CAL, NAMES, measured_deadline, run_name, timing

PERIOD_MS, TIME_LIMIT_MS, LANES = 2.5, 7.6, 4
SEEDS = (1, 2, 3, 4, 5)
POLICIES = ("n", "wd", "p30", "p70")
# tag -> recovery deadline (ms); fa, fy, fz were measured before this file was written
DEADLINES = {"ja": 11.0, "jb": 11.25, "fa": 11.5, "jc": 11.75, "fy": 12.0, "jd": 12.25, "fz": 12.5, "je": 12.75,
             "jf": 13.0, "jg": 13.25, "jh": 13.5, "ji": 13.75, "jk": 14.0}
MEASURED_BEFORE = ("fa", "fy", "fz")
REPEATS = 20


def arrivals(seed: int) -> list[np.ndarray]:
    """Number of candidates of every slot of the headline condition with this seed."""
    t = trace(run_name("fa", seed, "n", 16, "59313958"))
    return np.bincount(np.asarray(t["period"]), minlength=int(t["periods"]))[:int(t["periods"])]


def simulate(counts: np.ndarray, known: np.ndarray, length: np.ndarray, deadline: float, rng) -> dict:
    latest = deadline - TIME_LIMIT_MS
    free = np.zeros(LANES)
    total = lost = late = three = one = runs = 0
    for k, m in enumerate(counts):
        if not m:
            continue
        base = k * PERIOD_MS
        for c in np.sort(rng.choice(known, int(m))):
            total += 1
            lane = int(np.argmin(free))
            start = max(base + c, free[lane])
            if start > base + latest:
                lost += 1
                continue
            end = start + float(rng.choice(length))
            free[lane] = end
            runs += 1
            late += end > base + deadline
            three += end > base + 2 * PERIOD_MS + latest
            one += end <= base + PERIOD_MS + latest
    return {"lost": lost / total, "late": late / total, "three": three / max(1, runs), "one": one / max(1, runs)}


def predict() -> list[dict]:
    rng = np.random.default_rng(0)
    counts = [arrivals(seed) for seed in SEEDS]
    rows = []
    for tag, deadline in DEADLINES.items():
        for policy in POLICIES:
            known, length = timing(policy, "main")
            sims = [simulate(c, known, length, deadline, rng) for c in counts for _ in range(REPEATS // len(SEEDS) or 1)]
            mean = lambda key: 100.0 * float(np.mean([r[key] for r in sims]))
            rows.append({"condition": tag, "deadline_ms": deadline, "latest_start_ms": deadline - TIME_LIMIT_MS,
                         "policy": policy, "lost_pct": mean("lost"), "late_pct": mean("late"),
                         "not_recovered_pct": mean("lost") + mean("late"), "three_slot_pct": mean("three"),
                         "one_slot_pct": mean("one"), "measured_before": tag in MEASURED_BEFORE})
    return rows


def table(rows: list[dict], with_measured: bool) -> None:
    print("| recovery deadline | latest start | " + " | ".join(NAMES[p] for p in POLICIES) + " |")
    print("|---|---|" + "---|" * len(POLICIES))
    for tag, deadline in DEADLINES.items():
        cells = []
        for policy in POLICIES:
            r = next(x for x in rows if x["condition"] == tag and x["policy"] == policy)
            text = f"{r['not_recovered_pct']:.1f}%"
            m = r.get("measured")
            if with_measured and m:
                text += f" / {m['not_recovered_pct']:.1f}%"
            cells.append(text + f" ({r['three_slot_pct']:.0f}%, {r['one_slot_pct']:.0f}%"
                         + (f"; {m['three_slot_pct']:.0f}%" if with_measured and m else "") + ")")
        print(f"| {deadline:.2f} ms ({tag}) | {deadline - TIME_LIMIT_MS:.2f} ms | " + " | ".join(cells) + " |")
    print("\ncell: candidates not recovered in time, predicted" + (" / measured" if with_measured else "")
          + " (runs that hold three slots, runs that hold one slot" + ("; three-slot runs measured" if with_measured else "") + ")")
    if with_measured:
        for label, pick in (("deadlines measured before the prediction (11.5, 12.0, 12.5 ms)", True), ("new deadlines", False)):
            done = [r for r in rows if r.get("measured") and r["measured_before"] == pick]
            if done:
                a = np.array([r["measured"]["not_recovered_pct"] for r in done])
                b = np.array([r["not_recovered_pct"] for r in done])
                print(f"{label}: {len(done)} values, mean absolute error {np.abs(a - b).mean():.2f} points, mean error "
                      f"{np.mean(b - a):+.2f} points, largest {np.abs(a - b).max():.2f} points")


def main() -> None:
    mode, out = sys.argv[1], Path(sys.argv[2])
    if mode == "predict":
        rows = predict()
        for r in rows:                                  # the three deadlines that were measured before
            if r["measured_before"]:
                code = {"wd": "wm"}.get(r["policy"], r["policy"]) if r["condition"] == "fa" else r["policy"]
                jobs = ["59313958", "59210955"] if r["condition"] == "fa" else ["59632171"]
                r["measured"] = measured_deadline(r["condition"], code, jobs)
        out.write_text(json.dumps({"time_limit_ms": TIME_LIMIT_MS, "rows": rows}, indent=1))
        table(rows, with_measured=True)
    else:
        rows = json.loads(Path(sys.argv[3]).read_text())["rows"]
        jobs = sys.argv[4].split(",")
        for r in rows:
            if not r["measured_before"]:
                r["measured"] = measured_deadline(r["condition"], r["policy"], jobs)
        out.write_text(json.dumps({"time_limit_ms": TIME_LIMIT_MS, "jobs": jobs, "rows": rows}, indent=1))
        table(rows, with_measured=True)


if __name__ == "__main__":
    main()
