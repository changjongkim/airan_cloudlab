#!/usr/bin/env python3
"""Lost recoveries of conditions that were never run, computed before the run (closed form, loss_model.lane_chain).

usage: predict_new_conditions.py predict OUT.json          new conditions of v23_review.sh (steps pre_a, pre_b)
       predict_new_conditions.py retro OUT.json            the same procedure on conditions that were measured before
       predict_new_conditions.py compare OUT.json PRED.json JOB[,JOB]      the runs of the new conditions against PRED

Nothing of a predicted condition is measured.  The inputs of the closed form are
  N   the number of neural receivers (GPUs) of the condition;
  T   the transition matrix of the candidates per slot.  Which slots of a two-user cell are candidates is fixed
      by the data set, the cell index and the seed (the same in every job, condition and policy), so the
      candidates of a new set of two-user cells are put together from the runs without AI of earlier conditions
      that had those cells (SOURCES);
  c,S the time at which a candidate becomes known and the run length of the neural receiver under the policy,
      from the runs of that policy in ONE calibration condition: 16 cells, four GPUs, four two-user cells, full
      load (tag fa, seeds 1-5).  For the two-GPU condition a second value uses c, S of the two-GPU runs (tag sd).
The latest start time is 3.9 ms as in every run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from loss_model import arrival_transitions, lane_chain
from loss_trace import RAW, trace

LATEST_START_MS = 3.9
# runs without AI that give the candidates of a cell: (tag, cells, job, seeds)
SOURCES = (("fa", 16, "59313958", (1, 2, 3, 4, 5)), ("sw", 16, "59313958", (1, 2)), ("sd", 8, "59313958", (1, 2)),
           ("sy", 16, "59318975", (1, 2)))
# calibration runs for c and S: policy of the new runs -> (tag, cells, policy code there, {job: seeds})
CAL = {"main": {"n": ("fa", 16, "n", {"59313958": (1, 2, 3, 4, 5)}),
                "wd": ("fa", 16, "wm", {"59313958": (1, 2, 3, 4, 5)}),
                "p30": ("fa", 16, "p30", {"59210955": (1, 2), "59313958": (3, 4, 5)}),
                "p70": ("fa", 16, "p70", {"59210955": (1, 2), "59313958": (3, 4, 5)})},
       "two_gpus": {"n": ("sd", 8, "n", {"59313958": (1, 2)}), "wd": ("sd", 8, "wm", {"59318975": (1, 2)}),
                    "p30": ("sd", 8, "p30", {"59313958": (1, 2)}), "p70": ("sd", 8, "p70", {"59313958": (1, 2)})}}
# new conditions: tag -> (cells, GPUs, weak fraction, seeds)
NEW = {"oa": (12, 3, 0.1667, (1, 2, 3)), "ob": (12, 3, 0.25, (1, 2, 3)), "oc": (16, 4, 0.1875, (1, 2, 3)),
       "od": (16, 4, 0.3125, (1, 2)), "oe": (16, 4, 0.4375, (1, 2)), "of": (8, 2, 0.375, (1, 2))}
# measured conditions for the same procedure (retro): tag -> (cells, GPUs, weak fraction, seeds, {policy: (code, job)})
OLD = {"sx": (16, 4, 0.125, (1, 2), {"n": ("n", "59318975"), "wd": ("wm", "59318975"), "p30": ("p30", "59318975"), "p70": ("p70", "59318975")}),
       "sy": (16, 4, 0.375, (1, 2), {"n": ("n", "59318975"), "wd": ("wm", "59318975"), "p30": ("p30", "59318975"), "p70": ("p70", "59318975")}),
       "sw": (16, 4, 0.5, (1, 2), {"n": ("n", "59313958"), "wd": ("wm", "59318975"), "p30": ("p30", "59313958"), "p70": ("p70", "59313958")}),
       "sc": (8, 2, 0.25, (1, 2), {"n": ("n", "59313958"), "wd": ("wm", "59318975"), "p30": ("p30", "59313958"), "p70": ("p70", "59313958")}),
       "sd": (8, 2, 0.5, (1, 2), {"n": ("n", "59313958"), "wd": ("wm", "59318975"), "p30": ("p30", "59313958"), "p70": ("p70", "59313958")})}
POLICIES = ("n", "wd", "p30", "p70")
NAMES = {"n": "no AI", "wd": "Antiphase", "p30": "low priority + 30%", "p70": "low priority + 70%"}


def weak_cells(cells: int, gpus: int, fraction: float) -> list[int]:
    """The two-user cells of a configuration (the choice of make_config.py)."""
    total = round(cells * fraction)
    order = sorted(range(cells), key=lambda c: (c // gpus, c % gpus))
    chosen: set[int] = set()
    for cell in order:
        if len(chosen) < total and (cell // gpus) % 2 == 0:
            chosen.add(cell)
    for cell in order:
        if len(chosen) < total and cell not in chosen:
            chosen.add(cell)
    return sorted(chosen)


def run_name(tag: str, seed: int, code: str, cells: int, job: str) -> str:
    part = "n" if code == "n" else f"r32{code}"
    matches = sorted(RAW.glob(f"{tag}{seed}{part}_c{cells}_*_j{job}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{tag}{seed}{part}_c{cells} j{job}: {len(matches)} files")
    return matches[0].stem


_candidates: dict[tuple[int, int], tuple[np.ndarray, int, str]] = {}


def candidate_periods(cell: int, seed: int, skip_source: str | None = None) -> tuple[np.ndarray, int, str]:
    """Periods in which two-user cell ``cell`` has a candidate with seed ``seed`` (from a run without AI)."""
    key = (cell, seed)
    if key in _candidates and _candidates[key][2] != skip_source:
        return _candidates[key]
    for tag, cells, job, seeds in SOURCES:
        if seed not in seeds or tag == skip_source:
            continue
        name = run_name(tag, seed, "n", cells, job)
        config = json.loads((RAW / f"{name}.json").read_text())["config"]
        ids = [c["cell"] for c in config["cells"] if c.get("nrx_gpu") is not None]
        if cell not in ids:
            continue
        t = trace(name)
        found = (np.asarray(t["period"])[np.asarray(t["cell"]) == ids.index(cell)], int(t["periods"]), tag)
        if skip_source is None:
            _candidates[key] = found
        return found
    raise LookupError(f"no run without AI has two-user cell {cell} with seed {seed}")


def transitions(cells: int, gpus: int, fraction: float, seeds, skip_source: str | None = None):
    ids = weak_cells(cells, gpus, fraction)
    tables, per_slot, used = [], [], set()
    for seed in seeds:
        parts = [candidate_periods(cell, seed, skip_source) for cell in ids]
        periods = min(p[1] for p in parts)
        used.update(p[2] for p in parts)
        period = np.concatenate([p[0][p[0] < periods] for p in parts])
        tables.append(arrival_transitions(period, periods, len(ids)))
        per_slot.append(len(period) / periods)
    return np.mean(tables, axis=0), ids, float(np.mean(per_slot)), sorted(used)


def timing(policy: str, source: str):
    tag, cells, code, jobs = CAL[source][policy]
    ts = [trace(run_name(tag, seed, code, cells, job)) for job, seeds in jobs.items() for seed in seeds]
    known = np.concatenate([t["known_ms"] for t in ts])
    length = np.concatenate([t["run_ms"][t["start_ms"] >= 0] for t in ts])
    return known, length


def predict(conditions: dict, retro: bool) -> list[dict]:
    rows = []
    for tag, spec in conditions.items():
        cells, gpus, fraction, seeds = spec[:4]
        table, ids, per_slot, used = transitions(cells, gpus, fraction, seeds)      # the candidates of a cell are the same in every run
        for policy in POLICIES:
            row = {"condition": tag, "cells": cells, "gpus": gpus, "two_user_cells": ids, "seeds": list(seeds),
                   "policy": policy, "candidates_per_slot": per_slot, "candidates_from": used}
            for source in ("main", "two_gpus") if gpus == 2 else ("main",):
                known, length = timing(policy, source)
                model = lane_chain(gpus, table, known, length, latest_start_ms=LATEST_START_MS)
                suffix = "" if source == "main" else "_two_gpu_timing"
                row[f"lost_pct{suffix}"] = 100.0 * model["lost_share"]
                row[f"three_slot_pct{suffix}"] = 100.0 * model["three_slot_share"]
            rows.append(row)
    return rows


def measured(tag: str, cells: int, seeds, code: str, job: str | list[str]) -> dict | None:
    jobs = [job] if isinstance(job, str) else job
    lost, three, n = [], [], 0
    for seed in seeds:
        for j in jobs:
            try:
                t = trace(run_name(tag, seed, code, cells, j))
            except FileNotFoundError:
                continue
            ran = t["start_ms"] >= 0
            lost.append(100.0 * float((~ran).mean()))
            three.append(100.0 * float((t["start_ms"][ran] + t["run_ms"][ran] > 5.0 + LATEST_START_MS).mean()))
            n += 1
            break
    return {"lost_pct": float(np.mean(lost)), "three_slot_pct": float(np.mean(three)), "runs": n,
            "lost_by_seed": lost} if n else None


def table(rows: list[dict], with_measured: bool) -> None:
    head = "| condition | N | two-user cells | candidates per slot | policy | lost: predicted |"
    head += " measured | error | three-slot runs: predicted | measured |" if with_measured else " three-slot runs: predicted |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for r in rows:
        line = (f"| {r['condition']} ({r['cells']} cells) | {r['gpus']} | {len(r['two_user_cells'])} | {r['candidates_per_slot']:.2f} | "
                f"{NAMES[r['policy']]} | {r['lost_pct']:.1f}%")
        if "lost_pct_two_gpu_timing" in r:
            line += f" ({r['lost_pct_two_gpu_timing']:.1f}% with the two-GPU timing)"
        if with_measured:
            m = r.get("measured")
            line += (f" | {m['lost_pct']:.1f}% ({m['runs']} runs) | {r['lost_pct'] - m['lost_pct']:+.1f} | {r['three_slot_pct']:.0f}% | "
                     f"{m['three_slot_pct']:.0f}% |") if m else f" | - | - | {r['three_slot_pct']:.0f}% | - |"
        else:
            line += f" | {r['three_slot_pct']:.0f}% |"
        print(line)
    if with_measured:
        done = [r for r in rows if r.get("measured")]
        if done:
            a = np.array([r["measured"]["lost_pct"] for r in done])
            b = np.array([r["lost_pct"] for r in done])
            print(f"\n{len(done)} condition-policy pairs: mean absolute error {np.abs(a - b).mean():.2f} points, mean error "
                  f"{np.mean(b - a):+.2f} points, largest {np.abs(a - b).max():.2f} points, correlation {np.corrcoef(a, b)[0, 1]:.3f}")
            ai = [r for r in done if r["policy"] != "n"]
            a = np.array([r["measured"]["lost_pct"] for r in ai])
            b = np.array([r["lost_pct"] for r in ai])
            print(f"policies with AI ({len(ai)}): mean absolute error {np.abs(a - b).mean():.2f} points, mean error {np.mean(b - a):+.2f} points")


def main() -> None:
    mode, out = sys.argv[1], Path(sys.argv[2])
    if mode == "predict":
        rows = predict(NEW, retro=False)
        out.write_text(json.dumps({"latest_start_ms": LATEST_START_MS, "rows": rows}, indent=1))
        table(rows, with_measured=False)
    elif mode == "retro":
        rows = predict(OLD, retro=True)
        for r in rows:
            cells, _, _, seeds, runs = OLD[r["condition"]]
            code, job = runs[r["policy"]]
            r["measured"] = measured(r["condition"], cells, seeds, code, job)
        out.write_text(json.dumps({"latest_start_ms": LATEST_START_MS, "rows": rows}, indent=1))
        table(rows, with_measured=True)
    else:
        rows = json.loads(Path(sys.argv[3]).read_text())["rows"]
        jobs = sys.argv[4].split(",")
        for r in rows:
            r["measured"] = measured(r["condition"], r["cells"], r["seeds"], r["policy"], jobs)
        out.write_text(json.dumps({"latest_start_ms": LATEST_START_MS, "jobs": jobs, "rows": rows}, indent=1))
        table(rows, with_measured=True)


if __name__ == "__main__":
    main()
