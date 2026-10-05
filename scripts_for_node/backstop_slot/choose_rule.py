#!/usr/bin/env python3
"""The lane reserve chosen by the recovery-loss model, against the measured rules.

usage: choose_rule.py OUT.json JOB:TAG:CELLS [JOB:TAG:CELLS ...]
Input of the model per condition: the candidates of the no-AI runs (which slots, when known),
the run length of the neural receiver alone (median of the same runs), the run length next to
low-priority AI (ratio 7.97 / 6.28), the delay of the conventional result with AI (0.33 ms),
and the stop latency measured for stoppable pieces (STOP_MS).
The model evaluates "never next to a neural receiver" and every reserve R = N-1 .. 1 and picks
the smallest R (most AI time) whose loss stays within TOLERANCE of "never".
The table puts the measured runs of the same condition next to the prediction.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from analyze_sweep import metrics
from loss_model import Setting, simulate
from loss_trace import RAW, trace

TOLERANCE = 0.0015           # share of the candidates (0.15 points)
KNOWN_SHIFT_MS = 0.33
SLOW = 7.97 / 6.28


def stop_ms(lanes: int, reserve: int) -> float:
    """Measured mean time from the stop word to the end of the piece (job 59210955)."""
    if reserve >= lanes:
        return 0.57          # never next to a neural receiver: the unit in flight ran alone
    return 0.74 if reserve == lanes - 1 else 1.0


def main() -> None:
    out_path = Path(sys.argv[1])
    out = {}
    print("| condition | neural receivers | two-user cells | rule | lost candidates: model | measured | AI time: model | AI served: measured | recovered kept: measured |")
    print("|---|---|---|---|---|---|---|---|---|")
    for spec in sys.argv[2:]:
        job, tag, cells = spec.split(":")
        pattern = re.compile(rf"^{re.escape(tag)}(\d)(?:n|r(\d+)([a-z0-9]+))_c{cells}_")
        refs, runs = {}, {}
        for path in sorted(RAW.glob(f"{tag}[0-9]*_c{cells}_*_j{job}.json")):
            m = pattern.match(path.name)
            if not m:
                continue
            if m.group(2) is None:
                refs[int(m.group(1))] = (trace(path.stem), metrics(path))
            else:
                runs.setdefault(m.group(3), {})[int(m.group(1))] = (trace(path.stem), metrics(path))
        if not refs:
            continue
        lanes = int(next(iter(refs.values()))[0]["lanes"])
        weak = int(next(iter(refs.values()))[0]["weak_cells"])
        total = sum(len(t["period"]) for t, _ in refs.values())
        rules = {}
        for reserve in range(lanes, 0, -1):
            rule = "never" if reserve == lanes else f"reserve:{reserve}"
            lost, ai_time = 0, []
            for t, _ in refs.values():
                alone = float(np.median(t["run_ms"][t["start_ms"] >= 0]))
                setting = Setting(lanes=lanes, latest_start_ms=float(t["rescue_ms"]) - float(t["bound_ms"]),
                                  run_alone_ms=alone, run_ai_ms=alone * SLOW, known_shift_ms=KNOWN_SHIFT_MS,
                                  stop_latency_ms=stop_ms(lanes, reserve))
                r = simulate(t["period"], t["known_ms"], rule, setting, periods=int(t["periods"]))
                lost += r["lost"]
                ai_time.append(r["ai_time"])
            rules[reserve] = {"rule": rule, "lost_model": lost / total, "ai_time_model": float(np.mean(ai_time))}
        chosen = lanes
        for reserve in range(lanes - 1, 0, -1):
            if rules[reserve]["lost_model"] <= rules[lanes]["lost_model"] + TOLERANCE:
                chosen = reserve
            else:
                break
        for reserve, row in rules.items():
            code = "wn" if reserve == lanes else f"wr{reserve}"
            by_seed = runs.get(code, {})
            seeds = [s for s in by_seed if s in refs]
            if seeds:
                cands = sum(len(refs[s][0]["period"]) for s in seeds)
                row["lost_measured"] = sum(int((by_seed[s][0]["start_ms"] < 0).sum()) for s in seeds) / cands
                row["ai_tokens_per_s"] = float(np.mean([by_seed[s][1]["ai_slo"] for s in seeds]))
                row["recovered_pct"] = float(np.mean([100 * by_seed[s][1]["recovered"] / max(1, refs[s][1]["recovered"]) for s in seeds]))
            name = "never next to NeuralRx" if reserve == lanes else f"while {reserve} free"
            mark = " **(chosen)**" if reserve == chosen else ""
            meas = f"{100 * row['lost_measured']:.1f}%" if "lost_measured" in row else "–"
            served = f"{row['ai_tokens_per_s'] / 1e3:.1f}k" if "ai_tokens_per_s" in row else "–"
            kept = f"{row['recovered_pct']:.1f}%" if "recovered_pct" in row else "–"
            print(f"| {tag} | {lanes} | {weak} | {name}{mark} | {100 * row['lost_model']:.1f}% | {meas} | "
                  f"{row['ai_time_model']:.2f} | {served} | {kept} |")
        out[f"{tag} c{cells}"] = {"lanes": lanes, "weak_cells": weak, "candidates": total, "chosen_reserve": chosen,
                                  "rules": {str(k): v for k, v in rules.items()}}
    out_path.write_text(json.dumps({"tolerance": TOLERANCE, "conditions": out}, indent=1))


if __name__ == "__main__":
    main()
