#!/usr/bin/env python3
"""Recovery-loss model against the measured runs.

usage: analyze_loss_model.py OUT.json JOB:TAG:CELLS [JOB:TAG:CELLS ...]
For every run of the given conditions the model gets
  the candidates of the no-AI run of the same seed (which slots, when they became known),
  the run length of the neural receiver alone (median of that no-AI run),
  the run length next to AI and the delay of the conventional result for the policy, both
  taken from one calibration run (the first seed of the first condition), and
  the AI rule of the policy.
It predicts the candidates that find no free neural receiver; the table compares that with the
candidates the run actually lost.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from loss_model import Setting, simulate
from loss_trace import RAW, trace

STOP_LATENCY_MS = 2.0       # an AI piece in flight when the rule stops AI (median piece: 2.3 ms)


def rule_of(policy: str, lanes: int, reserve: int | None) -> str | None:
    if policy == "n":
        return "none"
    if re.fullmatch(r"[sp]\d+", policy):
        return "always"
    if policy == "ve":
        return "never"
    if policy == "vd":
        return "reserve:2"
    if policy == "vf":
        r = 3 if reserve is None else reserve
        return "always" if r == 0 else f"reserve:{r}"
    if policy in ("wn", "wm", "ws", "wc"):  # v14 rules: pieces can be stopped, no AI next to a neural receiver
        return "never"
    if re.fullmatch(r"wr\d", policy):
        return f"reserve:{policy[2]}"
    if re.fullmatch(r"wu\d", policy):
        return f"reuse:{policy[2]}"
    return None


# Stop latency of the v14 rules where the run did not record it (mean over the runs that did).
STOP_LATENCY_V14_MS = {"never": 0.57, "reserve:3": 0.74, "reserve:2": 1.0, "reserve:1": 1.0, "reuse": 1.5}


def stop_latency_of(policy: str, rule: str, result: dict) -> float:
    if rule in ("always", "none"):
        return 0.0
    if not policy.startswith("w"):
        return STOP_LATENCY_MS
    measured = (result.get("controller") or {}).get("stop_latency_ms")
    if measured:
        return float(measured["mean"])
    return STOP_LATENCY_V14_MS.get(rule if not rule.startswith("reuse") else "reuse", 1.0)


def main() -> None:
    out_path = Path(sys.argv[1])
    calibration: dict[str, tuple[float, float]] = {}
    rows = []
    for spec in sys.argv[2:]:
        job, tag, cells = spec.split(":")
        names = sorted(p.stem for p in RAW.glob(f"{tag}[0-9]*_c{cells}_*_j{job}.json"))
        pattern = re.compile(rf"^{re.escape(tag)}(\d)(?:n|r(\d+)([a-z0-9]+))_c{cells}_")
        refs = {}
        for name in names:
            m = pattern.match(name)
            if m and m.group(2) is None:
                refs[int(m.group(1))] = trace(name)
        for name in names:
            m = pattern.match(name)
            if not m or m.group(2) is None or int(m.group(1)) not in refs:
                continue
            seed, rate, policy = int(m.group(1)), int(m.group(2)), m.group(3)
            if rate != 32:
                continue
            ref, run = refs[seed], trace(name)
            lanes = int(ref["lanes"])
            result = json.loads((RAW / f"{name}.json").read_text())
            config = result["config"]
            reserve = (config.get("ai_unit_gating") or {}).get("free_lane_reserve")
            rule = rule_of(policy, lanes, reserve)
            if rule is None:
                continue
            ran_ref, ran = ref["start_ms"] >= 0, run["start_ms"] >= 0
            alone = float(np.median(ref["run_ms"][ran_ref]))
            # Run length next to AI and delay of the conventional result: from the calibration run.
            if rule == "always":
                key = policy if policy != "vf" else "p100"
                if key not in calibration:
                    calibration[key] = (float(np.median(run["run_ms"][ran])) / alone,
                                        float(np.median(run["known_ms"]) - np.median(ref["known_ms"])))
            elif "ours" not in calibration:
                calibration["ours"] = (7.97 / 6.28, float(np.median(run["known_ms"]) - np.median(ref["known_ms"])))
            ratio, shift = calibration[key if rule == "always" else "ours"]
            setting = Setting(lanes=lanes, latest_start_ms=float(ref["rescue_ms"]) - float(ref["bound_ms"]),
                              run_alone_ms=alone, run_ai_ms=alone * ratio, known_shift_ms=shift,
                              stop_latency_ms=stop_latency_of(policy, rule, result))
            base = simulate(ref["period"], ref["known_ms"], "none", Setting(
                lanes=lanes, latest_start_ms=setting.latest_start_ms, run_alone_ms=alone), periods=int(ref["periods"]))
            model = simulate(ref["period"], ref["known_ms"], rule, setting, periods=int(ref["periods"]))
            rows.append({
                "condition": f"{tag} c{cells}", "seed": seed, "policy": policy, "rule": rule, "lanes": lanes,
                "weak_cells": int(ref["weak_cells"]), "candidates": int(len(ref["period"])),
                "lost_no_ai_measured": int((~ran_ref).sum()), "lost_no_ai_model": base["lost"],
                "lost_measured": int((~ran).sum()), "lost_model": model["lost"],
                "stop_latency_ms": setting.stop_latency_ms,
                "run_ms_measured": float(np.median(run["run_ms"][ran])), "run_ms_model": model["run_ms_p50"],
                "ai_time_model": model["ai_time"],
            })
    out_path.write_text(json.dumps({"calibration": calibration, "rows": rows}, indent=1))
    print("| condition | policy | rule | seeds | candidates | lost, no AI: measured / model | lost with AI: measured / model |")
    print("|---|---|---|---|---|---|---|")
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["condition"], r["policy"], r["rule"]), []).append(r)
    for (condition, policy, rule), items in groups.items():
        total = sum(i["candidates"] for i in items)
        f = lambda key: 100.0 * sum(i[key] for i in items) / total
        print(f"| {condition} | {policy} | {rule} | {len(items)} | {total} | {f('lost_no_ai_measured'):.1f}% / "
              f"{f('lost_no_ai_model'):.1f}% | {f('lost_measured'):.1f}% / {f('lost_model'):.1f}% |")
    measured = np.array([100.0 * r["lost_measured"] / r["candidates"] for r in rows])
    model = np.array([100.0 * r["lost_model"] / r["candidates"] for r in rows])
    print(f"\nruns {len(rows)}, correlation {np.corrcoef(measured, model)[0, 1]:.3f}, "
          f"mean absolute error {np.abs(measured - model).mean():.2f} points, mean error {np.mean(model - measured):+.2f} points")


if __name__ == "__main__":
    main()
