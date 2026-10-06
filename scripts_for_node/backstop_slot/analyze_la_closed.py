#!/usr/bin/env python3
"""Closed-loop link adaptation: what each GPU-sharing policy leaves of the cell goodput.

usage: analyze_la_closed.py OUT.json JOB[,JOB] TAG CELLS      (runs of la_cl.sh: TAG<seed>x|n|r32<policy>)

Per run, for the two-user cells (the cells with an outer loop), after the first WARM periods:
  MCS             mean MCS of the slots, and the share of the slots at each level
  first-transmission errors   TBs the conventional receiver did not decode
  retransmissions TBs neither decoded nor recovered before the recovery deadline (what the outer
                  loop steers to its target)
  goodput         new data per uplink slot of a user: sum of TB bits / (TBs + retransmissions / RETX_OK);
                  a retransmission takes a slot of that user and succeeds with probability RETX_OK
  recovered       TBs recovered per second; candidates that got no neural receiver, and candidates whose
                  neural receiver run ended after the recovery deadline
and the AI served within the time limit and the layer-1 misses of the run.
Policies are compared with the run without a recovery path (x) and with the recovery path
without AI (n) of the same seed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from analyze_sweep import metrics
from loss_trace import RAW

WARM = 800            # periods (2 s) left to the outer loop to settle
RETX_OK = 0.95
NAMES = {"x": "No recovery path", "xp100": "No recovery path, low-priority AI", "n": "Recovery path, no AI",
         "wm": "Rule (stoppable pieces, no AI next to NeuralRx)", "wr3": "Rule, AI next to NeuralRx while 3 are free",
         "wr2": "Rule, AI next to NeuralRx while 2 are free", "wr1": "Rule, AI next to NeuralRx while 1 is free",
         "s10": "Fixed 10%", "s30": "Fixed 30%", "p30": "Fixed 30% + low priority", "p50": "Fixed 50% + low priority",
         "p70": "Fixed 70% + low priority", "p100": "Low priority, no cap"}
ORDER = ("x", "xp100", "n", "wm", "wr3", "wr2", "wr1", "s10", "s30", "p30", "p50", "p70", "p100")


def run_stats(path: Path) -> dict:
    result = json.loads(path.read_text())
    config = result["config"]
    work = Path(str(path)[:-5] + "_work")
    la = config["la"]
    rescue_ns = float(config["rescue_deadline_ms"]) * 1e6
    k_max = int(config.get("nrx_max_cb_fail", 1))
    lanes = {}
    for lane in sorted(work.glob("lane*.json")):
        for r in json.loads(lane.read_text())["records"]:
            lanes[(r[0], r[1])] = r
    seconds = (int(config["periods"]) - WARM) * float(config["period_ms"]) / 1e3
    bits = tbs = fails = retx = recovered = candidates = lost = late = 0
    levels = len(la["levels"])                 # with channel states the records carry state * levels + MCS level
    level_slots = np.zeros(levels)
    state_slots: dict[int, list] = {}          # channel state -> [slots, sum of MCS]
    cells = 0
    for cell in config["cells"]:
        if cell.get("nrx_gpu") is None:
            continue
        cells += 1
        data = json.loads((work / f"conv{cell['cell']}.json").read_text())
        size = [8 * b for b in data["la"]["tb_bytes"]]
        ues = int(cell.get("num_ue", 2))
        every = (1 << ues) - 1
        for r in data["records"]:
            period, release, good, level = r[0], r[2], r[9], r[10]
            if period < WARM:
                continue
            level_slots[level % levels] += 1
            entry = state_slots.setdefault(level // levels, [0, 0.0])
            entry[0] += 1
            entry[1] += float(la["mcs"][level % levels])
            ran = lanes.get((cell["cell"], period))
            saved = 0
            if ran is not None and ran[3] - release <= rescue_ns:
                saved = ran[7] & ~good & every
            if good != every and r[7] <= k_max:
                candidates += 1
                lost += ran is None
                late += ran is not None and ran[3] - release > rescue_ns
            final = good | saved
            tbs += ues
            bits += ues * size[level]
            fails += ues - bin(good).count("1")
            recovered += bin(saved).count("1")
            retx += ues - bin(final).count("1")
    m = metrics(path)
    mcs = np.asarray(la["mcs"], dtype=float)
    return {"cells": cells, "tbs": tbs, "mean_mcs": float((level_slots * mcs).sum() / level_slots.sum()),
            "level_share": (level_slots / level_slots.sum()).tolist(),
            "first_error_pct": 100.0 * fails / tbs, "retx_pct": 100.0 * retx / tbs,
            "goodput_bits": bits / (tbs + retx / RETX_OK), "recovered_per_s": recovered / seconds,
            "nrx_demand_pct": 100.0 * candidates / (tbs / 2), "lost_candidates_pct": 100.0 * lost / max(1, candidates),
            "late_candidates_pct": 100.0 * late / max(1, candidates),
            "mcs_by_state": [state_slots[k][1] / state_slots[k][0] for k in sorted(state_slots)],
            "state_share": [state_slots[k][0] / level_slots.sum() for k in sorted(state_slots)],
            "ai_slo": m["ai_slo"], "l1_late_pct": m["l1_late_pct"], "target": float(la["target"])}


def main() -> None:
    out_path, jobs, tag, cells = Path(sys.argv[1]), sys.argv[2].split(","), sys.argv[3], sys.argv[4]
    pattern = re.compile(rf"^{re.escape(tag)}(\d)(x|n|r32([a-z0-9]+))_c{cells}_")
    runs: dict[str, dict[int, dict]] = {}
    for job in jobs:
        for path in sorted(RAW.glob(f"{tag}[0-9]*_c{cells}_*_j{job}.json")):
            m = pattern.match(path.name)
            if m:
                runs.setdefault(m.group(3) or m.group(2), {}).setdefault(int(m.group(1)), run_stats(path))
    if not runs:
        raise SystemExit("no runs")
    some = next(iter(runs.values()))
    target = next(iter(some.values()))["target"]
    print(f"closed-loop link adaptation, target {100 * target:.0f}% of the TBs needing a retransmission; "
          f"two-user cells: {next(iter(some.values()))['cells']}; warm-up {WARM} periods\n")
    print("| policy | seeds | mean MCS | first-transmission errors | TBs needing a retransmission | goodput per user slot | "
          "vs no recovery path | vs recovery path without AI | slots with a candidate | candidates without a NeuralRx | "
          "NeuralRx ended late | recovered TBs/s (vs recovery path without AI) | AI served (tokens/s) | L1 late |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    out = {"target": target, "policies": {}}
    order = [p for p in ORDER if p in runs] + [p for p in runs if p not in ORDER]
    for policy in order:
        by_seed = runs[policy]
        seeds = sorted(by_seed)
        mean = lambda key: float(np.mean([by_seed[s][key] for s in seeds]))
        versus = {}
        for ref in ("x", "n"):
            common = [s for s in seeds if s in runs.get(ref, {})]
            versus[ref] = (100.0 * (np.mean([by_seed[s]["goodput_bits"] / runs[ref][s]["goodput_bits"] for s in common]) - 1.0)
                           if common else float("nan"))
        row = {key: mean(key) for key in ("mean_mcs", "first_error_pct", "retx_pct", "goodput_bits", "nrx_demand_pct",
                                          "lost_candidates_pct", "late_candidates_pct", "recovered_per_s", "ai_slo",
                                          "l1_late_pct")}
        common = [s for s in seeds if s in runs.get("n", {})]
        kept = (100.0 * (np.mean([by_seed[s]["recovered_per_s"] / runs["n"][s]["recovered_per_s"] for s in common]) - 1.0)
                if common else float("nan"))
        row["recovered_vs_no_ai_pct"] = kept
        row.update(seeds=seeds, vs_no_recovery_pct=versus["x"], vs_recovery_no_ai_pct=versus["n"],
                   goodput_by_seed=[by_seed[s]["goodput_bits"] for s in seeds],
                   level_share=np.mean([by_seed[s]["level_share"] for s in seeds], axis=0).tolist(),
                   mcs_by_state=np.mean([by_seed[s]["mcs_by_state"] for s in seeds], axis=0).tolist(),
                   retx_by_seed=[by_seed[s]["retx_pct"] for s in seeds])
        out["policies"][policy] = row
        pct = lambda v: "–" if v != v else f"{v:+.1f}%"
        print(f"| {NAMES.get(policy, policy)} | {len(seeds)} | {row['mean_mcs']:.2f} | {row['first_error_pct']:.1f}% | "
              f"{row['retx_pct']:.1f}% | {row['goodput_bits'] / 1e3:.1f}k | {pct(versus['x'])} | {pct(versus['n'])} | "
              f"{row['nrx_demand_pct']:.1f}% | {row['lost_candidates_pct']:.1f}% | {row['late_candidates_pct']:.1f}% | "
              f"{row['recovered_per_s']:.0f} ({pct(kept)}) | "
              f"{row['ai_slo'] / 1e3:.1f}k | {row['l1_late_pct']:.3f}% |")
    out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
