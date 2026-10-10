"""Loss against the latest start L, from the measured candidates of runs without AI (event model)."""
import sys, json
import numpy as np
sys.path.insert(0, "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/backstop_slot")
from loss_trace import trace, RAW
from loss_model import Setting, simulate

def names(tag, job):
    return sorted(p.stem for p in RAW.glob(f"{tag}[0-9]n_c16_*_none_j{job}.json"))

cases = {"full load, fixed MCS (4 two-user cells)": names("fa", "59313958"),
         "rate control, target 1%, 8 two-user cells": names("lec", "59482496")[:3]}
rules = {"no AI": ("none", {}),
         "AI stopped beside the NRx (tail 0.4 ms, result 0.3 ms later)": ("never", {"stop_latency_ms": 0.4, "known_shift_ms": 0.3}),
         "AI always beside the NRx (low priority, no cap)": ("always", {})}
out = {}
for case, runs in cases.items():
    print("\n" + case, f"({len(runs)} runs)")
    print("L (ms) | " + " | ".join(rules))
    for L in (2.4, 2.9, 3.4, 3.9, 4.4, 4.9, 5.4, 5.9, 6.4):
        cells = []
        for label, (rule, extra) in rules.items():
            lost = cand = 0
            three = []
            for name in runs:
                t = trace(name)
                s = Setting(lanes=int(t["lanes"]), latest_start_ms=L, run_alone_ms=float(np.median(t["run_ms"][t["start_ms"] >= 0])), **extra)
                r = simulate(t["period"], t["known_ms"], rule, s, periods=int(t["periods"]))
                lost += r["lost"]; cand += r["candidates"]; three.append(r["held_3_or_more"])
            cells.append(f"{100 * lost / cand:.1f}% lost")
            out.setdefault(case, {}).setdefault(label, {})[L] = round(100 * lost / cand, 2)
        print(f"{L:.1f} | " + " | ".join(cells))
json.dump(out, open("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/latest_start_sweep.json", "w"), indent=1)
