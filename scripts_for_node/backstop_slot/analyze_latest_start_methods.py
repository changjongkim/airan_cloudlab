"""Loss of the measured methods against the latest start L (event model, measured candidates without AI)."""
import sys, json
import numpy as np
sys.path.insert(0, "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/backstop_slot")
from loss_trace import trace, RAW
from loss_model import Setting, simulate

def names(tag, job, n=5):
    return sorted(p.stem for p in RAW.glob(f"{tag}[0-9]n_c16_*_none_j{job}.json"))[:n]

cases = {"full load": names("fa", "59313958"), "8 two-user cells, target 1%": names("lec", "59482496", 3)}
# (rule, run length next to AI, result later by, stop tail): medians measured at full load
methods = {"no AI": ("none", 7.97, 0.0, 0.0), "Antiphase": ("never", 7.97, 0.3, 0.4),
           "Priority (30% cap)": ("always", 7.04, 0.13, 0.0), "Priority-70": ("always", 7.62, 0.29, 0.0),
           "Priority-max": ("always", 7.97, 0.39, 0.0)}
out = {}
for case, runs in cases.items():
    print("\n" + case)
    print("L | " + " | ".join(methods))
    for L in (3.9, 4.4, 4.9):
        row = []
        for label, (rule, s1, shift, tail) in methods.items():
            lost = cand = 0
            for name in runs:
                t = trace(name)
                alone = float(np.median(t["run_ms"][t["start_ms"] >= 0]))
                s = Setting(lanes=int(t["lanes"]), latest_start_ms=L, run_alone_ms=alone, run_ai_ms=s1 + (alone - 6.29),
                            known_shift_ms=shift, stop_latency_ms=tail)
                r = simulate(t["period"], t["known_ms"], rule, s, periods=int(t["periods"]))
                lost += r["lost"]; cand += r["candidates"]
            row.append(f"{100 * lost / cand:.1f}")
            out.setdefault(case, {}).setdefault(label, {})[L] = round(100 * lost / cand, 2)
        print(f"{L} | " + " | ".join(row))
json.dump(out, open("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/latest_start_methods.json", "w"), indent=1)
