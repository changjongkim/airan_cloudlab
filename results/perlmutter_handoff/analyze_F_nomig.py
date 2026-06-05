#!/usr/bin/env python3
"""Aggregate F_nomig (priority 1) per-condition L1 frame-time stats."""
import json, glob, os, re
import numpy as np

D = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/F_nomig"

# condition -> list of raw_ms arrays (one per run)
conds = {}
for f in sorted(glob.glob(os.path.join(D, "realL1_*.json"))):
    d = json.load(open(f))
    label = d["label"]
    cond = re.sub(r"_run\d+$", "", label)   # strip _runN
    conds.setdefault(cond, []).append(np.array(d["raw_ms"]))

# Order conditions for display
order = ["F_0_alone","F_E_chanpred_b64","F_E_resnet_b64","F_E_xapp","F_E_qwen_small",
         "F_E_forecaster_d384","F_E_neuralrx","F_E_sat_compute","F_E_sat_hbm",
         "F_B_D2D_256MB_str4","F_C_H2D_256MB_str4","F_D_GEMM_4096",
         "F_F_stack_resnet_x2","F_F_stack_chanpred_x4","F_G_kitchen"]
rest = [c for c in conds if c not in order]
ordered = [c for c in order if c in conds] + sorted(rest)

# baseline
base = np.concatenate(conds["F_0_alone"]) if "F_0_alone" in conds else None
base_p99 = float(np.percentile(base, 99)) if base is not None else None

print(f"{'condition':24s} {'runs':>4s} {'mean':>8s} {'p50':>8s} {'p99':>9s} {'max':>9s} {'miss1ms%':>8s} {'p99/alone':>9s}")
print("-"*92)
rows = []
for c in ordered:
    allms = np.concatenate(conds[c])
    mean, p50, p99 = allms.mean(), np.percentile(allms,50), np.percentile(allms,99)
    mx = allms.max()
    miss = 100.0*np.mean(allms > 1.0)
    ratio = (p99/base_p99) if base_p99 else float('nan')
    rows.append((c,p99))
    print(f"{c:24s} {len(conds[c]):>4d} {mean:8.2f} {p50:8.2f} {p99:9.2f} {mx:9.2f} {miss:8.1f} {ratio:8.1f}x")

print("-"*92)
print(f"baseline (F_0_alone) p99 = {base_p99:.2f} ms")
worst = sorted(rows, key=lambda r:-r[1])[:5]
print("\nWorst-5 by p99:")
for c,p in worst:
    print(f"  {c:24s} p99={p:.1f}ms ({p/base_p99:.0f}x alone)")
