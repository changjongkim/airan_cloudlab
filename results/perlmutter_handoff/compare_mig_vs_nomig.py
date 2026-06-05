#!/usr/bin/env python3
"""Compare no-MIG (priority 1) vs MIG (CloudLab F_saturation, cross-partition) L1 p99."""
import json, glob, os
import numpy as np

NOMIG = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/F_nomig"
MIG   = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/20260601/F_saturation"

# no-MIG label -> MIG label  (closest matching condition)
MAP = [
    ("F_0_alone",              "F_0_alone",              "baseline"),
    ("F_E_chanpred_b64",       "F_E_chanpred_b64",       "chanpred (safe AI)"),
    ("F_E_resnet_b64",         "F_E_resnet_b64",         "ResNet50"),
    ("F_E_forecaster_d384",    "F_E_forecaster_d512",    "forecaster (d384 vs d512)"),
    ("F_D_GEMM_4096",          "F_D_GEMM_d4096",         "GEMM 4096"),
    ("F_B_D2D_256MB_str4",     "F_B_D2D_sz256_str4",     "D2D memcpy 256"),
    ("F_C_H2D_256MB_str4",     "F_C_H2D_sz256_str4",     "H2D memcpy 256"),
    ("F_F_stack_resnet_x2",    "F_F_stack_resnet_x2",    "ResNet x2"),
    ("F_F_stack_chanpred_x4",  "F_F_stack_chanpred_x4",  "chanpred x4"),
    ("F_G_kitchen",            "F_G_kitchen_all",        "kitchen sink"),
]

def agg(d, cond):
    ms = []
    for f in glob.glob(os.path.join(d, f"realL1_{cond}_run*.json")):
        ms += json.load(open(f))["raw_ms"]
    if not ms:
        return None
    a = np.array(ms)
    return np.percentile(a, 50), np.percentile(a, 99), len(a)

print(f"{'condition':22s} | {'MIG p50/p99':>15s} | {'no-MIG p50/p99':>17s} | {'no-MIG/MIG p99':>14s}")
print("-"*82)
for nm, mg, desc in MAP:
    m = agg(MIG, mg); n = agg(NOMIG, nm)
    if not m or not n:
        print(f"{desc:22s} | missing (MIG={bool(m)} noMIG={bool(n)})"); continue
    ratio = n[1]/m[1]
    print(f"{desc:22s} | {m[0]:6.1f}/{m[1]:6.1f}ms | {n[0]:7.1f}/{n[1]:7.1f}ms | {ratio:11.1f}x")
print("-"*82)
print("MIG = CloudLab cross-partition (L1 on 3g, AI on separate 2g — isolated)")
print("no-MIG = Perlmutter full A100, L1 + AI time-sliced on same GPU")
