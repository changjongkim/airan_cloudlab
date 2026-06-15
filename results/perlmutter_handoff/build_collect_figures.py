#!/usr/bin/env python3
"""figF10 (P5 sustained vs burst) and figF11 (NCU DRAM no-MIG vs MIG)."""
import json, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({"font.size":11,"savefig.dpi":145,"savefig.bbox":"tight",
                     "axes.grid":True,"grid.alpha":0.3,"font.family":["DejaVu Sans","sans-serif"]})
R="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig"
OUT="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/figures"
C_BURST="#fb8072"; C_SUST="#d7301f"; C_NM="#d7301f"; C_MIG="#2c7fb8"
def p99(d,pat):
    ms=[]
    for f in glob.glob(d+pat):
        try: ms+=json.load(open(f))["raw_ms"]
        except: pass
    return np.percentile(ms,99) if ms else np.nan

# ---- figF10: P5 sustained p99 vs F burst p99 ----
order=[("alone","alone","F_0_alone"),("xApp","xapp","F_E_xapp"),("qwen","qwen_small","F_E_qwen_small"),
       ("chanpred","chanpred","F_E_chanpred_b64"),("forecaster","forecaster","F_E_forecaster_d384"),
       ("NeuralRx","neuralrx","F_E_neuralrx"),("sat_compute","sat_compute","F_E_sat_compute"),
       ("resnet","resnet","F_E_resnet_b64"),("sat_hbm","sat_hbm","F_E_sat_hbm")]
disp=[o[0] for o in order]
sust=[p99(f"{R}/P5_nomig/",f"realL1_P5_{o[1]}_run*.json") for o in order]
burst=[p99(f"{R}/F_nomig/",f"realL1_{o[2]}_run*.json") for o in order]
fig,ax=plt.subplots(figsize=(12,6)); x=np.arange(len(disp)); w=0.4
ax.bar(x-w/2,burst,w,color=C_BURST,label="F burst (ITERS=100)")
ax.bar(x+w/2,sust,w,color=C_SUST,label="P5 sustained (~5 min)")
ax.set_ylabel("L1 frame-time p99 (ms)")
ax.set_title("Contention PERSISTS over 5-min sustained load (not a transient warmup)")
ax.set_xticks(x); ax.set_xticklabels(disp,rotation=30,ha="right")
for xi,v in zip(x+w/2,sust): ax.text(xi,v+5,f"{v:.0f}",ha="center",fontsize=8,color=C_SUST)
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/figF10_p5_sustained.png"); plt.close(fig)

# ---- figF11: NCU DRAM throughput no-MIG vs MIG (both low, both unchanged) ----
fig,ax=plt.subplots(figsize=(9,6))
cats=["L1 alone","L1 + NeuralRx"]
nm=[8.56,8.07]   # measured no-MIG (mean over 11056 kernels)
mg=[11.1,11.0]   # CloudLab MIG (upper doc §18.2)
xx=np.arange(2); w=0.35
ax.bar(xx-w/2,mg,w,color=C_MIG,label="MIG (CloudLab §18.2)")
ax.bar(xx+w/2,nm,w,color=C_NM,label="no-MIG (Perlmutter, measured)")
for xi,v in zip(xx-w/2,mg): ax.text(xi,v+0.2,f"{v:.1f}%",ha="center")
for xi,v in zip(xx+w/2,nm): ax.text(xi,v+0.2,f"{v:.2f}%",ha="center")
ax.set_xticks(xx); ax.set_xticklabels(cats)
ax.set_ylabel("L1-kernel DRAM throughput (% of peak)"); ax.set_ylim(0,100)
ax.set_title("Throughput hypothesis REJECTED on both: L1 DRAM ~8-11%, unchanged by NeuralRx\n(slowdown is queue arbitration, not DRAM saturation)")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/figF11_ncu_dram_throughput.png"); plt.close(fig)
print("figF10/figF11 written. P5 sustained:",{d:round(s,0) for d,s in zip(disp,sust)})
