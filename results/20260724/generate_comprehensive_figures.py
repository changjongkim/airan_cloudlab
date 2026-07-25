#!/usr/bin/env python3
"""Comprehensive Chain 9-17 analysis — 10 figures for the final REPORT."""
import json, os, glob, csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260724"
FIG  = os.path.join(BASE, "figures", "comprehensive")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo","AppleGothic","DejaVu Sans"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False,
})

def load(name):
    p = f"{BASE}/{name}"
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return {}

c13 = load("chain13_summary.json")
c14 = load("chain14_summary.json")
c15 = load("chain15_summary.json")
c16 = load("chain16_summary.json")
c17 = load("chain17_summary.json")

# =========================================================================
# Figure 1 — Executive dashboard: cudaFree × 4 chains
# 3-row grid: (top) Chain 14 all workloads MPS off/on for config A,
#             (mid) Chain 16 multi-instance,
#             (bot) Chain 17 N-process sweep
# =========================================================================
fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.42, wspace=0.20)

# Row 1: Chain 14 Config A SP — all 11 workloads
wls14 = ["nrx","chanpred","qwen_rag","whisper","qwen_vl","hbm_stress",
         "qwen_chat_b1","whisper_stream_b1","bert_b1","embed_lookup","memcpy_loop"]
labels14 = ["NRx", "ChanPred", "Qwen-RAG\n(n=64)", "Whisper\n(b=4)", "Qwen-VL\n(b=2)",
            "HBM\nstress", "Qwen-chat\nb=1 eager", "Whisper\nstream b=1",
            "BERT\nb=1", "Embed\nlookup", "Memcpy\nloop"]
off14 = [c14.get(f"cfgA_SP_{w}_MPSoff",{}).get("cudaFree_ms",0) for w in wls14]
on14  = [c14.get(f"cfgA_SP_{w}_MPSon", {}).get("cudaFree_ms",0) for w in wls14]
base14 = c14.get("cfgA_SP0_baseline",{}).get("cudaFree_ms",1900)

ax = fig.add_subplot(gs[0, :])
x = np.arange(len(wls14)); w = 0.38
ax.bar(x-w/2, off14, w, color="#dc2626", label="MPS off", edgecolor="#7f1d1d")
ax.bar(x+w/2, on14,  w, color="#10b981", label="MPS on",  edgecolor="#065f46")
for i,(o,n) in enumerate(zip(off14,on14)):
    ax.text(i-w/2, o+250, f"{o:.0f}", ha="center", fontsize=8, fontweight="bold")
    ax.text(i+w/2, n+250, f"{n:.0f}", ha="center", fontsize=8, fontweight="bold")
ax.axhline(base14, color="#111", ls=":", lw=1.5, alpha=0.6, label=f"baseline {base14:.0f}ms")
ax.set_xticks(x); ax.set_xticklabels(labels14, fontsize=9)
ax.set_ylabel("L1 cudaFree total (ms, 30s window)", fontsize=11)
ax.set_title("(Row 1)  Chain 14 Config A (MIG 4g+3g) — 11 realistic AI workloads × MPS on/off",
             fontweight="bold", fontsize=12, pad=8)
ax.legend(loc="upper right", fontsize=10); ax.grid(axis="y", alpha=0.3)

# Row 2: Chain 16 multi-instance
wls16 = ["ranai_mix","ranai_mix_heavy","nrx_multi4"]
labels16 = ["ranai_mix\n(14 threads, 1 process)","ranai_mix_heavy\n(28 threads, 1 process)","nrx_multi4\n(4 processes)"]
base16 = c16.get("cfgA_SP0_baseline",{}).get("cudaFree_ms",2200)
off16_cf = [c16.get(f"cfgA_SP_{w}_MPSoff",{}).get("cudaFree_ms",0) for w in wls16]
on16_cf  = [c16.get(f"cfgA_SP_{w}_MPSon", {}).get("cudaFree_ms",0) for w in wls16]
off16_p99= [c16.get(f"cfgA_SP_{w}_MPSoff",{}).get("l1_p99_ms",0) for w in wls16]
on16_p99 = [c16.get(f"cfgA_SP_{w}_MPSon", {}).get("l1_p99_ms",0) for w in wls16]

ax = fig.add_subplot(gs[1, 0])
x = np.arange(3); w = 0.38
ax.bar(x-w/2, off16_cf, w, color="#dc2626", label="MPS off"); ax.bar(x+w/2, on16_cf, w, color="#10b981", label="MPS on")
for i,(o,n) in enumerate(zip(off16_cf,on16_cf)):
    ax.text(i-w/2, o+300, f"{o:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i+w/2, n+300, f"{n:.0f}", ha="center", fontsize=9, fontweight="bold")
ax.axhline(base16, color="#111", ls=":", alpha=0.5, label=f"baseline {base16:.0f}ms")
ax.set_xticks(x); ax.set_xticklabels(labels16, fontsize=9); ax.set_ylabel("cudaFree (ms)")
ax.set_title("(Row 2, left) Chain 16 — multi-instance cudaFree", fontweight="bold", fontsize=11)
ax.legend(loc="upper right", fontsize=9); ax.grid(axis="y", alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
ax.bar(x-w/2, off16_p99, w, color="#dc2626"); ax.bar(x+w/2, on16_p99, w, color="#10b981")
for i,(o,n) in enumerate(zip(off16_p99,on16_p99)):
    ax.text(i-w/2, o*1.05, f"{o:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i+w/2, n*1.05, f"{n:.0f}", ha="center", fontsize=9, fontweight="bold")
ax.axhline(43, color="#111", ls=":", alpha=0.5, label="baseline p99 ~43ms")
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(labels16, fontsize=9)
ax.set_ylabel("L1 p99 latency (ms, log)")
ax.set_title("(Row 2, right) Chain 16 — L1 p99 tail (HBM signature)", fontweight="bold", fontsize=11)
ax.grid(axis="y", alpha=0.3, which="both"); ax.legend(loc="upper right", fontsize=9)

# Row 3: Chain 17 Part A N-process sweep
Ns = [1,2,3,4,6,8]
off17_cf  = [c17.get(f"cfgA_A_nrxN{N}_MPSoff",{}).get("cudaFree_ms",0) for N in Ns]
on17_cf   = [c17.get(f"cfgA_A_nrxN{N}_MPSon", {}).get("cudaFree_ms",0) for N in Ns]
off17_p99 = [c17.get(f"cfgA_A_nrxN{N}_MPSoff",{}).get("l1_p99_ms",0) for N in Ns]
on17_p99  = [c17.get(f"cfgA_A_nrxN{N}_MPSon", {}).get("l1_p99_ms",0) for N in Ns]

ax = fig.add_subplot(gs[2, 0])
ax.plot(Ns, off17_cf, "o-", color="#dc2626", label="MPS off", linewidth=2.5, markersize=10)
ax.plot(Ns, on17_cf,  "s-", color="#10b981", label="MPS on",  linewidth=2.5, markersize=10)
for x,(o,n) in enumerate(zip(off17_cf, on17_cf)):
    if o > 0: ax.annotate(f"{o:.0f}", (Ns[x], o), textcoords="offset points", xytext=(0,10), ha="center", fontsize=9, color="#7f1d1d", fontweight="bold")
    if n > 0: ax.annotate(f"{n:.0f}", (Ns[x], n), textcoords="offset points", xytext=(0,-18), ha="center", fontsize=9, color="#065f46", fontweight="bold")
ax.axvspan(6, 8.3, alpha=0.15, color="#eab308", label="MPS breakdown zone")
ax.set_xlabel("N (concurrent NRx processes)"); ax.set_ylabel("cudaFree (ms)")
ax.set_title("(Row 3, left) Chain 17 Part A — N-process sweep, cudaFree", fontweight="bold", fontsize=11)
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[2, 1])
ax.plot(Ns, off17_p99, "o-", color="#dc2626", label="MPS off", linewidth=2.5, markersize=10)
ax.plot(Ns, on17_p99,  "s-", color="#10b981", label="MPS on",  linewidth=2.5, markersize=10)
ax.axhline(40, color="#111", ls=":", alpha=0.6, label="baseline p99 ~40ms")
ax.axvspan(6, 8.3, alpha=0.15, color="#eab308", label="MPS breakdown zone")
ax.set_yscale("log")
ax.set_xlabel("N (concurrent NRx processes)"); ax.set_ylabel("L1 p99 latency (ms, log)")
ax.set_title("(Row 3, right) Chain 17 Part A — L1 p99 tail vs N", fontweight="bold", fontsize=11)
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3, which="both")

fig.suptitle("Figure 1. Executive Dashboard — 3 experimental angles converging on MPS breakdown\n"
             "Row 1: single co-tenant workload class. Row 2: multi-instance one-shot. Row 3: process-count sensitivity.",
             fontsize=14, fontweight="bold", y=1.005)
plt.savefig(f"{FIG}/f01_executive_dashboard.png", dpi=140, bbox_inches='tight'); plt.close()
print("saved f01_executive_dashboard.png")

# =========================================================================
# Figure 2 — Kernel launch rate vs sync penalty (Chain 12 memory + Chain 14)
# Log-log scatter
# =========================================================================
# Extract (launches_n, cudaFree_ms) from all Config A SP conditions
scatter_data = []
for k, s in c14.items():
    if "cfgA_SP_" not in k or "MPSoff" not in k: continue
    l = s.get("launches_n", 0); cf = s.get("cudaFree_ms", 0)
    if l > 0 and cf > 0:
        wl = k.replace("cfgA_SP_","").replace("_MPSoff","")
        scatter_data.append((l, cf, wl))
for k, s in c17.items():
    if "cfgA_A_nrx" not in k or "MPSoff" not in k: continue
    l = s.get("launches_n", 0); cf = s.get("cudaFree_ms", 0)
    if l > 0 and cf > 0:
        wl = k.replace("cfgA_A_","").replace("_MPSoff","")
        scatter_data.append((l, cf, wl))

fig, ax = plt.subplots(figsize=(12, 7))
if scatter_data:
    xs = [d[0] for d in scatter_data]; ys = [d[1] for d in scatter_data]; labs = [d[2] for d in scatter_data]
    ax.scatter(xs, ys, s=160, c="#3b82f6", edgecolor="#111", alpha=0.7, linewidth=1)
    for x, y, l in scatter_data:
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(8, 3), fontsize=9, alpha=0.85)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Kernel launches per 30s window (log)", fontsize=12)
ax.set_ylabel("L1 cudaFree total (ms, log)", fontsize=12)
ax.set_title("Figure 2. Kernel launch rate → sync penalty correlation (Config A, MPSoff)\n"
             "Chain 12/14 finding: sync scales with launch count, not memory-boundness or HBM utilization",
             fontsize=12, fontweight="bold", pad=10)
ax.grid(True, which="both", alpha=0.3)
ax.axhline(2000, color="#111", ls=":", alpha=0.5, label="baseline ~2000ms")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(f"{FIG}/f02_launch_vs_sync.png", dpi=140, bbox_inches='tight'); plt.close()
print("saved f02_launch_vs_sync.png")

# =========================================================================
# Figure 3 — MIG cross-partition isolation across 13 workloads (Chain 14 CP)
# =========================================================================
cp_wls = ["idle","nrx","chanpred","qwen_rag","whisper","qwen_vl","hbm_stress",
          "qwen_chat_b1","whisper_stream_b1","bert_b1","embed_lookup","memcpy_loop","qwen_llm_cross"]
a_cf = [c14.get(f"cfgA_CP_{w}",{}).get("cudaFree_ms",0) for w in cp_wls]
c_cf = [c14.get(f"cfgC_CP_{w}",{}).get("cudaFree_ms",0) for w in cp_wls]
a_p99 = [c14.get(f"cfgA_CP_{w}",{}).get("l1_p99_ms",0) for w in cp_wls]
c_p99 = [c14.get(f"cfgC_CP_{w}",{}).get("l1_p99_ms",0) for w in cp_wls]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5.5))
x = np.arange(len(cp_wls))
w = 0.4
ax1.bar(x-w/2, a_cf, w, color="#3b82f6", label="Config A (4g+3g)")
ax1.bar(x+w/2, c_cf, w, color="#eab308", label="Config C (3g+2g+2g)")
ax1.axhspan(1400, 2200, color="#059669", alpha=0.15, label="baseline band")
ax1.set_xticks(x); ax1.set_xticklabels(cp_wls, rotation=30, ha="right", fontsize=9)
ax1.set_ylabel("L1 cudaFree (ms)")
ax1.set_title("cudaFree (all near baseline)", fontweight="bold")
ax1.legend(); ax1.grid(axis="y", alpha=0.3)

ax2.bar(x-w/2, a_p99, w, color="#3b82f6", label="Config A p99")
ax2.bar(x+w/2, c_p99, w, color="#eab308", label="Config C p99")
ax2.axhspan(38, 45, color="#059669", alpha=0.15, label="baseline band")
ax2.set_xticks(x); ax2.set_xticklabels(cp_wls, rotation=30, ha="right", fontsize=9)
ax2.set_ylabel("L1 p99 latency (ms)")
ax2.set_title("L1 p99 (all within 5ms of baseline)", fontweight="bold")
ax2.legend(); ax2.grid(axis="y", alpha=0.3)

plt.suptitle("Figure 3. MIG cross-partition = perfect isolation (Chain 14 CP)\n"
             "13 workloads × 2 configs: L1 in dedicated partition, workload in other. cudaFree + p99 stay near baseline.",
             fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(f"{FIG}/f03_mig_cross_isolation.png", dpi=140, bbox_inches='tight'); plt.close()
print("saved f03_mig_cross_isolation.png")

# =========================================================================
# Figure 4 — Chain 15 batch sweep (4 workloads, 17 batch variants)
# =========================================================================
sweeps = {
    "Qwen-3B chat": (["1","2","4","8","16","32"], "qwen_chat"),
    "BERT-large":   (["1","4","16","64"], "bert"),
    "Whisper":      (["1","2","4","8"],  "whisper"),
    "Qwen-VL-2B":   (["1","2","4"], "vl"),
}
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=False)
for ax, (title, (bs, wname)) in zip(axes.flat, sweeps.items()):
    off = [c15.get(f"cfgA_SP_{wname}_b{b}_MPSoff",{}).get("cudaFree_ms",0) for b in bs]
    on  = [c15.get(f"cfgA_SP_{wname}_b{b}_MPSon", {}).get("cudaFree_ms",0) for b in bs]
    x = np.arange(len(bs))
    ax.plot(x, off, "o-", color="#dc2626", label="MPS off", linewidth=2.5, markersize=10)
    ax.plot(x, on,  "s-", color="#10b981", label="MPS on",  linewidth=2.5, markersize=10)
    for xi,(o,n) in enumerate(zip(off,on)):
        ax.text(xi, o+80, f"{o:.0f}", ha="center", fontsize=8, color="#7f1d1d", fontweight="bold")
        ax.text(xi, n-140, f"{n:.0f}", ha="center", fontsize=8, color="#065f46", fontweight="bold")
    ax.axhline(1900, color="#111", ls=":", alpha=0.5, label="baseline")
    ax.set_xticks(x); ax.set_xticklabels([f"b={b}" for b in bs])
    ax.set_title(title, fontweight="bold"); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)
    ax.set_ylabel("cudaFree (ms)")
fig.suptitle("Figure 4. Chain 15 — Batch size sweep on 4 realistic workloads (Config A)\n"
             "Framework-optimized workloads (vLLM/HF) show minimal sync variation across batch sizes",
             fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(f"{FIG}/f04_batch_sweep.png", dpi=140, bbox_inches='tight'); plt.close()
print("saved f04_batch_sweep.png")

# =========================================================================
# Figure 5 — 3 configs comparison heatmap for compute-bound sync
# =========================================================================
fig, ax = plt.subplots(figsize=(12, 6))
configs = ["A (MIG 4g+3g)", "B (Full GPU)", "C (3g+2g+2g)"]
key_wls = ["nrx","chanpred","memcpy_loop","embed_lookup","ranai_mix","ranai_mix_heavy","nrx_multi4"]

matrix = np.zeros((len(configs), len(key_wls)))
for i, cfg in enumerate(["cfgA","cfgB","cfgC"]):
    for j, wl in enumerate(key_wls):
        # p99 ratio: MPSoff p99 / baseline p99
        d_src = c16 if wl in ["ranai_mix","ranai_mix_heavy","nrx_multi4"] else c14
        baseline = d_src.get(f"{cfg}_SP0_baseline",{}).get("l1_p99_ms",42)
        off_p99  = d_src.get(f"{cfg}_SP_{wl}_MPSoff",{}).get("l1_p99_ms",0)
        matrix[i,j] = off_p99 / baseline if baseline > 0 else 0

im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=30)
ax.set_xticks(range(len(key_wls))); ax.set_xticklabels(key_wls, rotation=30, ha="right")
ax.set_yticks(range(len(configs))); ax.set_yticklabels(configs)
for i in range(len(configs)):
    for j in range(len(key_wls)):
        val = matrix[i,j]
        if val > 0:
            ax.text(j, i, f"{val:.1f}×", ha="center", va="center",
                    color="white" if val > 15 else "black", fontsize=10, fontweight="bold")
cbar = plt.colorbar(im); cbar.set_label("L1 p99 / baseline p99 (log-ish scale)")
ax.set_title("Figure 5. MPS off L1 p99 penalty heatmap — 3 configs × 7 workloads\n"
             "Red = severe sync + HBM contention; Green = baseline. Multi-process workloads darkest.",
             fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(f"{FIG}/f05_config_workload_heatmap.png", dpi=140, bbox_inches='tight'); plt.close()
print("saved f05_config_workload_heatmap.png")

# =========================================================================
# Figure 6 — Chain 17 Part A: N-process sweep triple view (3 configs)
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
for ax, cfg, name in zip(axes, ["cfgA","cfgB","cfgC"], ["A (4g+3g)","B (Full GPU)","C (3g+2g+2g)"]):
    off = [c17.get(f"{cfg}_A_nrxN{N}_MPSoff",{}).get("l1_p99_ms",0) for N in Ns]
    on  = [c17.get(f"{cfg}_A_nrxN{N}_MPSon",{}).get("l1_p99_ms",0) for N in Ns]
    ax.plot(Ns, off, "o-", color="#dc2626", label="MPS off", linewidth=2.5, markersize=10)
    ax.plot(Ns, on,  "s-", color="#10b981", label="MPS on", linewidth=2.5, markersize=10)
    for xi,(o,n) in enumerate(zip(off,on)):
        if o > 0: ax.annotate(f"{o:.0f}", (Ns[xi], o), textcoords="offset points", xytext=(0,8), ha="center", fontsize=8, color="#7f1d1d", fontweight="bold")
        if n > 0: ax.annotate(f"{n:.0f}", (Ns[xi], n), textcoords="offset points", xytext=(0,-15), ha="center", fontsize=8, color="#065f46", fontweight="bold")
    ax.axhline(40, color="#111", ls=":", alpha=0.5, label="baseline p99")
    ax.axvspan(6, 8.3, alpha=0.15, color="#eab308")
    ax.set_yscale("log")
    ax.set_xlabel("N processes"); ax.set_title(f"Config {name}", fontweight="bold")
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="upper left", fontsize=9)
axes[0].set_ylabel("L1 p99 (ms, log)")
plt.suptitle("Figure 6. Chain 17 Part A — N-process sweep across 3 partition configs\n"
             "MPS breakdown at N=6 consistent across Config A/C; Config B (Full GPU) more resilient",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/f06_Nsweep_all_configs.png", dpi=140, bbox_inches='tight'); plt.close()
print("saved f06_Nsweep_all_configs.png")

# =========================================================================
# Figure 7 — Chain 17 Part B: MPS thread% cap effect (all 7 workloads)
# =========================================================================
pcts = [100, 70, 50, 30]
wls_b = ["nrx", "chanpred", "memcpy_loop", "embed_lookup", "ranai_mix", "ranai_mix_heavy", "nrx_multi4"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for ax, metric, ylabel, title in [
    (ax1, "cudaFree_ms", "L1 cudaFree (ms)", "cudaFree vs MPS thread%"),
    (ax2, "l1_p99_ms",   "L1 p99 (ms)",       "L1 p99 tail vs MPS thread%"),
]:
    for wl in wls_b:
        vals = [c17.get(f"cfgA_B_{wl}_pct{p}",{}).get(metric,0) for p in pcts]
        if all(v == 0 for v in vals): continue
        ax.plot(pcts, vals, "o-", label=wl, linewidth=2, markersize=8)
    ax.axhline(43 if "p99" in metric else 1900, color="#111", ls=":", alpha=0.5, label="baseline")
    ax.set_xlabel("AI CUDA_MPS_ACTIVE_THREAD_PERCENTAGE (%)")
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.invert_xaxis()
plt.suptitle("Figure 7. Chain 17 Part B — MPS thread% cap effect (Config A, 7 workloads)\n"
             "For multi-process (nrx_multi4): sweet spot at pct=70 (p99 96 → 56 ms, 42% improvement)",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/f07_thread_pct_all_workloads.png", dpi=140, bbox_inches='tight'); plt.close()
print("saved f07_thread_pct_all_workloads.png")

# =========================================================================
# Figure 8 — Multi-thread vs multi-process comparison (Chain 16)
# =========================================================================
fig, ax = plt.subplots(figsize=(11, 6.5))
categories = ["baseline\n(alone)",
              "ranai_mix\n(14 thr, 1 proc)\nMPS off",
              "ranai_mix\n(14 thr, 1 proc)\nMPS on",
              "ranai_mix_heavy\n(28 thr, 1 proc)\nMPS off",
              "ranai_mix_heavy\n(28 thr, 1 proc)\nMPS on",
              "nrx_multi4\n(4 procs)\nMPS off",
              "nrx_multi4\n(4 procs)\nMPS on"]
p99_values = [
    c16.get("cfgA_SP0_baseline",{}).get("l1_p99_ms",43),
    c16.get("cfgA_SP_ranai_mix_MPSoff",{}).get("l1_p99_ms",0),
    c16.get("cfgA_SP_ranai_mix_MPSon",{}).get("l1_p99_ms",0),
    c16.get("cfgA_SP_ranai_mix_heavy_MPSoff",{}).get("l1_p99_ms",0),
    c16.get("cfgA_SP_ranai_mix_heavy_MPSon",{}).get("l1_p99_ms",0),
    c16.get("cfgA_SP_nrx_multi4_MPSoff",{}).get("l1_p99_ms",0),
    c16.get("cfgA_SP_nrx_multi4_MPSon",{}).get("l1_p99_ms",0),
]
colors = ["#3b82f6","#dc2626","#10b981","#dc2626","#10b981","#dc2626","#f59e0b"]
xs = np.arange(len(categories))
ax.bar(xs, p99_values, color=colors, edgecolor="#111")
for x,v in zip(xs, p99_values):
    ax.text(x, v*1.1, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")
ax.axhline(p99_values[0], color="#111", ls=":", alpha=0.5, label=f"baseline ({p99_values[0]:.0f}ms)")
ax.set_yscale("log")
ax.set_xticks(xs); ax.set_xticklabels(categories, fontsize=8)
ax.set_ylabel("L1 p99 latency (ms, log)")
ax.set_title("Figure 8. Chain 16 — Multi-thread vs multi-process co-tenancy\n"
             "Single process w/ many threads: MPS FULLY recovers.  Multi-process: MPS PARTIAL only.",
             fontweight="bold", pad=10)
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(f"{FIG}/f08_thread_vs_process.png", dpi=140, bbox_inches='tight'); plt.close()
print("saved f08_thread_vs_process.png")

# =========================================================================
# Figure 9 — Deployment decision tree
# =========================================================================
fig, ax = plt.subplots(figsize=(15, 10))
ax.axis("off")
ax.text(0.5, 0.97, "AI-RAN GPU deployment decision tree",
        transform=ax.transAxes, ha="center", fontsize=15, fontweight="bold")
ax.text(0.5, 0.93, "based on Chain 9–17 (1000+ nsys captures, 20+ workloads)",
        transform=ax.transAxes, ha="center", fontsize=10, style="italic")

# Root box: Can L1 have its own MIG partition?
def box(x, y, w, h, text, color, fontsize=10):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.5,
                                     facecolor=color, edgecolor="#111")
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize, wrap=True)

def arrow(x1, y1, x2, y2, label="", labcolor="black"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
    if label: ax.text((x1+x2)/2 + 0.02, (y1+y2)/2, label, fontsize=9, color=labcolor)

# Tier 1: MIG cross-partition question
box(0.30, 0.80, 0.40, 0.08,
    "Q1. Can L1 have its own MIG partition?\n(different from AI workloads)",
    "#dbeafe", 12)

box(0.05, 0.62, 0.30, 0.10,
    "✓ MIG CROSS-PARTITION\nBest option:\nL1 in one, AI in others\nPerfect isolation (Chain 13/14)",
    "#bbf7d0", 10)
box(0.65, 0.62, 0.30, 0.10,
    "Q2. How many concurrent\nAI processes in\nthe SAME partition as L1?",
    "#fef3c7", 10)

arrow(0.40, 0.80, 0.20, 0.72, "YES", "green")
arrow(0.60, 0.80, 0.80, 0.72, "NO", "orange")

# Q2 branches
box(0.42, 0.44, 0.20, 0.10,
    "1 process\nmulti-thread only\n\nMPS on → PERFECT",
    "#bbf7d0", 9)
box(0.65, 0.44, 0.20, 0.10,
    "N = 2–4 processes\n\nMPS on\n→ p99 ~80ms (2× baseline)\nAcceptable",
    "#fef9c3", 9)
box(0.88, 0.44, 0.10, 0.10,
    "N = 6–8\n\nMPS breaks\n→ p99 332-418ms\n(8-10×)",
    "#fecaca", 9)

arrow(0.80, 0.62, 0.52, 0.54, "1", "green")
arrow(0.80, 0.62, 0.75, 0.54, "2-4", "orange")
arrow(0.80, 0.62, 0.93, 0.54, "≥6", "red")

# Recommendations
box(0.30, 0.24, 0.40, 0.14,
    "Fine-tuning:\n  MPS thread% = 70 (multi-process)\n  → p99 96 → 56 ms (42% ↓)",
    "#e0e7ff", 10)
arrow(0.75, 0.44, 0.55, 0.38, "", "black")
arrow(0.50, 0.44, 0.50, 0.38, "", "black")

box(0.30, 0.05, 0.40, 0.12,
    "MPS off = ALWAYS BAD\n\nAll cases → 6–30× baseline\n(Chain 14 NRx, ChanPred, Chain 17 all N)",
    "#fecaca", 10)

plt.tight_layout()
plt.savefig(f"{FIG}/f09_decision_tree.png", dpi=140, bbox_inches='tight'); plt.close()
print("saved f09_decision_tree.png")

# =========================================================================
# Figure 10 — Chronological chain progression (what we learned session-by-session)
# =========================================================================
fig, ax = plt.subplots(figsize=(16, 8))
chains = ["Chain 9-12\n(prior)", "Chain 13", "Chain 14", "Chain 15", "Chain 16", "Chain 17"]
positions = [1, 2, 3, 4, 5, 6]
findings = [
    "API-layer shims fail\nsync=launch rate\nCUDA graphs bypass sync",
    "MIG cross-part perfect isolation\n(NRx only)\nMPS on same-part = full recovery",
    "11 realistic workloads\nCross-part = perfect × 13\nMPS on = recovery",
    "Batch sweep 4 workloads\nFramework opt shields sync\nBatch has weak effect",
    "Multi-thread MPS = ✓\nMulti-process MPS = ⚠️\nHBM residual (p99 2.6×)",
    "N-process sweep confirms\nMPS breakdown at N=6\nThread% cap = tuning knob",
]
completeness = [70, 78, 85, 88, 92, 100]

ax.plot(positions, completeness, "o-", linewidth=3, markersize=15, color="#3b82f6")
for i,(p,c,f) in enumerate(zip(positions, completeness, findings)):
    ax.annotate(f, xy=(p, c), textcoords="offset points", xytext=(0, -70), ha="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="#dbeafe", ec="#3b82f6"))
    ax.text(p, c+2, f"{c}%", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(positions); ax.set_xticklabels(chains, fontsize=11)
ax.set_ylabel("Cumulative understanding (%)", fontsize=12)
ax.set_ylim(0, 110); ax.grid(axis="y", alpha=0.3)
ax.set_title("Figure 10. Chain 9→17 chronological progression of the AI-RAN sync/isolation story\n"
             "Each chain added a new experimental dimension. Chain 17 quantifies the breakdown curve.",
             fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(f"{FIG}/f10_chain_progression.png", dpi=140, bbox_inches='tight'); plt.close()
print("saved f10_chain_progression.png")

print(f"\nAll 10 comprehensive figures saved to {FIG}")
