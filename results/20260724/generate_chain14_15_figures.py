#!/usr/bin/env python3
"""Chain 14 + Chain 15 figures.

- fig_ch14_sp_gradient: MPS on/off cudaFree by workload class (config A)
- fig_ch14_cp: cross-partition isolation across workloads (config A/C)
- fig_ch14_config_compare: Same workload across 3 configs (A vs B vs C)
- fig_ch15_batch_sweep: cudaFree vs batch size (4 workload types)
- fig_ch15_mps_effect: MPS on/off × batch size heatmap
"""
import json, os
import matplotlib.pyplot as plt
import numpy as np

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260724"
FIG  = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo","AppleGothic","DejaVu Sans"], "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
})

def load(name):
    p = f"{BASE}/{name}"
    if not os.path.exists(p): return {}
    with open(p) as f: return json.load(f)

c14 = load("chain14_summary.json")
c15 = load("chain15_summary.json")

# ============================================================
# Figure 1 — Chain 14 Config A: SP MPS off vs on by workload
# ============================================================
wls = ["nrx","chanpred","qwen_rag","whisper","qwen_vl","hbm_stress",
       "qwen_chat_b1","whisper_stream_b1","bert_b1","embed_lookup","memcpy_loop"]
labels = ["NRx\n(compute)","ChanPred\n(compute)","Qwen-RAG\n(batch n=64)","Whisper\n(b=4)",
          "Qwen-VL\n(b=2)","HBM_stress\n(triad)","Qwen-chat\nb=1 eager","Whisper\nstream b=1",
          "BERT\nb=1","Embed\nlookup","Memcpy\nloop"]
off_v = [c14.get(f"cfgA_SP_{w}_MPSoff",{}).get("cudaFree_ms",0) for w in wls]
on_v  = [c14.get(f"cfgA_SP_{w}_MPSon", {}).get("cudaFree_ms",0) for w in wls]

x = np.arange(len(wls)); w = 0.38
fig, ax = plt.subplots(figsize=(15,6))
ax.bar(x-w/2, off_v, w, color="#dc2626", label="MPS off (temporal)")
ax.bar(x+w/2, on_v,  w, color="#10b981", label="MPS on (spatial)")
for i,(o,n) in enumerate(zip(off_v,on_v)):
    ax.text(i-w/2, o+300, f"{o:.0f}", ha="center", fontsize=8, fontweight="bold")
    ax.text(i+w/2, n+300, f"{n:.0f}", ha="center", fontsize=8, fontweight="bold")

baseline = c14.get("cfgA_SP0_baseline",{}).get("cudaFree_ms",1700)
ax.axhline(baseline, color="#111", linestyle=":", alpha=0.6, label=f"baseline {baseline:.0f}ms")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("L1 cudaFree total (ms, 30s window)")
ax.set_title("Figure 1. Chain 14 Config A (MIG 4g+3g) — SP MPS on/off across 11 realistic workloads",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{FIG}/ch14_A_sp_mps.png", dpi=150, bbox_inches='tight'); plt.close()
print("saved ch14_A_sp_mps.png")

# ============================================================
# Figure 2 — Chain 14 CP isolation (config A vs C)
# ============================================================
cp_wls = ["idle","nrx","chanpred","qwen_rag","whisper","qwen_vl","hbm_stress",
          "qwen_chat_b1","whisper_stream_b1","bert_b1","embed_lookup","memcpy_loop","qwen_llm_cross"]
a_v = [c14.get(f"cfgA_CP_{w}",{}).get("cudaFree_ms",0) for w in cp_wls]
c_v = [c14.get(f"cfgC_CP_{w}",{}).get("cudaFree_ms",0) for w in cp_wls]

fig, ax = plt.subplots(figsize=(14,5))
x = np.arange(len(cp_wls))
ax.bar(x-0.2, a_v, 0.4, color="#3b82f6", label="Config A (4g+3g)")
ax.bar(x+0.2, c_v, 0.4, color="#eab308", label="Config C (3g+2g+2g)")
ax.axhline(2000, color="#111", linestyle=":", alpha=0.5, label="baseline ~2000ms")
ax.set_xticks(x); ax.set_xticklabels(cp_wls, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("L1 cudaFree total (ms)")
ax.set_title("Figure 2. Chain 14 CP — L1 in dedicated partition, workload in another partition\n"
             "All workloads → L1 stays near baseline (MIG cross-part isolation)",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{FIG}/ch14_cp_isolation.png", dpi=150, bbox_inches='tight'); plt.close()
print("saved ch14_cp_isolation.png")

# ============================================================
# Figure 3 — Chain 15 batch sweep for each workload type
# ============================================================
sweeps = {
    "Qwen-3B chat": (["1","2","4","8","16","32"], "qwen_chat"),
    "BERT-large":   (["1","4","16","64"],          "bert"),
    "Whisper":      (["1","2","4","8"],            "whisper"),
    "Qwen-VL-2B":   (["1","2","4"],                "vl"),
}

fig, axes = plt.subplots(1, 4, figsize=(18,5), sharey=True)
for ax, (title, (bs, wname)) in zip(axes, sweeps.items()):
    off = [c15.get(f"cfgA_SP_{wname}_b{b}_MPSoff",{}).get("cudaFree_ms",0) for b in bs]
    on  = [c15.get(f"cfgA_SP_{wname}_b{b}_MPSon", {}).get("cudaFree_ms",0) for b in bs]
    x = np.arange(len(bs))
    ax.plot(x, off, "o-", color="#dc2626", label="MPS off", linewidth=2, markersize=8)
    ax.plot(x, on,  "s-", color="#10b981", label="MPS on",  linewidth=2, markersize=8)
    for xi,(o,n) in enumerate(zip(off,on)):
        ax.text(xi, o+100, f"{o:.0f}", ha="center", fontsize=8, color="#dc2626")
        ax.text(xi, n-200, f"{n:.0f}", ha="center", fontsize=8, color="#10b981")
    ax.set_xticks(x); ax.set_xticklabels([f"b={b}" for b in bs])
    ax.set_title(title, fontweight="bold"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
axes[0].set_ylabel("L1 cudaFree total (ms)")
plt.suptitle("Figure 3. Chain 15 Config A — cudaFree vs batch size (memory→compute transition)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout(); plt.savefig(f"{FIG}/ch15_batch_sweep.png", dpi=150, bbox_inches='tight'); plt.close()
print("saved ch15_batch_sweep.png")

# ============================================================
# Figure 4 — Full GPU (Config B) MPS breakdown by workload
# ============================================================
b_off = [c14.get(f"cfgB_SP_{w}_MPSoff",{}).get("cudaFree_ms",0) for w in wls]
b_on  = [c14.get(f"cfgB_SP_{w}_MPSon", {}).get("cudaFree_ms",0) for w in wls]

fig, ax = plt.subplots(figsize=(15,6))
xB = np.arange(len(wls)); wB = 0.38
ax.bar(xB-wB/2, b_off, wB, color="#dc2626", label="MPS off (temporal)")
ax.bar(xB+wB/2, b_on,  wB, color="#10b981", label="MPS on (spatial)")
for i,(o,n) in enumerate(zip(b_off,b_on)):
    ax.text(i-wB/2, o+300, f"{o:.0f}", ha="center", fontsize=8, fontweight="bold")
    ax.text(i+wB/2, n+300, f"{n:.0f}", ha="center", fontsize=8, fontweight="bold")
baseline_b = c14.get("cfgB_SP0_baseline",{}).get("cudaFree_ms",1700)
ax.axhline(baseline_b, color="#111", linestyle=":", alpha=0.6, label=f"baseline {baseline_b:.0f}ms")
ax.set_xticks(xB); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("L1 cudaFree total (ms)")
ax.set_title("Figure 4. Chain 14 Config B (Full GPU, no MIG) — SP MPS on/off\n"
             "HBM_stress row shows L1 startup failure (5ms) — Full GPU + preallocation clash",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{FIG}/ch14_B_sp_mps.png", dpi=150, bbox_inches='tight'); plt.close()
print("saved ch14_B_sp_mps.png")

print("\nAll figures saved to", FIG)
