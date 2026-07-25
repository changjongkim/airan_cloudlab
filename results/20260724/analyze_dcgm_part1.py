#!/usr/bin/env python3
"""Chain 18 Part 1 — DCGM time-series analysis.

Parses Chain 17 DCGM tsv files (per-100ms samples of DRAM_ACTIVE, SM_ACTIVE, etc.)
and correlates with L1 kernel activity timing.

Output:
  - dcgm_stats.json: mean/p95/max DRAM & SM utilization per condition
  - figures/comprehensive/f11_dcgm_timeseries.png: time-series overlay
  - figures/comprehensive/f12_dcgm_summary.png: aggregate view by config × MPS
"""
import os, glob, json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260724"
CH17 = os.path.join(BASE, "chain17")
FIG  = os.path.join(BASE, "figures", "comprehensive")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo","AppleGothic","DejaVu Sans"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False,
})

# DCGM field IDs (per dcgmi doc):
FIELDS = {
    1001: "SM_ACTIVE",
    1002: "SM_OCCUPANCY",
    1003: "TENSOR_ACTIVE",
    1004: "DRAM_ACTIVE",
    1005: "FP64_ACTIVE",
    1006: "FP32_ACTIVE",
    1007: "FP16_ACTIVE",
    1008: "PIPE_TENSOR_ACTIVE",
}

def parse_dcgm_tsv(path):
    """Parse dcgmi dmon output. Returns dict of field_name -> list of floats."""
    data = defaultdict(list)
    with open(path, errors='ignore') as f:
        lines = f.readlines()
    header_idx = None
    for i, ln in enumerate(lines):
        if "GPU" in ln and "SMACT" in ln:
            header_idx = i; break
    if header_idx is None:
        # Try different format
        for i, ln in enumerate(lines):
            if any(fn in ln for fn in ["SM_ACT","DRAM"]):
                header_idx = i; break
    if header_idx is None: return {}
    # Parse header to find columns
    header = lines[header_idx].split()
    for ln in lines[header_idx+1:]:
        parts = ln.split()
        if len(parts) < len(header): continue
        for name, val in zip(header, parts):
            try:
                v = float(val)
                if 0 <= v <= 100: data[name].append(v)
            except ValueError:
                pass
    return data

# ============================================================
# Parse all dcgm files
# ============================================================
stats = {}
files = sorted(glob.glob(f"{CH17}/*_dcgm.tsv"))
print(f"Found {len(files)} DCGM tsv files")

for f in files:
    label = os.path.basename(f).replace("_dcgm.tsv","")
    data = parse_dcgm_tsv(f)
    if not data: continue
    s = {}
    for field, vals in data.items():
        if len(vals) < 5: continue
        arr = np.array(vals)
        s[field] = {"mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)),
                    "max": float(arr.max()), "n": len(arr)}
    stats[label] = s

with open(f"{BASE}/dcgm_stats.json","w") as fp: json.dump(stats, fp, indent=2)
print(f"Wrote dcgm_stats.json with {len(stats)} conditions")

# ============================================================
# Figure 11 — DCGM time-series overlay for key conditions
# ============================================================
def load_ts(label):
    """Return {field: array} for one label."""
    p = f"{CH17}/{label}_dcgm.tsv"
    if not os.path.exists(p): return {}
    return {k: np.array(v) for k, v in parse_dcgm_tsv(p).items() if len(v) > 5}

# Compare key conditions: N-process sweep (Config A, MPS on)
key_conds = [
    ("cfgA_A_nrxN1_MPSon_t1",  "N=1"),
    ("cfgA_A_nrxN2_MPSon_t1",  "N=2"),
    ("cfgA_A_nrxN4_MPSon_t1",  "N=4"),
    ("cfgA_A_nrxN6_MPSon_t1",  "N=6"),
    ("cfgA_A_nrxN8_MPSon_t1",  "N=8"),
]
fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
colors = ["#3b82f6","#10b981","#f59e0b","#f97316","#dc2626"]
for (label, name), color in zip(key_conds, colors):
    ts = load_ts(label)
    if not ts: continue
    for field, arr in ts.items():
        if "DRAM" in field.upper() or "MEM" in field.upper() or "FBMEM" in field.upper() or field.upper() in ("DRAM","MEM","FBMU"):
            x = np.arange(len(arr)) * 0.1  # 100ms sampling
            axes[0].plot(x, arr, label=f"{name} DRAM", color=color, linewidth=1.5, alpha=0.85); break
    for field, arr in ts.items():
        if "SM_ACT" in field.upper() or field.upper() in ("SMACT","SM"):
            x = np.arange(len(arr)) * 0.1
            axes[1].plot(x, arr, label=f"{name} SM", color=color, linewidth=1.5, alpha=0.85); break

axes[0].set_ylabel("DRAM ACTIVE (%)"); axes[0].set_title("DRAM utilization over 30s window", fontweight="bold")
axes[1].set_ylabel("SM ACTIVE (%)"); axes[1].set_title("SM utilization over 30s window", fontweight="bold")
axes[1].set_xlabel("Time (s)")
for ax in axes: ax.legend(loc="upper right", fontsize=9, ncol=3); ax.grid(alpha=0.3); ax.set_ylim(0, 105)
plt.suptitle("Figure 11. Chain 17 DCGM time-series — N-process sweep (Config A, MPS on)\n"
             "DRAM/SM utilization pattern as N increases (100ms sampling)",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/f11_dcgm_timeseries.png", dpi=140, bbox_inches='tight'); plt.close()
print("saved f11_dcgm_timeseries.png")

# ============================================================
# Figure 12 — Aggregate: mean DRAM/SM utilization across all conditions
# ============================================================
def collect_metric(field_pattern):
    """Collect mean values grouped by (config, workload, N, MPS)."""
    out = defaultdict(list)
    for label, s in stats.items():
        for field, m in s.items():
            if field_pattern.upper() in field.upper():
                out[label].append(m["mean"])
                break
    return {k: np.mean(v) for k, v in out.items() if v}

dram_means = collect_metric("DRAM")
sm_means   = collect_metric("SM")

# Only Config A N-sweep
Ns = [1, 2, 3, 4, 6, 8]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
for ax, means, title, ylabel in [
    (ax1, dram_means, "DRAM ACTIVE mean %", "DRAM utilization (%)"),
    (ax2, sm_means,   "SM ACTIVE mean %",   "SM utilization (%)"),
]:
    for mps, color in [("off", "#dc2626"), ("on", "#10b981")]:
        vals = [np.mean([means.get(f"cfgA_A_nrxN{N}_MPS{mps}_t{t}",0) for t in [1,2,3] if f"cfgA_A_nrxN{N}_MPS{mps}_t{t}" in means]) for N in Ns]
        vals = [v if v > 0 else 0 for v in vals]
        ax.plot(Ns, vals, "o-", color=color, label=f"MPS {mps}", linewidth=2.5, markersize=10)
    ax.set_xlabel("N (concurrent NRx processes)"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold"); ax.grid(alpha=0.3); ax.legend()
    ax.axvspan(6, 8.3, alpha=0.15, color="#eab308", label="MPS breakdown zone")
plt.suptitle("Figure 12. DCGM aggregate — DRAM/SM utilization as function of N processes (Config A)\n"
             "Direct evidence of HBM saturation vs SM activity",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/f12_dcgm_summary.png", dpi=140, bbox_inches='tight'); plt.close()
print("saved f12_dcgm_summary.png")

print("\nDCGM analysis done.")
