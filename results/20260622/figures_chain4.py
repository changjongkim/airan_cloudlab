#!/usr/bin/env python3
"""Generate Chain 4 v3 figures from analyzed sqlites."""
import sqlite3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": ["DejaVu Sans"], "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "savefig.dpi": 140, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})

ROOT = Path(__file__).parent
C4   = ROOT / "chain4"
OUT  = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def get_metrics(label):
    """Returns (cudafree_ms, slow_pct, cudafree_us_array) or None if sqlite broken."""
    f = C4 / f"{label}_L1.sqlite"
    if not f.exists(): return None
    try:
        con = sqlite3.connect(f); cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_RUNTIME'")
        if not cur.fetchone(): con.close(); return None
        cur.execute("""
            SELECT (r.end-r.start)/1000.0
              FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
              WHERE s.value='cudaFree_v3020'
        """)
        durs = np.array([row[0] for row in cur.fetchall()])
        cur.execute("""
            SELECT SUM(r.end-r.start)/1e6
              FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
              WHERE s.value='cudaMemcpyAsync_v3020'
        """)
        mc_ms = cur.fetchone()[0] or 0
        con.close()
        return {
            "cf_ms": durs.sum() / 1000.0,  # μs → ms
            "slow_pct": float(np.sum(durs > 1000)) / max(len(durs), 1) * 100,
            "cf_us": durs,
            "memcpy_ms": mc_ms,
            "cf_count": len(durs),
        }
    except Exception:
        return None


# =========================================================================
# Figure A — Partition sweep: alone vs NeuralRx coloc vs chanpred coloc
# =========================================================================
parts = ["7g", "4g", "3g", "2g"]
scenarios = [("alone", "L1 alone", "#10b981"),
             ("chanpred_coloc", "+ chanpred coloc", "#f59e0b"),
             ("neuralrx_coloc", "+ NeuralRx coloc", "#dc2626")]

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(parts))
w = 0.27
for i, (sc, lbl, col) in enumerate(scenarios):
    vals = []
    for p in parts:
        m = get_metrics(f"{p}_{sc}")
        vals.append(m["cf_ms"] if m else None)
    offset = (i - 1) * w
    valid_x = [x[j] + offset for j, v in enumerate(vals) if v is not None]
    valid_v = [v for v in vals if v is not None]
    bars = ax.bar(valid_x, valid_v, w, label=lbl, color=col, edgecolor='black', lw=0.6)
    for b, v in zip(bars, valid_v):
        ax.text(b.get_x() + b.get_width() / 2, v + 100, f"{int(v)}",
                ha='center', fontsize=9, fontweight='bold')
    # mark missing
    for j, v in enumerate(vals):
        if v is None:
            ax.text(x[j] + offset, 200, "n/a", ha='center', fontsize=9, color='gray')

ax.set_xticks(x); ax.set_xticklabels(parts)
ax.set_xlabel("L1 MIG partition size")
ax.set_ylabel("L1 cudaFree total time (ms, 30 s NSYS window)")
ax.set_title("Partition size sweep — cudaFree contention dominates regardless of partition size\n"
             "NeuralRx coloc: ~9000 ms across 7g/4g/3g.  chanpred coloc: scales modestly with smaller partition.")
ax.legend(loc='upper left', framealpha=0.95)
plt.tight_layout()
fig.savefig(OUT / "fig6_partition_sweep.png")
plt.close(fig)
print("fig6 saved")


# =========================================================================
# Figure B — Cross-partition workload type variation (4g and 3g)
# =========================================================================
cross_part = ["alone", "neuralrx_coloc", "coloc_qwen", "coloc_hbm", "coloc_chanpred", "coloc_resnet"]
labels = ["L1\nalone", "+ NRx\ncoloc", "+ NRx coloc\n+ Qwen (2g)", "+ NRx coloc\n+ HBM (2g)",
          "+ NRx coloc\n+ chanpred (2g)", "+ NRx coloc\n+ ResNet (2g)"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, p in zip(axes, ["4g", "3g"]):
    vals = [(get_metrics(f"{p}_{sc}") or {}).get("cf_ms") for sc in cross_part]
    valid_x = [j for j, v in enumerate(vals) if v is not None]
    valid_v = [v for v in vals if v is not None]
    colors = ["#10b981" if "alone" in cross_part[j] else
              "#dc2626" if cross_part[j] == "neuralrx_coloc" else
              "#3b82f6" for j in valid_x]
    bars = ax.bar(valid_x, valid_v, color=colors, edgecolor='black', lw=0.6)
    for b, v in zip(bars, valid_v):
        ax.text(b.get_x() + b.get_width() / 2, v + 100, f"{int(v)}",
                ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("L1 cudaFree total (ms)" if p == "4g" else "")
    ax.set_title(f"L1 = {p}")
    ax.set_ylim(0, 11000)
    # baseline reference line
    base = vals[1]  # NRX coloc
    if base:
        ax.axhline(base, color='red', linestyle='--', alpha=0.4, lw=1)

plt.suptitle("Cross-partition workload type has zero effect on L1 cudaFree\n"
             "Adding Qwen / HBM / chanpred / ResNet in 2g sidecar leaves coloc time within ±2% of NRx-coloc baseline",
             fontsize=12, y=1.04)
plt.tight_layout()
fig.savefig(OUT / "fig7_crosspart_workload_variation.png")
plt.close(fig)
print("fig7 saved")


# =========================================================================
# Figure C — Per-call cudaFree distribution by partition (NRX coloc)
# =========================================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
bins = np.logspace(0, 4, 60)
colors = {"7g": "#10b981", "4g": "#3b82f6", "3g": "#f59e0b", "2g": "#dc2626"}
for p in parts:
    m = get_metrics(f"{p}_neuralrx_coloc")
    if m is None: continue
    durs = m["cf_us"]
    ax.hist(durs, bins=bins, alpha=0.5, label=f"{p}  (N={len(durs)}, slow%={m['slow_pct']:.1f})",
            color=colors[p], edgecolor='black', lw=0.4)
ax.set_xscale("log")
ax.set_xlabel("L1 cudaFree per-call duration (μs, log)")
ax.set_ylabel("call count")
ax.set_title("L1 cudaFree per-call distribution under NeuralRx coloc — invariant across partition sizes\n"
             "Bimodal (fast <100μs + slow >1ms) pattern identical at 7g, 4g, 3g")
ax.axvline(1000, color='gray', ls='--', alpha=0.5)
ax.text(3000, ax.get_ylim()[1] * 0.85, "slow mode\n(>1 ms)", ha='center', color='gray', fontsize=10)
ax.legend(loc='upper left', framealpha=0.95)
plt.tight_layout()
fig.savefig(OUT / "fig8_distribution_by_partition.png")
plt.close(fig)
print("fig8 saved")


# =========================================================================
# Figure D — 2g shows different pattern (interesting outlier)
# =========================================================================
twog = [(sc, get_metrics(f"2g_{sc}")) for sc in cross_part]
twog_valid = [(sc, m) for sc, m in twog if m]
labels2g = ["alone" if sc == "alone" else
            "NRx coloc" if sc == "neuralrx_coloc" else
            sc.replace("coloc_", "+") for sc, _ in twog_valid]
vals2g = [m["cf_ms"] for _, m in twog_valid]
slow2g = [m["slow_pct"] for _, m in twog_valid]

fig, ax1 = plt.subplots(figsize=(11, 5))
x2 = np.arange(len(labels2g))
b1 = ax1.bar(x2 - 0.2, vals2g, 0.4, color="#dc2626", label="cudaFree total (ms)", edgecolor='black', lw=0.6)
ax1.set_xticks(x2); ax1.set_xticklabels(labels2g, fontsize=10)
ax1.set_ylabel("cudaFree total (ms)", color="#dc2626")
ax1.tick_params(axis='y', labelcolor="#dc2626")
ax2 = ax1.twinx()
b2 = ax2.bar(x2 + 0.2, slow2g, 0.4, color="#3b82f6", label="slow% (>1ms)", edgecolor='black', lw=0.6)
ax2.set_ylabel("slow mode %", color="#3b82f6")
ax2.tick_params(axis='y', labelcolor="#3b82f6")
ax2.set_ylim(0, 100); ax2.grid(False)
ax1.set_title("L1 = 2g — coloc effect DEPENDS on workload type (unlike larger partitions)\n"
              "Small partition + SM contention reduces AI kernel duration → shorter cudaFree wait")
fig.tight_layout()
fig.savefig(OUT / "fig9_2g_anomaly.png")
plt.close(fig)
print("fig9 saved")

print("\nAll figures → ", OUT)
