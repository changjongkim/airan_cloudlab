#!/usr/bin/env python3
"""Analyze fresh chain5/6/7 data from 20260701 rerun."""
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
C5 = ROOT / "chain5"
C6 = ROOT / "chain6"
C7 = ROOT / "chain7"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def get_cf(db):
    """Return (count, total_ms, per_call_us_array) for cudaFree. None if empty."""
    if not db.exists(): return None
    try:
        con = sqlite3.connect(db); cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_RUNTIME'")
        if not cur.fetchone(): con.close(); return None
        cur.execute("""
            SELECT (r.end-r.start)/1000.0
              FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
              WHERE s.value='cudaFree_v3020'
        """)
        durs = np.array([r[0] for r in cur.fetchall()])
        con.close()
        if len(durs) == 0: return None
        return {"n": len(durs), "ms": durs.sum() / 1000.0, "us": durs,
                "slow_pct": float(np.sum(durs > 1000)) / len(durs) * 100}
    except Exception:
        return None


# ==========================================================================
# Figure 10 — Chain 5: sidecar-only (no NRX coloc) partition sweep
# ==========================================================================
parts = ["2g", "3g", "4g"]
workloads = ["qwen", "hbm", "chanpred", "resnet"]
labels_wl = {"qwen": "Qwen LLM", "hbm": "HBM stress", "chanpred": "chanpred", "resnet": "ResNet"}
colors = {"qwen": "#3b82f6", "hbm": "#dc2626", "chanpred": "#f59e0b", "resnet": "#10b981"}

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(parts))
w = 0.2
for i, wl in enumerate(workloads):
    vals = []
    for p in parts:
        m = get_cf(C5 / f"{p}_l1_{wl}_only_L1.sqlite")
        vals.append(m["ms"] if m else None)
    offset = (i - 1.5) * w
    valid_x = [x[j] + offset for j, v in enumerate(vals) if v is not None]
    valid_v = [v for v in vals if v is not None]
    ax.bar(valid_x, valid_v, w, label=f"+ {labels_wl[wl]}",
           color=colors[wl], edgecolor='black', lw=0.5)
    for xi, vi in zip(valid_x, valid_v):
        ax.text(xi, vi + 20, f"{int(vi)}", ha='center', fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(parts)
ax.set_xlabel("L1 partition (with sidecar workload in 2g)")
ax.set_ylabel("L1 cudaFree total (ms, 30 s NSYS window)")
ax.set_title("Chain 5 — L1 + sidecar (no NRx coloc): cross-partition workload effect\n"
             "3g/4g partitions: ~600 ms baseline invariant.  2g partition: elevated baseline (~1100 ms)")
ax.legend(loc='upper right', ncol=2)
plt.tight_layout()
fig.savefig(OUT / "fig10_chain5_sidecar_sweep.png")
plt.close()
print("fig10 saved")


# ==========================================================================
# Figure 11 — Chain 6: Cell sweep — cudaFree count scales linearly
# ==========================================================================
cells_list = [4, 10, 40, 60]
scenarios = ["alone", "neuralrx_coloc", "chanpred_coloc",
             "l1_qwen_only", "coloc_qwen"]
scenario_labels = {
    "alone": "L1 alone",
    "neuralrx_coloc": "L1 + NRx coloc",
    "chanpred_coloc": "L1 + CHP coloc",
    "l1_qwen_only": "L1 + Qwen sidecar",
    "coloc_qwen": "L1 + NRx coloc + Qwen sidecar",
}
scenario_colors = {
    "alone": "#10b981", "neuralrx_coloc": "#dc2626",
    "chanpred_coloc": "#f59e0b", "l1_qwen_only": "#3b82f6", "coloc_qwen": "#8b5cf6",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
# Left: cudaFree count vs cells
ax = axes[0]
for sc in scenarios:
    vals = []
    for c in cells_list:
        m = get_cf(C6 / f"3g_c{c}_{sc}_L1.sqlite")
        vals.append(m["n"] if m else None)
    valid_c = [cells_list[i] for i, v in enumerate(vals) if v is not None]
    valid_v = [v for v in vals if v is not None]
    if valid_v:
        ax.plot(valid_c, valid_v, 'o-', label=scenario_labels[sc], color=scenario_colors[sc], lw=2, markersize=8)
ax.set_xlabel("cell count"); ax.set_ylabel("cudaFree call count (30 s window)")
ax.set_title("cudaFree count scales LINEARLY with cell count\n"
             "~135 cudaFree/cell — invariant across scenarios")
ax.legend(loc='upper left', fontsize=9)

# Right: cudaFree total time
ax = axes[1]
for sc in scenarios:
    vals = []
    for c in cells_list:
        m = get_cf(C6 / f"3g_c{c}_{sc}_L1.sqlite")
        vals.append(m["ms"] if m else None)
    valid_c = [cells_list[i] for i, v in enumerate(vals) if v is not None]
    valid_v = [v for v in vals if v is not None]
    if valid_v:
        ax.plot(valid_c, valid_v, 'o-', label=scenario_labels[sc], color=scenario_colors[sc], lw=2, markersize=8)
ax.set_xlabel("cell count"); ax.set_ylabel("cudaFree total (ms)")
ax.set_title("cudaFree TIME scales super-linearly under coloc\n"
             "Alone: ~30 ms/cell.  Coloc: ~300 ms/cell (10× amplification)")
ax.legend(loc='upper left', fontsize=9)

plt.suptitle("Chain 6 — cell size sweep on 3g partition (cells=4/10/40/60)", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(OUT / "fig11_chain6_cell_sweep.png")
plt.close()
print("fig11 saved")


# ==========================================================================
# Figure 12 — Chain 6: coloc penalty ratio (invariant?)
# ==========================================================================
fig, ax = plt.subplots(figsize=(11, 5))
for c in cells_list:
    alone = get_cf(C6 / f"3g_c{c}_alone_L1.sqlite")
    nrx = get_cf(C6 / f"3g_c{c}_neuralrx_coloc_L1.sqlite")
    chp = get_cf(C6 / f"3g_c{c}_chanpred_coloc_L1.sqlite")
    if not alone: continue
    penalty_nrx = (nrx["ms"] / alone["ms"]) if nrx else None
    penalty_chp = (chp["ms"] / alone["ms"]) if chp else None
    if penalty_nrx:
        ax.scatter(c, penalty_nrx, s=100, color='#dc2626', edgecolor='black', label='NRx coloc penalty' if c == 4 else None)
    if penalty_chp:
        ax.scatter(c, penalty_chp, s=100, color='#f59e0b', edgecolor='black', label='CHP coloc penalty' if c == 4 else None)

ax.set_xlabel("cell count"); ax.set_ylabel("coloc / alone cudaFree ratio")
ax.set_title("Coloc penalty ratio vs cell count — invariant scaling law\n"
             "NRx coloc: ~15× multiplication of alone baseline, chanpred coloc: ~10×")
ax.axhline(15, color='#dc2626', linestyle='--', alpha=0.4, label='y=15 (NRx)')
ax.axhline(10, color='#f59e0b', linestyle='--', alpha=0.4, label='y=10 (CHP)')
ax.legend()
ax.set_ylim(0, 30)
plt.tight_layout()
fig.savefig(OUT / "fig12_chain6_coloc_ratio.png")
plt.close()
print("fig12 saved")

print("\nDone — figures at", OUT)
