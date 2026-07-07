#!/usr/bin/env python3
"""Chain 8 mitigation shim analysis + figure."""
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

ROOT = Path(__file__).parent.parent
C8 = ROOT / "results"
OUT = ROOT / "results"

APIS_TO_TRACK = [
    ("cudaFree_v3020",              "cudaFree"),
    ("cudaFreeAsync_v11020",        "cudaFreeAsync"),
    ("cudaMallocFromPoolAsync_v11020", "cudaMallocFromPoolAsync"),
    ("cudaMalloc_v3020",            "cudaMalloc"),
    ("cudaMemcpyAsync_v3020",       "cudaMemcpyAsync"),
    ("cudaStreamSynchronize_v3020", "cudaStreamSync"),
]

def get_api_time(db, api):
    if not db.exists(): return 0
    try:
        con = sqlite3.connect(db); cur = con.cursor()
        cur.execute("""
            SELECT SUM(r.end-r.start)/1e6 FROM CUPTI_ACTIVITY_KIND_RUNTIME r
              JOIN StringIds s ON s.id=r.nameId WHERE s.value=?
        """, (api,))
        r = cur.fetchone()
        con.close()
        return r[0] or 0 if r else 0
    except: return 0

cells_list = [4, 10, 40, 60]
variants = [
    ("A_c{c}_alone",         "Baseline\n(no coloc)"),
    ("A_c{c}_nrx_baseline",  "NRx coloc\n(no shim)"),
    ("A_c{c}_nrx_freeasync", "Option A\n(cudaFreeAsync)"),
    ("A_c{c}_nrx_pool",      "Option B\n(memPool)"),
]

# ==========================================================================
# Figure 16 — Host time breakdown by shim (cells=40)
# ==========================================================================
c = 40
fig, ax = plt.subplots(figsize=(12, 6))
labels = [lbl for _, lbl in variants]
colors_apis = {
    "cudaFree": "#dc2626",
    "cudaFreeAsync": "#ec4899",
    "cudaMallocFromPoolAsync": "#8b5cf6",
    "cudaMalloc": "#3b82f6",
    "cudaMemcpyAsync": "#f59e0b",
    "cudaStreamSync": "#10b981",
}

bottom = np.zeros(len(variants))
for api_id, api_name in APIS_TO_TRACK:
    vals = []
    for pat, _ in variants:
        f = C8 / f"{pat.format(c=c)}_L1.sqlite"
        vals.append(get_api_time(f, api_id))
    vals = np.array(vals)
    if np.any(vals > 0):
        ax.bar(labels, vals, bottom=bottom, label=api_name,
               color=colors_apis.get(api_name, '#666'), edgecolor='black', lw=0.5)
        for i, v in enumerate(vals):
            if v > 500:  # only label significant bars
                ax.text(i, bottom[i] + v/2, f'{int(v)}',
                        ha='center', va='center', fontsize=9,
                        color='white' if v > 3000 else 'black', fontweight='bold')
        bottom += vals

# Total labels on top
for i, total in enumerate(bottom):
    ax.text(i, total + 200, f'total\n{int(total)} ms',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel("Host CUDA API total time (ms, 30 s NSYS window)")
ax.set_title(f"Chain 8 — cells={c}: async APIs REPLACE cudaFree but total wait CONSERVED\n"
             "Baseline coloc: 25 s host wait.  Option A/B: still 25 s host wait — shifted to cudaMemcpyAsync")
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
fig.savefig(OUT / "fig16_chain8_sync_conservation.png")
plt.close()
print("fig16 saved")

# ==========================================================================
# Figure 17 — mitigation effectiveness across cells
# ==========================================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
colors = ['#10b981', '#dc2626', '#ec4899', '#8b5cf6']
markers = ['o', 's', 'v', 'D']

for (pat, lbl), col, mk in zip(variants, colors, markers):
    totals = []
    for c in cells_list:
        # Sum all major APIs
        total = 0
        for api_id, _ in APIS_TO_TRACK:
            total += get_api_time(C8 / f"{pat.format(c=c)}_L1.sqlite", api_id)
        totals.append(total if total > 0 else None)
    valid = [(c, t) for c, t in zip(cells_list, totals) if t]
    if valid:
        ax.plot([c for c, _ in valid], [t for _, t in valid],
                mk + '-', color=col, label=lbl.replace('\n', ' '), lw=2, markersize=10)

ax.set_xlabel("cell count"); ax.set_ylabel("Total host CUDA API time (ms)")
ax.set_title("Chain 8 — Total host wait time invariant across mitigation attempts\n"
             "cudaFreeAsync (A) + cudaMemPool (B): NO REDUCTION vs baseline coloc")
ax.legend()
plt.tight_layout()
fig.savefig(OUT / "fig17_chain8_mitigation_across_cells.png")
plt.close()
print("fig17 saved")

# ==========================================================================
# Summary table
# ==========================================================================
print("\n=== Chain 8 summary table ===")
print(f"{'variant':<25} {'c=4':>10} {'c=10':>10} {'c=40':>10} {'c=60':>10}  (host CUDA total ms)")
for pat, lbl in variants:
    row = f"{lbl.replace(chr(10), ' '):<25}"
    for c in cells_list:
        total = sum(get_api_time(C8 / f"{pat.format(c=c)}_L1.sqlite", api) for api, _ in APIS_TO_TRACK)
        row += f" {int(total):>10}"
    print(row)
