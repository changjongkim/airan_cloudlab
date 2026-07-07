#!/usr/bin/env python3
"""Extract key metrics from chain4 L1 + AI sqlites.

Produces:
  - SUMMARY_chain4_local.md  — table of L1 cudaFree/memcpy/kernel + AI runtime per condition
  - figures/chain4_*.png      — partition sweep + cross-part variation figures
"""
import sqlite3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "savefig.dpi": 140, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})

ROOT = Path(__file__).parent
C4   = ROOT / "chain4"
OUT  = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def query_l1_metrics(db):
    """Return dict of L1 process metrics from one sqlite."""
    if not db.exists(): return None
    try:
        con = sqlite3.connect(db); cur = con.cursor()
        # Check if tables exist
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_RUNTIME'")
        if not cur.fetchone():
            con.close(); return None
    except Exception:
        return None
    r = {}
    apis = ["cudaFree_v3020", "cudaMemcpyAsync_v3020", "cudaMalloc_v3020",
            "cudaStreamSynchronize_v3020", "cudaEventSynchronize_v3020",
            "cudaLaunchKernel_v7000", "cuLaunchKernel"]
    for api in apis:
        try:
            cur.execute("""
                SELECT COUNT(*), SUM(r.end-r.start)/1e6
                  FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
                  WHERE s.value=?
            """, (api,))
            n, ms = cur.fetchone()
            r[api] = (n or 0, ms or 0.0)
        except Exception:
            r[api] = (0, 0.0)
    for key, tbl in [("kernel", "CUPTI_ACTIVITY_KIND_KERNEL"),
                     ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
                     ("memset", "CUPTI_ACTIVITY_KIND_MEMSET")]:
        try:
            cur.execute(f"SELECT COUNT(*), SUM(end-start)/1e6 FROM {tbl}")
            n, ms = cur.fetchone()
            r[key] = (n or 0, ms or 0.0)
        except Exception:
            r[key] = (0, 0.0)
    try:
        cur.execute("""
            SELECT (r.end-r.start)/1000.0
              FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
              WHERE s.value='cudaFree_v3020'
        """)
        r["cudafree_us"] = np.array([row[0] for row in cur.fetchall()])
    except Exception:
        r["cudafree_us"] = np.array([])
    con.close()
    return r


def query_ai_throughput(db):
    """Estimate AI process throughput from sqlite kernel + AI-specific counters."""
    if not db.exists(): return None
    try:
        con = sqlite3.connect(db); cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_KERNEL'")
        if not cur.fetchone(): con.close(); return None
        cur.execute("SELECT COUNT(*), SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL")
        k_n, k_ms = cur.fetchone()
        cur.execute("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL")
        mn, mx = cur.fetchone()
        wall_s = (mx - mn) / 1e9 if (mn and mx) else 0
        try:
            cur.execute("""
                SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
                WHERE s.value='cudaLaunchKernel_v7000' OR s.value='cuLaunchKernel'
            """)
            launches = cur.fetchone()[0]
        except Exception:
            launches = 0
        con.close()
        return {
            "kernel_count": k_n or 0, "kernel_ms": k_ms or 0.0,
            "wall_s": wall_s, "launches": launches,
            "kernel_per_sec": (k_n / wall_s) if wall_s > 0 else 0,
        }
    except Exception:
        return None


# ===========================================================================
# Build master table
# ===========================================================================
conditions = [
    # partition, scenario, label
    ("7g", "alone",            "7g_alone"),
    ("7g", "neuralrx_coloc",   "7g_neuralrx_coloc"),
    ("7g", "chanpred_coloc",   "7g_chanpred_coloc"),
    ("4g", "alone",            "4g_alone"),
    ("4g", "neuralrx_coloc",   "4g_neuralrx_coloc"),
    ("4g", "chanpred_coloc",   "4g_chanpred_coloc"),
    ("4g", "coloc_qwen",       "4g_coloc_qwen"),
    ("4g", "coloc_hbm",        "4g_coloc_hbm"),
    ("4g", "coloc_chanpred",   "4g_coloc_chanpred"),
    ("4g", "coloc_resnet",     "4g_coloc_resnet"),
    ("3g", "alone",            "3g_alone"),
    ("3g", "neuralrx_coloc",   "3g_neuralrx_coloc"),
    ("3g", "chanpred_coloc",   "3g_chanpred_coloc"),
    ("3g", "coloc_qwen",       "3g_coloc_qwen"),
    ("3g", "coloc_hbm",        "3g_coloc_hbm"),
    ("3g", "coloc_chanpred",   "3g_coloc_chanpred"),
    ("3g", "coloc_resnet",     "3g_coloc_resnet"),
    ("2g", "alone",            "2g_alone"),
    ("2g", "neuralrx_coloc",   "2g_neuralrx_coloc"),
    ("2g", "chanpred_coloc",   "2g_chanpred_coloc"),
    ("2g", "coloc_qwen",       "2g_coloc_qwen"),
    ("2g", "coloc_hbm",        "2g_coloc_hbm"),
    ("2g", "coloc_chanpred",   "2g_coloc_chanpred"),
    ("2g", "coloc_resnet",     "2g_coloc_resnet"),
]

print("Extracting L1 metrics for 24 conditions...")
results = {}
for part, sc, label in conditions:
    l1_db = C4 / f"{label}_L1.sqlite"
    metrics = query_l1_metrics(l1_db)
    results[(part, sc)] = metrics
    if metrics:
        cf_n, cf_ms = metrics["cudaFree_v3020"]
        mc_n, mc_ms = metrics["cudaMemcpyAsync_v3020"]
        k_n, k_ms = metrics["kernel"]
        cf_us = metrics["cudafree_us"]
        slow_pct = float(np.sum(cf_us > 1000)) / max(len(cf_us), 1) * 100
        print(f"  {label:30s}  cudaFree={cf_n:5d}/{cf_ms:8.1f}ms  slow%={slow_pct:5.1f}  kernel={k_ms:6.1f}ms")

# AI metrics
ai_files = list(C4.glob("*_AI_*.sqlite"))
print(f"\nExtracting AI metrics for {len(ai_files)} AI sqlites...")
ai_results = {}
for f in ai_files:
    name = f.stem
    metrics = query_ai_throughput(f)
    if metrics:
        ai_results[name] = metrics

# ===========================================================================
# Save SUMMARY markdown
# ===========================================================================
SUM = ROOT / "SUMMARY_chain4_local.md"
with open(SUM, "w") as f:
    f.write("# Chain 4 v3 — local analysis (24 conditions)\n\n")
    f.write("## L1 process metrics\n\n")
    f.write("| partition | scenario | cudaFree (n / ms) | slow% (>1ms) | memcpyAsync ms | kernel ms |\n")
    f.write("|---|---|---|---|---|---|\n")
    for part, sc, label in conditions:
        m = results[(part, sc)]
        if m is None: continue
        cf_n, cf_ms = m["cudaFree_v3020"]
        mc_ms = m["cudaMemcpyAsync_v3020"][1]
        k_ms = m["kernel"][1]
        cf_us = m["cudafree_us"]
        slow_pct = float(np.sum(cf_us > 1000)) / max(len(cf_us), 1) * 100
        f.write(f"| {part} | {sc} | {cf_n} / {cf_ms:.1f} | {slow_pct:.1f}% | {mc_ms:.1f} | {k_ms:.1f} |\n")
    f.write("\n## AI process metrics (per-process per-condition)\n\n")
    f.write("| condition | workload | kernels | wall s | kernel/s |\n")
    f.write("|---|---|---|---|---|\n")
    for name, m in sorted(ai_results.items()):
        parts = name.replace("_AI_", "|").split("|")
        cond = parts[0]
        wl = parts[1] if len(parts) > 1 else "?"
        f.write(f"| {cond} | {wl} | {m['kernel_count']} | {m['wall_s']:.1f} | {m['kernel_per_sec']:.0f} |\n")

print(f"\nSummary → {SUM}")
