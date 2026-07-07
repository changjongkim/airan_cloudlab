#!/usr/bin/env python3
"""Build 5 figures for 20260622 report.

All data from today's measurements only — no prior deck data.
"""
import sqlite3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "savefig.dpi": 140, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})

ROOT = Path(__file__).parent
S18  = ROOT / "s18_ai_nsys"
H1H2 = ROOT / "cudafree_h1h2"
OUT  = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def query_cudafree(db):
    """Return per-call cudaFree durations in microseconds."""
    if not db.exists(): return np.array([])
    con = sqlite3.connect(db); cur = con.cursor()
    cur.execute("""
        SELECT (r.end-r.start)/1000.0 AS us
          FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
          WHERE s.value='cudaFree_v3020'
    """)
    durs = np.array([r[0] for r in cur.fetchall()])
    con.close()
    return durs


def query_runtime_breakdown(db, patterns):
    """Return {api_name: (n, total_ms)} for runtime APIs matching patterns."""
    if not db.exists(): return {}
    con = sqlite3.connect(db); cur = con.cursor()
    out = {}
    for pat in patterns:
        cur.execute(f"""
            SELECT s.value, COUNT(*), SUM(r.end-r.start)/1e6
              FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
              WHERE s.value='{pat}'
              GROUP BY s.value
        """)
        rows = cur.fetchall()
        for name, n, ms in rows:
            out[name] = (n, ms or 0)
    con.close()
    return out


def query_cf_ai_pairs(l1_db, ai_db, threshold_us=0):
    """For each L1 cudaFree, compute AI kernel overlap duration in us.
    Returns list of (cf_us, ai_overlap_us) tuples."""
    if not l1_db.exists() or not ai_db.exists(): return []
    con = sqlite3.connect(l1_db); cur = con.cursor()
    cur.execute(f"ATTACH '{ai_db}' AS ai")
    cur.execute("""
        WITH cf AS (
          SELECT r.start AS s, r.end AS e, (r.end-r.start)/1000.0 AS us
            FROM main.CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN main.StringIds n ON n.id=r.nameId
            WHERE n.value='cudaFree_v3020'
        )
        SELECT cf.us,
          COALESCE((SELECT SUM(MIN(k.end, cf.e) - MAX(k.start, cf.s))/1000.0
                     FROM ai.CUPTI_ACTIVITY_KIND_KERNEL k
                     WHERE k.start < cf.e AND k.end > cf.s), 0)
        FROM cf
    """)
    pairs = cur.fetchall()
    con.close()
    return pairs


# ----------------------------------------------------------------------------
# Figure 1 — L1 vs AI cudaFree call counts (asymmetric victim)
# ----------------------------------------------------------------------------
def fig1_asymmetric_victim():
    conds = [
        ("NeuralRx\nalone (3g)",      "X1_neuralrx_alone_3g_AI.sqlite",          None),
        ("NeuralRx\ncross-part",      "X2_neuralrx_2g_L1_3g_crosspart_AI.sqlite", "X2_neuralrx_2g_L1_3g_crosspart_L1.sqlite"),
        ("NeuralRx\nsame-part coloc", "X3_neuralrx_L1_coloc_3g_AI.sqlite",       "X3_neuralrx_L1_coloc_3g_L1.sqlite"),
        ("chanpred\nalone (3g)",      "X4_chanpred_alone_3g_AI.sqlite",          None),
        ("chanpred\ncross-part",      "X5_chanpred_2g_L1_3g_crosspart_AI.sqlite", "X5_chanpred_2g_L1_3g_crosspart_L1.sqlite"),
        ("chanpred\nsame-part coloc", "X6_chanpred_L1_coloc_3g_AI.sqlite",       "X6_chanpred_L1_coloc_3g_L1.sqlite"),
    ]
    labels, ai_counts, l1_counts = [], [], []
    for lbl, ai, l1 in conds:
        labels.append(lbl)
        ai_n = len(query_cudafree(S18 / ai))
        l1_n = len(query_cudafree(S18 / l1)) if l1 else 0
        ai_counts.append(ai_n)
        l1_counts.append(l1_n)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(conds))
    w = 0.38
    bL = ax.bar(x - w/2, l1_counts, w, label="L1 process (cuPHY)",  color="#dc2626", edgecolor="black", lw=0.6)
    bA = ax.bar(x + w/2, ai_counts, w, label="AI process",          color="#3b82f6", edgecolor="black", lw=0.6)
    for bars in (bL, bA):
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x()+b.get_width()/2, h+30, f"{int(h):,}", ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("cudaFree calls (30 s NSYS window)")
    ax.set_title("L1 vs AI process cudaFree calls — L1 is the asymmetric caller\n"
                 "Same-partition coloc: L1 calls cudaFree thousands of times, AI almost never")
    ax.legend(loc='upper left'); ax.set_ylim(0, max(max(l1_counts), max(ai_counts))*1.15)
    plt.tight_layout()
    fig.savefig(OUT / "fig1_asymmetric_victim.png")
    plt.close(fig)
    print("  fig1 saved")


# ----------------------------------------------------------------------------
# Figure 2 — cudaFree per-call distribution (bimodal split)
# ----------------------------------------------------------------------------
def fig2_bimodal_distribution():
    conds = [
        ("X2  NeuralRx cross-part",  "X2_neuralrx_2g_L1_3g_crosspart_L1.sqlite", "#10b981"),
        ("X3  NeuralRx coloc",       "X3_neuralrx_L1_coloc_3g_L1.sqlite",        "#dc2626"),
        ("X5  chanpred cross-part",  "X5_chanpred_2g_L1_3g_crosspart_L1.sqlite", "#34d399"),
        ("X6  chanpred coloc",       "X6_chanpred_L1_coloc_3g_L1.sqlite",        "#f59e0b"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bins = np.logspace(0, 4, 60)
    for lbl, fn, col in conds:
        durs = query_cudafree(S18 / fn)
        if len(durs) == 0: continue
        ax.hist(durs, bins=bins, alpha=0.55, label=f"{lbl}  (N={len(durs)})", color=col, edgecolor='black', lw=0.4)
    ax.set_xscale('log')
    ax.set_xlabel("L1 cudaFree per-call duration (µs, log scale)")
    ax.set_ylabel("call count")
    ax.set_title("L1 cudaFree per-call duration distribution — bimodal split under coloc\n"
                 "Cross-partition: balanced fast+mid.  Same-partition coloc: 60-92 % shift to slow mode (>1 ms)")
    ax.axvline(100, color='gray', ls='--', alpha=0.5)
    ax.axvline(1000, color='gray', ls='--', alpha=0.5)
    ax.text(30, ax.get_ylim()[1]*0.9, "fast\n(<100 µs)", ha='center', fontsize=9, color='gray')
    ax.text(316, ax.get_ylim()[1]*0.9, "mid", ha='center', fontsize=9, color='gray')
    ax.text(3162, ax.get_ylim()[1]*0.9, "slow\n(>1 ms)", ha='center', fontsize=9, color='gray')
    ax.legend(loc='upper left', framealpha=0.95)
    plt.tight_layout()
    fig.savefig(OUT / "fig2_bimodal_distribution.png")
    plt.close(fig)
    print("  fig2 saved")


# ----------------------------------------------------------------------------
# Figure 3 — THE killer proof: cudaFree duration vs AI kernel overlap
# ----------------------------------------------------------------------------
def fig3_cross_process_sync():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (title, l1, ai, color) in zip(axes, [
        ("X3 — NeuralRx same-partition coloc",
         "X3_neuralrx_L1_coloc_3g_L1.sqlite",
         "X3_neuralrx_L1_coloc_3g_AI.sqlite", "#dc2626"),
        ("X2 — NeuralRx cross-partition (control)",
         "X2_neuralrx_2g_L1_3g_crosspart_L1.sqlite",
         "X2_neuralrx_2g_L1_3g_crosspart_AI.sqlite", "#10b981"),
    ]):
        pairs = query_cf_ai_pairs(S18 / l1, S18 / ai)
        if not pairs: continue
        cf = np.array([p[0] for p in pairs])
        ov = np.array([p[1] for p in pairs])
        ax.scatter(ov, cf, alpha=0.25, s=10, color=color, edgecolor='none')

        # 1:1 line
        mx = max(cf.max(), ov.max())
        ax.plot([0, mx], [0, mx], 'k--', alpha=0.6, label="y=x (1:1)")

        # Linear regression on slow data (>1ms) — that's where the mechanism applies
        slow = cf > 1000
        if slow.sum() > 10:
            slope, intercept = np.polyfit(ov[slow], cf[slow], 1)
            x_fit = np.linspace(0, ov[slow].max(), 100)
            ax.plot(x_fit, slope*x_fit + intercept, color='blue', lw=2,
                    label=f"slow-mode fit: y = {slope:.2f}x + {intercept:.0f}")
            # r²
            yhat = slope*ov[slow] + intercept
            ss_res = np.sum((cf[slow] - yhat)**2)
            ss_tot = np.sum((cf[slow] - cf[slow].mean())**2)
            r2 = 1 - ss_res/ss_tot
            ax.text(0.05, 0.95, f"N (slow) = {slow.sum():,}\nr² = {r2:.3f}\nslope = {slope:.2f}",
                    transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, lw=1.5),
                    fontsize=11, fontweight='bold')

        ax.set_xlabel("AI kernel overlap duration during cudaFree (µs)")
        ax.set_ylabel("L1 cudaFree duration (µs)")
        ax.set_title(title)
        ax.legend(loc='lower right', framealpha=0.95)
        ax.set_xlim(left=-100)
        ax.set_ylim(bottom=-100)

    plt.suptitle("Cross-process sync proof — L1 cudaFree waits for concurrent AI kernel\n"
                 "Same-partition: tight 1:1 correlation (r²=0.94).  Cross-partition: no correlation (AI in different MIG instance).",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "fig3_cross_process_sync_proof.png")
    plt.close(fig)
    print("  fig3 saved")


# ----------------------------------------------------------------------------
# Figure 4 — Mitigation challenge: skip cudaFree → memcpy explodes
# ----------------------------------------------------------------------------
def fig4_sync_conservation():
    apis = ["cudaFree_v3020", "cudaMemcpyAsync_v3020", "cudaMalloc_v3020",
            "cudaStreamSynchronize_v3020", "cudaEventSynchronize_v3020"]

    p5 = query_runtime_breakdown(H1H2 / "p5_callchain_L1.sqlite", apis)
    p6 = query_runtime_breakdown(H1H2 / "p6_defer_L1.sqlite", apis)

    short_names = {
        "cudaFree_v3020": "cudaFree",
        "cudaMemcpyAsync_v3020": "cudaMemcpyAsync",
        "cudaMalloc_v3020": "cudaMalloc",
        "cudaStreamSynchronize_v3020": "cudaStreamSync",
        "cudaEventSynchronize_v3020": "cudaEventSync",
    }
    labels = [short_names[a] for a in apis]
    baseline_ms = [p5.get(a, (0, 0))[1] for a in apis]
    p6_ms       = [p6.get(a, (0, 0))[1] for a in apis]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(apis))
    w = 0.4
    b1 = ax.bar(x - w/2, baseline_ms, w, label=f"Baseline (no shim)  total = {sum(baseline_ms):.0f} ms",
                color="#dc2626", edgecolor='black', lw=0.6)
    b2 = ax.bar(x + w/2, p6_ms, w, label=f"Skip cudaFree (shim)  total = {sum(p6_ms):.0f} ms",
                color="#3b82f6", edgecolor='black', lw=0.6)
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if h > 5:
                ax.text(b.get_x()+b.get_width()/2, h+150, f"{int(h):,}", ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Host CUDA API total time (ms, 30 s window)")
    ax.set_title("Mitigation challenge — Sync wait time is CONSERVED across APIs\n"
                 "Skipping cudaFree (9.2 s → 0) does not reduce host blocking — wait moves to cudaMemcpyAsync (3.7 s → 12.8 s)")
    ax.legend(loc='upper right', framealpha=0.95)

    # Annotation arrow showing the shift
    ax.annotate("", xy=(1+w/2, 12000), xytext=(0-w/2, 9200),
                arrowprops=dict(arrowstyle="->", color='purple', lw=2))
    ax.text(0.5, 13500, "9 s shifts here", color='purple', fontsize=11, fontweight='bold', ha='center')

    plt.tight_layout()
    fig.savefig(OUT / "fig4_sync_conservation.png")
    plt.close(fig)
    print("  fig4 saved")


# ----------------------------------------------------------------------------
# Figure 5 — chanpred uses NO cudaFree (production pattern)
# ----------------------------------------------------------------------------
def fig5_chanpred_no_cudafree():
    conds = [
        ("NeuralRx alone (3g)\n— uses cudaFree", "X1_neuralrx_alone_3g_AI.sqlite"),
        ("chanpred alone (3g)\n— NO cudaFree",  "X4_chanpred_alone_3g_AI.sqlite"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (title, fn) in zip(axes, conds):
        db = S18 / fn
        if not db.exists(): continue
        con = sqlite3.connect(db); cur = con.cursor()
        cur.execute("""
            SELECT s.value, COUNT(*), SUM(r.end-r.start)/1e6
              FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
              GROUP BY s.value ORDER BY COUNT(*) DESC LIMIT 8
        """)
        rows = cur.fetchall()
        con.close()
        names = [r[0].replace("_v3020","").replace("_v7000","").replace("_v10000","") for r in rows]
        counts = [r[1] for r in rows]
        colors = ["#dc2626" if "Free" in n or "Malloc" in n else "#3b82f6" for n in names]
        y = np.arange(len(names))
        bars = ax.barh(y, counts, color=colors, edgecolor='black', lw=0.5)
        for b, c in zip(bars, counts):
            ax.text(b.get_width() * 1.05, b.get_y()+b.get_height()/2,
                    f"{c:,}", va='center', fontsize=9, fontweight='bold')
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xscale('log')
        ax.set_xlabel("call count (log scale)")
        ax.set_title(title)

    plt.suptitle("AI workload memory pattern dichotomy\n"
                 "chanpred — pre-allocated memory pool, 3.4M kernels, 0 cudaFree calls  →  zero impact on L1\n"
                 "NeuralRx — per-frame allocate/free, 900+ cudaFree calls  →  triggers cross-process sync",
                 fontsize=12, y=1.05)
    plt.tight_layout()
    fig.savefig(OUT / "fig5_chanpred_no_cudafree.png")
    plt.close(fig)
    print("  fig5 saved")


if __name__ == "__main__":
    print(f"Building figures → {OUT}")
    fig1_asymmetric_victim()
    fig2_bimodal_distribution()
    fig3_cross_process_sync()
    fig4_sync_conservation()
    fig5_chanpred_no_cudafree()
    print("Done.")
