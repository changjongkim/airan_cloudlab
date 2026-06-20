"""Build two climax figures for the next progress slide deck.

Output (in figures/):
  fig_slide7_60kb_bimodal_histogram.png  — per-call 60KB memcpy duration histogram
      (alone single peak vs coloc bimodal) — Slide 7 NSYS Layer 3
  fig_slide8_memcpy_direction_breakdown.png — H2D / D2D / D2H counts
      (chip-wide PCIe/DMA queue location proof) — Slide 8 NSYS Layer 4

Data: results/20260531/nsys_sqlite_v2/*.sqlite
Reference: MIG_AIRAN_VISUAL_EVIDENCE_KR.md §16.1, §20.1
"""
import sqlite3
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

mpl.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "savefig.dpi": 140, "savefig.bbox": "tight",
    "font.family": ["DejaVu Sans"],
})

ROOT = Path(__file__).parent.parent
SQLITE = ROOT / "20260531" / "nsys_sqlite_v2"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def per_call_60kb_durations(db_pattern):
    """Aggregate 60KB memcpy per-call durations (μs) across matching SQLite files."""
    durs = []
    for f in sorted(glob.glob(str(SQLITE / db_pattern))):
        try:
            con = sqlite3.connect(f)
            cur = con.cursor()
            cur.execute("""
                SELECT (end - start) / 1000.0
                FROM CUPTI_ACTIVITY_KIND_MEMCPY
                WHERE bytes BETWEEN 50000 AND 80000
            """)
            durs.extend(r[0] for r in cur.fetchall())
            con.close()
        except Exception as e:
            print(f"  skip {f}: {e}")
    return np.array(durs)


def memcpy_direction_counts(db_pattern):
    """Total memcpy call counts per direction across matching SQLite files."""
    counts = {"H2D": 0, "D2H": 0, "D2D": 0, "Other": 0}
    # copyKind: 1=HtoD, 2=DtoH, 3=DtoD, others=HtoH/PtoP/etc
    name_map = {1: "H2D", 2: "D2H", 3: "D2D"}
    for f in sorted(glob.glob(str(SQLITE / db_pattern))):
        try:
            con = sqlite3.connect(f)
            cur = con.cursor()
            cur.execute("SELECT copyKind, COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind")
            for kind, n in cur.fetchall():
                counts[name_map.get(kind, "Other")] += n
            con.close()
        except Exception as e:
            print(f"  skip {f}: {e}")
    return counts


# ============================================================================
# Figure 1 — Slide 7: 60KB memcpy per-call bimodal histogram (Layer 3 signature)
# ============================================================================
def fig_slide7():
    print("\nBuilding Slide 7 — bimodal histogram (60KB memcpy per-call)")
    alone = per_call_60kb_durations("S5_3g_alone_run*.sqlite")
    neuralrx = per_call_60kb_durations("S7_3g_neuralrx_run*.sqlite")
    print(f"  3g alone:    n={len(alone):>5d}  median={np.median(alone):.2f}μs  max={alone.max():.2f}μs")
    print(f"  3g NeuralRx: n={len(neuralrx):>5d}  median={np.median(neuralrx):.2f}μs  max={neuralrx.max():.2f}μs")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

    # Common bin range
    bins = np.linspace(0, 25, 80)

    # Alone (single peak)
    ax1.hist(alone, bins=bins, color="#10b981", alpha=0.85, edgecolor="black", lw=0.4)
    ax1.set_title("3g L1 alone — single peak", fontsize=12)
    ax1.set_xlabel("60KB memcpy per-call duration (μs)")
    ax1.set_ylabel("call count")
    ax1.axvline(np.median(alone), color="#065f46", ls="--", lw=1.5,
                label=f"median = {np.median(alone):.1f}μs")
    ax1.legend(loc="upper right")
    ax1.set_xlim(0, 25)

    # Coloc (bimodal)
    ax2.hist(neuralrx, bins=bins, color="#dc2626", alpha=0.85, edgecolor="black", lw=0.4)
    ax2.set_title("3g L1 + NeuralRx coloc — bimodal split", fontsize=12)
    ax2.set_xlabel("60KB memcpy per-call duration (μs)")
    ax2.set_xlim(0, 25)

    # Annotate the two modes
    fast_med = 4.2
    slow_med = 14.3
    for ax in [ax1, ax2]:
        ax.axvline(fast_med, color="grey", ls=":", lw=1, alpha=0.5)
    ax2.axvline(slow_med, color="grey", ls=":", lw=1, alpha=0.5)
    ax2.annotate(f"fast mode\n~{fast_med}μs",
                 xy=(fast_med, ax2.get_ylim()[1]*0.85),
                 xytext=(fast_med-0.5, ax2.get_ylim()[1]*0.88),
                 ha="right", fontsize=10, color="#374151",
                 arrowprops=dict(arrowstyle="->", color="grey", alpha=0.7))
    ax2.annotate(f"slow mode\n~{slow_med}μs\n(queue wait)",
                 xy=(slow_med, ax2.get_ylim()[1]*0.40),
                 xytext=(slow_med+1.5, ax2.get_ylim()[1]*0.55),
                 ha="left", fontsize=10, color="#7f1d1d", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#7f1d1d", alpha=0.9, lw=1.2))

    fig.suptitle("60KB memcpy per-call duration — direct queue arbitration signature\n"
                 "(alone: gaussian noise around 4.2μs  vs  coloc: discrete binary state — fast/slow)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = OUT / "fig_slide7_60kb_bimodal_histogram.png"
    plt.savefig(out)
    plt.close()
    print(f"  saved: {out}")


# ============================================================================
# Figure 2 — Slide 8: memcpy direction breakdown (Layer 4 location proof)
# ============================================================================
def fig_slide8():
    print("\nBuilding Slide 8 — memcpy direction breakdown")

    # Aggregate across all 3g conditions (and a representative coloc condition)
    alone = memcpy_direction_counts("S5_3g_alone_run*.sqlite")
    neuralrx = memcpy_direction_counts("S7_3g_neuralrx_run*.sqlite")
    chanpred = memcpy_direction_counts("S27_3g_chanpred_run*.sqlite")
    sat_hbm = memcpy_direction_counts("S14_3g_sat_hbm_run*.sqlite")

    print(f"  3g alone:    {alone}")
    print(f"  3g NeuralRx: {neuralrx}")
    print(f"  3g chanpred: {chanpred}")
    print(f"  3g sat_hbm:  {sat_hbm}")

    # Left: stacked bar across conditions
    # Right: pie chart for one condition emphasizing H2D dominance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                    gridspec_kw={"width_ratios": [2, 1]})

    conditions = ["3g alone", "3g + NeuralRx", "3g + chanpred", "3g + sat_hbm"]
    data = [alone, neuralrx, chanpred, sat_hbm]
    directions = ["H2D", "D2H", "D2D", "Other"]
    colors = {"H2D": "#dc2626", "D2H": "#f59e0b", "D2D": "#3b82f6", "Other": "#9ca3af"}

    x = np.arange(len(conditions))
    bottoms = np.zeros(len(conditions))
    for d in directions:
        vals = np.array([c[d] for c in data])
        ax1.bar(x, vals, 0.55, bottom=bottoms, label=d, color=colors[d], edgecolor="black", lw=0.5)
        bottoms += vals

    # Annotate H2D and D2D values on bars
    for i, c in enumerate(data):
        total = sum(c.values())
        ax1.text(i, total + total*0.02, f"H2D: {c['H2D']:,}  /  D2D: {c['D2D']:,}",
                 ha="center", fontsize=9, color="#374151", fontweight="bold")

    ax1.set_xticks(x); ax1.set_xticklabels(conditions, fontsize=10)
    ax1.set_ylabel("memcpy call count (aggregated across runs)")
    ax1.set_title("memcpy direction breakdown — H2D dominant, D2D ≈ 0 across all conditions")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, max(sum(c.values()) for c in data) * 1.15)

    # Pie chart — alone condition; only label the dominant wedges, callout small ones
    sizes = [alone["H2D"], alone["D2H"], alone["D2D"] + alone["Other"]]
    total = sum(sizes)
    labels_pie = [
        f"H2D\n{alone['H2D']:,}\n({alone['H2D']/total*100:.1f}%)",
        f"D2H\n{alone['D2H']:,}\n({alone['D2H']/total*100:.1f}%)",
        "",  # combined tiny wedge — annotated separately
    ]
    colors_pie = [colors["H2D"], colors["D2H"], "#374151"]
    wedges, texts = ax2.pie(sizes, labels=labels_pie, colors=colors_pie,
                            startangle=90, wedgeprops=dict(edgecolor="black", lw=0.8),
                            textprops=dict(fontsize=10))
    # Callout for the near-zero D2D + Other slice
    ax2.annotate(f"D2D: {alone['D2D']}  ({alone['D2D']/total*100:.2f}%)\n"
                 f"Other: {alone['Other']}",
                 xy=(0.05, 0.99), xytext=(1.18, 0.99),
                 textcoords="data", ha="left", va="top",
                 fontsize=10, color="#7f1d1d", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef2f2", edgecolor="#dc2626"),
                 arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2))
    ax2.set_title("3g alone — proportional breakdown")

    fig.suptitle("Queue location proof — memcpy direction: 100% H2D, D2D ≈ 0\n"
                 "→ Queue is at chip-wide PCIe/DMA copy engine (not partition-isolated HBM controller)",
                 fontsize=12, y=1.03)
    plt.tight_layout()
    out = OUT / "fig_slide8_memcpy_direction_breakdown.png"
    plt.savefig(out)
    plt.close()
    print(f"  saved: {out}")


if __name__ == "__main__":
    fig_slide7()
    fig_slide8()
    print("\nDone.")
