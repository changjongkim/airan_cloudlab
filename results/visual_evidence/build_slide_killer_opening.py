"""Killer opening figure for the progress deck (Slide 2).

Output (figures/):
  fig_slide_8x_contrast.png  — 5 AI workloads measured at two placements:
       cross-partition (blue, ~45 ms) vs same-partition coloc (red, ~360 ms).
       One figure shows both 8× collapse and workload-invariance simultaneously.

Data sources:
  - Cross-partition values: 6/1 F_saturation + 6/14 E1/E4 (driver 550, 4-ant)
  - Same-partition coloc values: 6/1 G_coloc series (n=500/condition)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

mpl.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 10,
    "savefig.dpi": 140, "savefig.bbox": "tight",
    "font.family": ["DejaVu Sans"],
})

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def fig_8x_contrast():
    # Workload, cross-partition p99, same-partition coloc p99
    rows = [
        ("chanpred",     45.2, 361.1),
        ("NeuralRx",     43.6, 357.8),
        ("xApp",         39.4, 358.5),
        ("ResNet",       45.3, 360.9),
        ("forecaster",   45.4, 356.8),
    ]
    labels = [r[0] for r in rows]
    cross_vals = [r[1] for r in rows]
    coloc_vals = [r[2] for r in rows]

    x = np.arange(len(rows))
    width = 0.4

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    bars1 = ax.bar(x - width/2, cross_vals, width,
                   label="cross-partition (L1 on 3g, AI on 4g)",
                   color="#3b82f6", edgecolor="black", lw=0.7)
    bars2 = ax.bar(x + width/2, coloc_vals, width,
                   label="same-partition coloc (L1 + AI on 3g)",
                   color="#dc2626", edgecolor="black", lw=0.7)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("L1 p99 latency (ms)", fontsize=12)
    ax.set_ylim(0, 430)
    ax.set_title("Same workload, two placements — placement determines an 8× collapse",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.95)

    # Value labels on bars
    for bar, val in zip(bars1, cross_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 8, f"{val:.0f}",
                ha="center", fontsize=10, color="#1e3a8a", fontweight="bold")
    for bar, val in zip(bars2, coloc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 8, f"{val:.0f}",
                ha="center", fontsize=10, color="#7f1d1d", fontweight="bold")

    # Fold-change annotation between mean cross and mean coloc
    mean_cross = sum(cross_vals) / len(cross_vals)
    mean_coloc = sum(coloc_vals) / len(coloc_vals)
    fold = mean_coloc / mean_cross
    ax.text(0.99, 0.97,
            f"~{mean_cross:.0f} ms cross-partition  vs  ~{mean_coloc:.0f} ms coloc\n"
            f"= {fold:.1f}× collapse,  workload-invariant",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            color="#7f1d1d", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef2f2",
                      edgecolor="#dc2626", lw=1.2))

    plt.tight_layout()
    out = OUT / "fig_slide_8x_contrast.png"
    plt.savefig(out)
    plt.close()
    print(f"  saved: {out}")


if __name__ == "__main__":
    print("Building Slide 2 opening killer figure")
    fig_8x_contrast()
    print("Done.")
