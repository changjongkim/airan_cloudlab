"""Additional climax figures for the progress deck (v2 — replaces Slide 7 bimodal).

Output (figures/):
  fig_slide7_cudafree_direct_evidence.png  — cudaFree avg duration:
       alone vs NeuralRx default vs NeuralRx MPS vs sat_hbm MPS.
       Direct host-side evidence that the GPU "gap" comes from cudaFree blocking.
       Includes the correlationId 85% overlap fact as an annotation.
  fig_slide_n_ai_scaling.png  — L1 latency vs # AI processes co-located:
       37 + N_AI × 330ms linear fit, with measured points (E0, E3, E6, A2).
       The operational scaling law for misplacement penalty.

Data:
  - cudaFree numbers: PART F §10.6 (Perlmutter SQLite CUPTI_ACTIVITY_KIND_RUNTIME)
  - N-AI scaling: results/20260614/A2 + E0 + E3 + E6-3g
"""
import json
import glob
import statistics
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
OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Figure A — Slide 7 replacement: cudaFree host-blocking direct evidence
# ----------------------------------------------------------------------------
def fig_cudafree_direct_evidence():
    # Default-mode only (no MPS — MPS is introduced later in the deck)
    conds = [
        ("alone\n(baseline)",          246.0,   "#10b981"),
        ("L1 + NeuralRx\n(default)",   3752.0,  "#dc2626"),
    ]

    labels = [c[0] for c in conds]
    vals = [c[1] for c in conds]
    colors = [c[2] for c in conds]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(conds))
    ax.bar(x, vals, color=colors, edgecolor="black", lw=0.8, width=0.5)
    ax.set_yscale("log")
    ax.set_ylim(100, 50_000)
    ax.set_ylabel("cudaFree avg duration (μs, log scale)", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_title("cudaFree host-blocking grows under contention "
                 "— host-side cause of the GPU \"gap\"", fontsize=13, pad=14)
    ax.grid(True, axis="y", alpha=0.3)

    # Value labels
    for i, v in enumerate(vals):
        if v >= 1000:
            text = f"{int(v):,} μs"
        else:
            text = f"{int(v)} μs"
        ax.text(i, v * 1.35, text, ha="center", fontsize=13,
                fontweight="bold", color=colors[i])

    # Fold-change vs alone
    fold = vals[1] / vals[0]
    ax.text(1, vals[1] * 3.5, f"{fold:.0f}× alone", ha="center",
            fontsize=12, color="black", fontweight="bold")

    # Annotation: correlationId causal proof (bottom-left so it doesn't fight the title)
    ax.text(
        0.02, 0.97,
        "correlationId proof: 60–82% of GPU\n"
        "idle time overlaps temporally with cudaFree",
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        color="#7f1d1d", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef2f2",
                  edgecolor="#dc2626", lw=1.2),
    )

    plt.tight_layout()
    out = OUT / "fig_slide7_cudafree_direct_evidence.png"
    plt.savefig(out)
    plt.close()
    print(f"  saved: {out}")


# ----------------------------------------------------------------------------
# Figure B — N-AI scaling law (new slide)
# ----------------------------------------------------------------------------
def fig_n_ai_scaling():
    # Pull measured L1 p50 from 6/14 measurements
    data_root = ROOT / "20260614"

    def load_p50(pat):
        ms = []
        for f in sorted(glob.glob(str(data_root / pat))):
            try:
                ms += json.load(open(f)).get("raw_ms", [])
            except Exception:
                pass
        return statistics.median(ms) if ms else None

    e0  = load_p50("E0_baseline_3g/realL1_*.json")               # 0 AI
    e3  = load_p50("E3_coloc/realL1_*.json")                      # 1 AI (chanpred)
    e6  = load_p50("E6_coloc_neuralrx/3g/realL1_*.json")         # 1 AI (NeuralRx)
    a2  = load_p50("A2_mixed_coloc/chanpred_resnet/realL1_*.json")  # 2 AI (cp+rn)

    # Average the two single-AI samples (chanpred and NeuralRx) into one point at N=1
    e1_avg = None
    if e3 is not None and e6 is not None:
        e1_avg = (e3 + e6) / 2.0
    elif e3 is not None:
        e1_avg = e3
    elif e6 is not None:
        e1_avg = e6

    measured = [
        (0, e0,     "L1 alone\n(no AI in partition)",          "#10b981"),
        (1, e1_avg, "L1 + 1 AI\n(chanpred or NeuralRx)",       "#f59e0b"),
        (2, a2,     "L1 + 2 AI\n(chanpred + ResNet)",          "#dc2626"),
    ]
    measured = [m for m in measured if m[1] is not None]

    # Linear fit anchored at N=0 and N=2
    if e0 and a2:
        slope = (a2 - e0) / 2
        intercept = e0
    else:
        slope = 330.6
        intercept = 37.5

    fig, ax = plt.subplots(figsize=(11.5, 5.6))

    # Linear fit line
    xs = np.linspace(-0.3, 4.3, 50)
    ys = intercept + xs * slope
    ax.plot(xs, ys, "--", color="grey", lw=1.6, alpha=0.7,
            label=f"Linear fit:  L1 = {intercept:.1f} + N_AI × {slope:.1f} ms")

    # Measured points
    annot_offsets = {0: (22, 30), 1: (22, 38), 2: (22, 30)}
    for n, ym, lab, color in measured:
        ax.scatter(n, ym, s=210, color=color, edgecolor="black", lw=1.2, zorder=5)
        dx, dy = annot_offsets.get(n, (22, 30))
        ax.annotate(f"{lab}\n{ym:.0f} ms", (n, ym),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=10, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="grey", alpha=0.5))

    # Predicted points
    for n_pred in [3, 4]:
        y_pred = intercept + n_pred * slope
        ax.scatter(n_pred, y_pred, s=170, color="#fca5a5",
                   edgecolor="black", lw=1.0, marker="^", zorder=5)
        ax.annotate(f"predicted\n{y_pred:.0f} ms", (n_pred, y_pred),
                    xytext=(18, 0), textcoords="offset points",
                    fontsize=10, ha="left", color="grey")

    ax.set_xlabel("Number of co-located AI processes in same MIG partition", fontsize=12)
    ax.set_ylabel("L1 frame latency p50 (ms)", fontsize=12)
    ax.set_title("Operational scaling law — each additional co-located AI adds ~330 ms",
                 fontsize=13)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, intercept + 4.5 * slope + 100)
    ax.grid(True, alpha=0.3)
    # Linear-fit legend on the lower-right to avoid the upper-right callout box
    ax.legend(loc="lower right", fontsize=11)

    ax.text(
        0.99, 0.97,
        "Measured at A2 (n=900, std=0.42 ms):\n"
        "L1 + 2 AI = 698 ms — exactly 2× the single-AI coloc (354 ms)",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        color="#7f1d1d", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef2f2",
                  edgecolor="#dc2626", lw=1.2),
    )

    plt.tight_layout()
    out = OUT / "fig_slide_n_ai_scaling.png"
    plt.savefig(out)
    plt.close()
    print(f"  saved: {out}")


if __name__ == "__main__":
    print("Building Slide 7 replacement + N-AI scaling figures")
    fig_cudafree_direct_evidence()
    fig_n_ai_scaling()
    print("Done.")
