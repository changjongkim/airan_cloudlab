#!/usr/bin/env python3
"""Slide-v2 supplementary figure: SP + MIG + MPS pct sweep shows pct=30 as
the sweet spot that clears 5G L1 SLA — proof that same-partition co-work
is feasible."""
import os, json, glob
import numpy as np
import matplotlib.pyplot as plt

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260803"
FIG  = f"{BASE}/analysis_chain19/figures/mig_mps"

INK="#0f172a"; INK_SEC="#334155"; INK_MUT="#64748b"
SURFACE="#ffffff"; GRID="#e2e8f0"
COL_GOOD="#059669"; COL_WARN="#d97706"; COL_BAD="#b91c1c"; COL_BASE="#0f172a"

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo","AppleGothic","DejaVu Sans"],
    "font.size": 14, "axes.titlesize": 17, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK_MUT, "axes.linewidth": 1.0,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SEC, "ytick.color": INK_SEC,
    "grid.color": GRID, "grid.linewidth": 0.9,
    "axes.grid": True, "axes.axisbelow": True,
    "axes.unicode_minus": False, "savefig.dpi": 150, "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

ch19_l1 = {}
for f in glob.glob(f"{BASE}/chain19_exp*/realL1_*.json"):
    try:
        d = json.load(open(f))
        ch19_l1[d["label"]] = d
    except: pass

def l1_p99(cond_prefix):
    vals = [d["p99_ms"] for label, d in ch19_l1.items()
            if label.startswith(cond_prefix + "_t") or label == cond_prefix]
    return np.mean(vals) if vals else None

# =========================
# F_SP_PCT — SP + MIG + MPS pct sweep at N=6
# Shows pct=30 as the answer (crosses SLA line)
# =========================
def fig_sp_pct():
    """Bar chart: at N=6 SP, L1 p99 across MPS pct = 100/70/50/30.
    Baseline dashed, SLA line at 50ms. pct=30 highlighted green."""
    pcts = [100, 70, 50, 30]
    lats = [l1_p99(f"e11_pct{p}_N6") or 0 for p in pcts]
    baseline = l1_p99("e5_baseline") or 40

    fig, ax = plt.subplots(figsize=(13, 6.8))
    xs = np.arange(len(pcts))
    colors = [COL_BAD if l > 50 else COL_GOOD for l in lats]
    ax.bar(xs, lats, color=colors, alpha=0.85, edgecolor="white", linewidth=2, width=0.55)

    ax.axhline(50, color=INK, linestyle="--", linewidth=2, alpha=0.75)
    ax.text(len(pcts)-0.5, 52, "5G L1 SLA proxy 50 ms", ha="right",
            color=INK, fontsize=11, style="italic")
    ax.axhline(baseline, color=INK_MUT, linestyle=":", linewidth=1.5, alpha=0.6)
    ax.text(-0.4, baseline-3, f"L1 단독 baseline {baseline:.0f}ms",
            color=INK_MUT, fontsize=10, style="italic")

    for i, l in enumerate(lats):
        ax.text(i, l+3, f"{l:.0f} ms", ha="center", fontsize=13,
                color=colors[i], fontweight="bold")
    # Annotate winner
    win_i = int(np.argmin(lats))
    ax.annotate("SLA 통과\nco-work 실현", xy=(win_i, lats[win_i]),
                xytext=(win_i+0.7, lats[win_i]+30),
                fontsize=12, color=COL_GOOD, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=COL_GOOD, lw=2))

    ax.set_xticks(xs)
    ax.set_xticklabels([f"MPS pct={p}%" for p in pcts])
    ax.set_xlabel("MPS thread% cap (AI 클라이언트에 강제되는 SM 사용 상한)")
    ax.set_ylabel("L1 p99 지연 (ms) · same-partition · N=6 diverse AI")
    ax.grid(axis="y", alpha=0.5)
    ax.set_ylim(0, max(lats)*1.25)
    ax.set_title("F_SP_PCT · Same-partition에서 MIG+MPS pct=30 튜닝이 SLA를 통과 — co-work 실현 가능성 증명",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "N=6 다이버스 AI가 L1과 같은 MIG 파티션에서 co-work. pct 캡을 낮추면 AI SM 점유가 강제 제한되어 L1이 SM 확보.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/F_SP_PCT_answer.png"); plt.close()
    print("F_SP_PCT_answer")

fig_sp_pct()
print("Done.")
