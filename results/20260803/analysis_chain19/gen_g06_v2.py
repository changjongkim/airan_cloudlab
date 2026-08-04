#!/usr/bin/env python3
"""F_G06 v2 · Corrected verdict figure.

All numbers verified from Chain 17 gap stats + Chain 19 realL1 per-iter
measurements. Displayed as a horizontal bar chart with mean bars and
worst-case whiskers, grouped by verdict category.

Categories:
  ✓ verified answers  — green
  △ warning zone      — orange
  ✗ failure modes     — red

Notes:
  · L1 p99 latency (Chain 19 · realL1_*.json) is the primary SLA metric
  · gap p99 (Chain 17 · gap stats) is included for the SP + identical
    NRx row because Chain 17 did not measure per-iter L1 latency; the
    metric is different but qualitatively comparable (small kernel gaps
    = smooth L1 execution)
"""
import json, glob, re
import numpy as np
import matplotlib.pyplot as plt

BASE_17 = "/Users/changjongkim/New_research/cloudlab_results/results/20260725"
BASE_19 = "/Users/changjongkim/New_research/cloudlab_results/results/20260803"
FIG     = f"{BASE_19}/analysis_chain19/figures/mig_mps"

INK="#0f172a"; INK_SEC="#334155"; INK_MUT="#64748b"
SURFACE="#ffffff"; GRID="#e2e8f0"
COL_GOOD="#059669"; COL_WARN="#d97706"; COL_BAD="#b91c1c"; COL_BASE="#334155"

def apply_font(lang):
    if lang == "ko":
        plt.rcParams["font.family"] = ["Apple SD Gothic Neo","AppleGothic","DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = ["Helvetica Neue","Helvetica","Arial","DejaVu Sans"]

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 17, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK_MUT, "axes.linewidth": 1.0,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SEC, "ytick.color": INK_SEC,
    "grid.color": GRID, "grid.linewidth": 0.9,
    "axes.grid": True, "axes.axisbelow": True,
    "axes.unicode_minus": False, "savefig.dpi": 150, "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

# ---- Verified data collection ----
def collect_l1(dirname, prefix_exact):
    """Collect trials matching prefix exactly (using _t suffix)."""
    trials = []
    for f in glob.glob(f"{BASE_19}/{dirname}/realL1_{prefix_exact}_t*.json"):
        d = json.load(open(f))
        trials.append(d["p99_ms"])
    return trials

ch17 = json.load(open(f"{BASE_17}/chain17_all_stats.json"))
def ch17_gap(cfg, N, mps):
    vs = [ch17[k]["gap_p99"]/1e6 for k in ch17 if re.match(f"cfg{cfg}_A_nrxN{N}_MPS{mps}_t\\d+", k)]
    return vs

def make_g06(lang):
    apply_font(lang)

    baseline    = collect_l1("chain19_exp5",  "e5_baseline")
    cp6         = collect_l1("chain19_exp5",  "e5_cpN6")
    cp16        = collect_l1("chain19_exp5",  "e5_cpN16")
    full_qwen1  = collect_l1("chain19_exp1",  "e1_cfgB_diverseN1")
    full_div6   = collect_l1("chain19_exp1",  "e1_cfgB_diverseN6")
    sp_p30_n6   = collect_l1("chain19_exp11", "e11_pct30_N6")
    sp_p100_n6  = collect_l1("chain19_exp11", "e11_pct100_N6")
    sp_nrx_n8   = ch17_gap("A", 8, "on")  # gap p99 in ms, identical NRx SP

    if lang == "ko":
        rows = [
            ("Baseline · L1 단독",                         "L1 p99",  baseline,   COL_BASE, "기준"),
            ("SP + identical NRx · MPS on · N=8 †",         "gap p99", sp_nrx_n8, COL_GOOD, "✓ 검증"),
            ("CP + diverse AI · MPS on · N=6",              "L1 p99",  cp6,        COL_GOOD, "✓ 검증"),
            ("CP + diverse AI · MPS on · N=16",             "L1 p99",  cp16,       COL_GOOD, "✓ 검증"),
            ("Full GPU + Qwen only · MPS on · N=1",         "L1 p99",  full_qwen1, COL_WARN, "△ SLA 아슬"),
            ("Full GPU + diverse AI · MPS on · N=6 (bimodal)","L1 p99", full_div6, COL_WARN, "△ worst 실패"),
            ("SP-4g + diverse AI · MPS pct=30 · N=6",       "L1 p99",  sp_p30_n6,  COL_BAD,  "✗ 실패"),
            ("SP-4g + diverse AI · MPS pct=100 · N=6",      "L1 p99",  sp_p100_n6, COL_BAD,  "✗ 파국"),
        ]
        title = "F_G06 v2 · 실측 verdict — 모든 숫자는 3-trial mean · whisker는 worst-case"
        xlabel = "L1 지연 지표 (ms · 로그 스케일)"
        note = "† identical NRx SP는 Chain 17 · gap p99 (kernel 사이 tail 간격) 사용. Chain 19에서는 per-iter L1 측정 안 함. 다른 지표지만 심각도 대비는 뚜렷 — kernel gap이 작으면 L1 실행 매끄러움."
    else:
        rows = [
            ("Baseline · L1 alone",                          "L1 p99",  baseline,   COL_BASE, "reference"),
            ("SP + identical NRx · MPS on · N=8 †",          "gap p99", sp_nrx_n8, COL_GOOD, "✓ verified"),
            ("CP + diverse AI · MPS on · N=6",               "L1 p99",  cp6,        COL_GOOD, "✓ verified"),
            ("CP + diverse AI · MPS on · N=16",              "L1 p99",  cp16,       COL_GOOD, "✓ verified"),
            ("Full GPU + Qwen only · MPS on · N=1",          "L1 p99",  full_qwen1, COL_WARN, "△ near SLA"),
            ("Full GPU + diverse AI · MPS on · N=6 (bimodal)","L1 p99", full_div6, COL_WARN, "△ worst fails"),
            ("SP-4g + diverse AI · MPS pct=30 · N=6",        "L1 p99",  sp_p30_n6,  COL_BAD,  "✗ fails"),
            ("SP-4g + diverse AI · MPS pct=100 · N=6",       "L1 p99",  sp_p100_n6, COL_BAD,  "✗ catastrophic"),
        ]
        title = "F_G06 v2 · Measured verdict — all values are 3-trial mean · whisker marks worst-case"
        xlabel = "L1 latency metric (ms · log scale)"
        note = "† SP + identical NRx uses Chain 17 gap p99 (inter-kernel tail). Chain 19 did not measure per-iter L1 for this condition. Different metric but severity contrast is clear — small kernel gaps mean smooth L1 execution."

    fig, ax = plt.subplots(figsize=(15, 7.2))
    ys = np.arange(len(rows))
    means = [np.mean(r[2]) if r[2] else 0 for r in rows]
    maxes = [max(r[2]) if r[2] else 0 for r in rows]
    mins  = [min(r[2]) if r[2] else 0 for r in rows]
    colors = [r[3] for r in rows]

    ax.barh(ys, means, color=colors, alpha=0.85, edgecolor="white", linewidth=1.8, height=0.62)
    for i, (mn, mx, m) in enumerate(zip(mins, maxes, means)):
        # Whisker for range
        ax.plot([mn, mx], [ys[i], ys[i]], color=INK, linewidth=2, alpha=0.85)
        ax.plot([mn, mn], [ys[i]-0.1, ys[i]+0.1], color=INK, linewidth=2, alpha=0.85)
        ax.plot([mx, mx], [ys[i]-0.1, ys[i]+0.1], color=INK, linewidth=2, alpha=0.85)
        # Value label
        text = f"{m:.1f} / worst {mx:.1f}" if ("gap" in rows[i][1]) else f"mean {m:.1f} · worst {mx:.1f}"
        ax.text(mx*1.12, ys[i], text, va="center", fontsize=11, color=colors[i], fontweight="bold")

    # SLA reference line at 50 ms
    ax.axvline(50, color=INK, linestyle="--", linewidth=1.8, alpha=0.75)
    ax.text(50*1.05, len(rows)-0.4, "5G L1 SLA proxy 50 ms",
            fontsize=11, color=INK, style="italic")

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.5, 1200)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold", pad=16, loc="left")
    ax.grid(axis="x", alpha=0.5, which="both")

    # Verdict badge column (right of label but before bar) — draw as text after y-tick
    # Actually place next to labels using axhline positions
    for i, r in enumerate(rows):
        badge_color = r[3]
        ax.text(0.55, ys[i], f"[{r[4]}]", va="center", fontsize=10, color=badge_color,
                fontweight="bold")

    fig.text(0.02, 0.008, note, fontsize=10.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_G06_VERDICT_v2{suffix}.png"); plt.close()
    print(f"F_G06_VERDICT_v2{suffix}")

for lang in ["ko", "en"]:
    make_g06(lang)
print("Done.")
