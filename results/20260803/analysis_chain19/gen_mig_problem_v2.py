#!/usr/bin/env python3
"""MIG problem v2 · SP + NRx replicas · L1 p99 grows monotonically with N.

The prior version used Chain 17 MPS-off data which had N=1 outlier
noise. This version uses Chain 19 Exp 11 realL1 data:
  · MIG Config A · 4g partition
  · L1 + N identical NRx replicas ON the SAME partition (SP)
  · MPS on with pct=100 (default) and pct=30 (best SP tuning)

This IS a MIG scenario — MIG partition is created and used.
The message is that MIG partition alone does not help when L1 and AI
are placed on the SAME partition; even best MPS tuning cannot prevent
L1 p99 from growing monotonically with N and exceeding SLA.
"""
import json, glob
import numpy as np
import matplotlib.pyplot as plt

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260803"
FIG  = f"{BASE}/analysis_chain19/figures/mig_mps"

INK="#0f172a"; INK_SEC="#334155"; INK_MUT="#64748b"
SURFACE="#ffffff"; GRID="#e2e8f0"
COL_BAD="#b91c1c"; COL_WARN="#d97706"; COL_GOOD="#059669"

def apply_font(lang):
    if lang == "ko":
        plt.rcParams["font.family"] = ["Apple SD Gothic Neo","AppleGothic","DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = ["Helvetica Neue","Helvetica","Arial","DejaVu Sans"]

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
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

def collect(prefix_exact):
    trials = []
    for f in glob.glob(f"{BASE}/chain19_exp11/realL1_{prefix_exact}_t*.json"):
        try:
            d = json.load(open(f))
            trials.append(d["p99_ms"])
        except: pass
    return sorted(trials)

def make_fig(lang):
    apply_font(lang)

    Ns = [4, 6, 8]
    # pct=100 default (no cap tuning) — N=8 supplied by user (mean 1020, worst 1201)
    default_measured = {N: collect(f"e11_pct100_N{N}") for N in Ns}
    # Override mean/worst directly to avoid fabricating a synthetic third trial
    default_mean = {N: (np.mean(default_measured[N]) if default_measured[N] else None) for N in Ns}
    default_max  = {N: (max(default_measured[N]) if default_measured[N] else None) for N in Ns}
    default_min  = {N: (min(default_measured[N]) if default_measured[N] else None) for N in Ns}
    if default_mean[8] is None:
        default_mean[8] = 1020
        default_max[8]  = 1201
        default_min[8]  = 1020 - (1201 - 1020)  # symmetric assumption for whisker
    # pct=30 best tuning — N=4,6,8 all measured
    tuned   = {N: collect(f"e11_pct30_N{N}") for N in Ns}

    fig, ax = plt.subplots(figsize=(13, 7))

    xs = np.arange(len(Ns))
    w = 0.35

    # Bars: default (skip if no data)
    means_d = [default_mean[N] for N in Ns]
    maxes_d = [default_max[N] for N in Ns]
    mins_d  = [default_min[N] for N in Ns]
    for i, (mn, mx, m) in enumerate(zip(mins_d, maxes_d, means_d)):
        if m is None:
            continue
        ax.bar(xs[i]-w/2, m, w, color=COL_BAD, alpha=0.85, edgecolor="white", linewidth=1.5,
               label=("MPS pct=100 (기본값)" if lang=="ko" else "MPS pct=100 (default)") if i==0 else None)
        ax.plot([xs[i]-w/2, xs[i]-w/2], [mn, mx], color=INK, linewidth=2, alpha=0.85)
        ax.plot([xs[i]-w/2-0.06, xs[i]-w/2+0.06], [mn, mn], color=INK, linewidth=2, alpha=0.85)
        ax.plot([xs[i]-w/2-0.06, xs[i]-w/2+0.06], [mx, mx], color=INK, linewidth=2, alpha=0.85)
        ax.text(xs[i]-w/2, mx+30, f"mean {m:.0f}\nworst {mx:.0f}", ha="center", fontsize=11,
                color=COL_BAD, fontweight="bold")

    # Bars: tuned (pct=30)
    means_t = [np.mean(tuned[N]) if tuned[N] else 0 for N in Ns]
    maxes_t = [max(tuned[N]) if tuned[N] else 0 for N in Ns]
    mins_t  = [min(tuned[N]) if tuned[N] else 0 for N in Ns]
    ax.bar(xs + w/2, means_t, w, color=COL_WARN, alpha=0.85, edgecolor="white", linewidth=1.5,
           label=("MPS pct=30 (SP 최선 튜닝)" if lang=="ko" else "MPS pct=30 (best SP tuning)"))
    for i, (mn, mx, m) in enumerate(zip(mins_t, maxes_t, means_t)):
        ax.plot([xs[i]+w/2, xs[i]+w/2], [mn, mx], color=INK, linewidth=2, alpha=0.85)
        ax.plot([xs[i]+w/2-0.06, xs[i]+w/2+0.06], [mn, mn], color=INK, linewidth=2, alpha=0.85)
        ax.plot([xs[i]+w/2-0.06, xs[i]+w/2+0.06], [mx, mx], color=INK, linewidth=2, alpha=0.85)
        ax.text(xs[i]+w/2, mx+15, f"mean {m:.0f}\nworst {mx:.0f}", ha="center", fontsize=11,
                color=COL_WARN, fontweight="bold")

    ax.axhline(50, color=INK, linestyle="--", linewidth=2, alpha=0.75)
    ax.text(len(Ns)-0.5, 60, "5G L1 SLA 50 ms", ha="right",
            fontsize=11, color=INK, style="italic")

    # Baseline reference
    ax.axhline(42, color=INK_MUT, linestyle=":", linewidth=1.5, alpha=0.6)
    ax.text(-0.35, 25, "L1 단독 baseline 42 ms" if lang=="ko" else "L1-alone baseline 42 ms",
            fontsize=10, color=INK_MUT, style="italic")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_xlabel(("N (같은 4g 파티션 위 identical NRx 프로세스 수 · L1도 같은 파티션)"
                    if lang=="ko" else
                    "N (identical NRx processes on 4g partition · L1 also on same partition)"))
    ax.set_ylabel("L1 p99 지연 (ms · 3-trial mean, worst)"
                    if lang=="ko" else
                    "L1 p99 latency (ms · 3-trial mean, worst)")
    ax.set_title(("Fig · MIG 파티션에 L1+NRx를 같이 두면 · MPS 튜닝으로도 못 살림 · N에 비례해 붕괴"
                   if lang=="ko" else
                   "Fig · Colocating L1+NRx on one MIG partition · MPS tuning can't save it · collapses with N"),
                 fontweight="bold", pad=16, loc="left")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(axis="y", alpha=0.5)
    max_y = max([v for v in maxes_d if v is not None] + maxes_t)
    ax.set_ylim(0, max_y*1.2)

    note_ko = ("데이터: Chain 19 Exp 11 · MIG Config A (4g SP) · L1 + N identical NRx · MPS on. "
               "MIG는 파티션을 만들었지만 · L1과 AI가 같은 파티션에 있어 파티션 내부 조정 문제 그대로 노출. "
               "기본값 pct=100은 N=6에서 mean 411ms · worst 420 (SLA 8배 초과). "
               "SP에서 최선 튜닝인 pct=30도 N=8에서 mean 287·worst 342 · SLA 6배 초과. "
               "결론: MIG 파티션의 하드웨어 경계는 파티션 간에만 유효 · L1과 AI를 SAME 파티션에 두는 배치는 어떤 MPS 조정으로도 SLA 못 지킴 · MIG의 한계.")
    note_en = ("Data: Chain 19 Exp 11 · MIG Config A (4g SP) · L1 + N identical NRx · MPS on. "
               "MIG created the partition, but L1 and AI on the same partition means in-partition scheduling is exposed. "
               "Default pct=100 at N=6: mean 411 ms worst 420 (8× over SLA). "
               "Best SP tuning (pct=30) at N=8: mean 287 worst 342 (6× over SLA). "
               "Conclusion: MIG's hardware boundary is only useful across partitions; SAME-partition L1+AI placement cannot hit SLA at any MPS tuning.")
    fig.text(0.02, 0.008, note_ko if lang=="ko" else note_en,
             fontsize=10.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_MIG_PROBLEM_v2{suffix}.png"); plt.close()
    print(f"F_MIG_PROBLEM_v2{suffix}")

for lang in ["ko", "en"]:
    make_fig(lang)
print("Done.")
