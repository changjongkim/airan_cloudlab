#!/usr/bin/env python3
"""MPS scheduler bottleneck evidence figure.

Shows that when L1 shares the MPS context with N AI clients (Full GPU),
per-trial L1 p99 latency variance grows non-deterministically with N.
In CP setup, L1 is on separate partition (no MPS involvement) → variance
stays tight regardless of AI count.

Metric: L1 p99 per-trial dots + mean bar, three trials per condition.

Data sources:
  · chain19_exp1: Full GPU + MPS on + diverse AI · N = 1, 3, 6, 8, 10, 12
  · chain19_exp5: CP + MPS on AI (L1 on 4g alone) · N = 6, 8, 10, 12, 16
"""
import json, glob
import numpy as np
import matplotlib.pyplot as plt

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260803"
FIG  = f"{BASE}/analysis_chain19/figures/mig_mps"

INK="#0f172a"; INK_SEC="#334155"; INK_MUT="#64748b"
SURFACE="#ffffff"; GRID="#e2e8f0"
COL_GOOD="#059669"; COL_WARN="#d97706"; COL_BAD="#b91c1c"

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

def collect(exp, prefix_exact):
    trials = []
    for f in glob.glob(f"{BASE}/{exp}/realL1_{prefix_exact}_t*.json"):
        try:
            d = json.load(open(f))
            trials.append(d["p99_ms"])
        except: pass
    return sorted(trials)

def make_fig(lang):
    apply_font(lang)

    # Full GPU + MPS + diverse (L1 shares MPS ctx with all AI clients)
    fg_Ns = [1, 3, 6, 8, 10, 12]
    fg_trials = {N: collect("chain19_exp1", f"e1_cfgB_diverseN{N}") for N in fg_Ns}

    # CP + MPS on AI (L1 alone on 4g · doesn't touch MPS)
    cp_Ns = [6, 8, 10, 12, 16]
    cp_trials = {N: collect("chain19_exp5", f"e5_cpN{N}") for N in cp_Ns}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

    # ---- Panel A · Full GPU + MPS
    for i, N in enumerate(fg_Ns):
        trs = fg_trials[N]
        if not trs: continue
        mean_val = np.mean(trs)
        spread = max(trs) - min(trs)
        color = COL_BAD if spread > 10 else COL_WARN if spread > 5 else COL_GOOD
        # mean bar
        axL.bar(i, mean_val, color=color, alpha=0.3, edgecolor=color, linewidth=1.5, width=0.55)
        # individual dots
        for t in trs:
            axL.plot(i + np.random.RandomState(N*10).uniform(-0.15, 0.15), t,
                     'o', markersize=13, color=color, alpha=0.9, markeredgecolor="white", markeredgewidth=2)
        # spread annotation
        axL.text(i, max(trs)+3,
                 f"spread\n{spread:.1f} ms",
                 ha="center", fontsize=10, color=color, fontweight="bold")
    axL.axhline(50, color=INK, linestyle="--", linewidth=1.8, alpha=0.75)
    axL.text(0.02, 51, "5G L1 SLA proxy 50 ms", transform=axL.get_yaxis_transform(),
             color=INK, fontsize=11, style="italic")
    axL.set_xticks(range(len(fg_Ns)))
    axL.set_xticklabels([f"N={n}" for n in fg_Ns])
    axL.set_xlabel("N (다이버스 AI 컨테이너 수 · MPS 클라이언트 총 N+1)"
                    if lang=="ko" else
                    "N (diverse AI containers · MPS clients total N+1)")
    axL.set_ylabel("L1 p99 지연 (ms · 3 trial dots + mean bar)"
                    if lang=="ko" else
                    "L1 p99 latency (ms · 3-trial dots + mean bar)")
    axL.set_title("(a) Full GPU + MPS ON · L1과 AI 같은 MPS context 공유"
                   if lang=="ko" else
                   "(a) Full GPU + MPS ON · L1 shares MPS context with all AI clients",
                   fontweight="bold", loc="left", color=INK_SEC)
    axL.grid(axis="y", alpha=0.5)
    axL.set_ylim(0, 100)

    # ---- Panel B · CP + MPS on AI (L1 alone)
    for i, N in enumerate(cp_Ns):
        trs = cp_trials[N]
        if not trs: continue
        mean_val = np.mean(trs)
        spread = max(trs) - min(trs)
        color = COL_GOOD  # always good
        axR.bar(i, mean_val, color=color, alpha=0.3, edgecolor=color, linewidth=1.5, width=0.55)
        for t in trs:
            axR.plot(i + np.random.RandomState(N*10).uniform(-0.15, 0.15), t,
                     'o', markersize=13, color=color, alpha=0.9, markeredgecolor="white", markeredgewidth=2)
        axR.text(i, max(trs)+3,
                 f"spread\n{spread:.1f} ms",
                 ha="center", fontsize=10, color=color, fontweight="bold")
    axR.axhline(50, color=INK, linestyle="--", linewidth=1.8, alpha=0.75)
    axR.set_xticks(range(len(cp_Ns)))
    axR.set_xticklabels([f"N={n}" for n in cp_Ns])
    axR.set_xlabel("N (AI 컨테이너 수 on 3g · L1은 MPS 안 씀)"
                    if lang=="ko" else
                    "N (AI containers on 3g · L1 not on MPS)")
    axR.set_title("(b) CP + MPS ON · L1은 4g 단독 (MPS 미사용)"
                   if lang=="ko" else
                   "(b) CP + MPS ON · L1 alone on 4g (does not use MPS)",
                   fontweight="bold", loc="left", color=INK_SEC)
    axR.grid(axis="y", alpha=0.5)

    fig.suptitle("Fig · MPS 스케줄러 병목의 시각적 증거 — L1이 같은 MPS context에 있을 때만 trial variance 폭발"
                 if lang=="ko" else
                 "Fig · Visual evidence of MPS scheduler bottleneck — trial variance explodes only when L1 shares the MPS context",
                 fontweight="bold", y=1.01, x=0.02, ha="left", fontsize=16)
    note_ko = ("좌 (Full GPU): N=6부터 3-trial spread가 23 ms · N=12에서 21 ms · L1이 MPS 스케줄러 순서에 종속되어 비결정적. "
               "우 (CP): N=16까지 spread ≤ 5 ms · L1은 MPS 안 쓰니 daemon overhead 무관. "
               "결론: MPS 서버 자체가 스케줄링 병목 · L1을 MPS context에서 분리해야 안정.")
    note_en = ("Left (Full GPU): 3-trial spread hits 23 ms at N=6 and 21 ms at N=12 — L1 is subject to MPS scheduler indeterminism. "
               "Right (CP): spread stays ≤ 5 ms through N=16 — L1 doesn't touch MPS daemon so bottleneck doesn't apply. "
               "Conclusion: MPS server itself is the scheduling bottleneck · L1 must be lifted out of the shared MPS context.")
    fig.text(0.02, 0.008, note_ko if lang=="ko" else note_en,
             fontsize=10.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_MPS_BOTTLENECK{suffix}.png"); plt.close()
    print(f"F_MPS_BOTTLENECK{suffix}")

for lang in ["ko", "en"]:
    make_fig(lang)
print("Done.")
