#!/usr/bin/env python3
"""MPS scheduler bottleneck v2 · SP + identical NRx (no workload confound).

Prior v1 used Full GPU + diverse AI on the left panel. That data has an
N-composition confound (N=1 is only Qwen · N=6+ adds lighter workloads)
so the L1 latency curve doesn't monotonically grow with N. Confusing.

v2 replaces left panel with SP-4g + N identical NRx replicas + MPS on
(Chain 19 Exp 11 pct=30 · best SP tuning setting). Now the workload
composition is IDENTICAL across N values — only the client count grows.
L1 latency and per-trial spread both grow monotonically with N,
showing MPS scheduler coordination overhead as designed.

Panels:
  (a) SP + N identical NRx + MPS on · L1 p99 grows with N
      (Chain 19 Exp 11 · pct=30 · N=4, 6, 8)
  (b) CP + MPS on AI · L1 alone on 4g · L1 p99 flat with N
      (Chain 19 Exp 5 · N=6, 8, 10, 12, 16 · L1 doesn't touch MPS)
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

    # Left: SP + identical NRx + MPS pct=30 (Chain 19 Exp 11)
    sp_Ns = [4, 6, 8]
    sp_trials = {N: collect("chain19_exp11", f"e11_pct30_N{N}") for N in sp_Ns}

    # Right: CP + MPS on AI (L1 alone on 4g)
    cp_Ns = [6, 8, 10, 12, 16]
    cp_trials = {N: collect("chain19_exp5", f"e5_cpN{N}") for N in cp_Ns}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.5))

    # ---- Panel A · SP + NRx replicas
    for i, N in enumerate(sp_Ns):
        trs = sp_trials[N]
        if not trs: continue
        mean_val = np.mean(trs)
        spread = max(trs) - min(trs)
        color = COL_BAD
        axL.bar(i, mean_val, color=color, alpha=0.28, edgecolor=color, linewidth=1.5, width=0.55)
        for t in trs:
            axL.plot(i + np.random.RandomState(N*7).uniform(-0.13, 0.13), t,
                     'o', markersize=13, color=color, alpha=0.9,
                     markeredgecolor="white", markeredgewidth=2)
        axL.text(i, max(trs)+15,
                 f"mean {mean_val:.0f}\nspread {spread:.0f}",
                 ha="center", fontsize=11, color=color, fontweight="bold")
    axL.axhline(50, color=INK, linestyle="--", linewidth=1.8, alpha=0.75)
    axL.text(0.02, 55, "5G L1 SLA 50 ms", transform=axL.get_yaxis_transform(),
             color=INK, fontsize=11, style="italic")
    axL.set_xticks(range(len(sp_Ns)))
    axL.set_xticklabels([f"N={n}" for n in sp_Ns])
    axL.set_xlabel(("N (같은 4g 파티션 위 동일 NRx 프로세스 · MPS pct=30)"
                    if lang=="ko" else
                    "N (identical NRx processes on 4g partition · MPS pct=30)"))
    axL.set_ylabel("L1 p99 지연 (ms · 3-trial dots + mean bar)"
                    if lang=="ko" else
                    "L1 p99 latency (ms · 3-trial dots + mean bar)")
    axL.set_title(("(a) SP + MPS on · N 증가 = mean · spread 모두 증가"
                   if lang=="ko" else
                   "(a) SP + MPS on · both mean and spread grow with N"),
                   fontweight="bold", loc="left", color=INK_SEC)
    axL.grid(axis="y", alpha=0.5)
    axL.set_ylim(0, 400)

    # Growth arrow annotation
    axL.annotate("", xy=(2.15, 300), xytext=(-0.15, 90),
                 arrowprops=dict(arrowstyle="->", color=COL_BAD, lw=3))
    axL.text(1.0, 350,
             ("N 4→8 · mean 73→287 · spread 3→88 ms\n"
              "MPS 다중화 오버헤드가 N에 비례해 증가"
              if lang=="ko" else
              "N 4→8 · mean 73→287 · spread 3→88 ms\n"
              "MPS coordination overhead grows with N"),
             ha="center", fontsize=11, color=COL_BAD, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=COL_BAD, lw=1.5))

    # ---- Panel B · CP + MPS on AI (L1 alone on 4g)
    for i, N in enumerate(cp_Ns):
        trs = cp_trials[N]
        if not trs: continue
        mean_val = np.mean(trs)
        spread = max(trs) - min(trs)
        color = COL_GOOD
        axR.bar(i, mean_val, color=color, alpha=0.28, edgecolor=color, linewidth=1.5, width=0.55)
        for t in trs:
            axR.plot(i + np.random.RandomState(N*7).uniform(-0.13, 0.13), t,
                     'o', markersize=13, color=color, alpha=0.9,
                     markeredgecolor="white", markeredgewidth=2)
        axR.text(i, max(trs)+3,
                 f"mean {mean_val:.0f}\nspread {spread:.1f}",
                 ha="center", fontsize=11, color=color, fontweight="bold")
    axR.axhline(50, color=INK, linestyle="--", linewidth=1.8, alpha=0.75)
    axR.text(0.02, 51, "5G L1 SLA 50 ms", transform=axR.get_yaxis_transform(),
             color=INK, fontsize=11, style="italic")
    axR.set_xticks(range(len(cp_Ns)))
    axR.set_xticklabels([f"N={n}" for n in cp_Ns])
    axR.set_xlabel(("N (AI 컨테이너 수 on 3g · L1은 4g 단독 · MPS 미사용)"
                    if lang=="ko" else
                    "N (AI containers on 3g · L1 alone on 4g · does not touch MPS)"))
    axR.set_title(("(b) CP + MPS on AI · L1은 MPS 밖 · N=16까지 평평"
                   if lang=="ko" else
                   "(b) CP + MPS on AI · L1 outside MPS · flat through N=16"),
                   fontweight="bold", loc="left", color=INK_SEC)
    axR.grid(axis="y", alpha=0.5)
    axR.set_ylim(0, 100)

    fig.suptitle(("Fig · MPS 다중화 오버헤드는 N에 비례 · L1이 MPS context에 있으면 그대로 노출 · MIG로 밖에 두면 회피"
                 if lang=="ko" else
                 "Fig · MPS coordination overhead scales with N · L1 gets hit when inside the MPS context · MIG lifts L1 out to avoid it"),
                 fontweight="bold", y=1.01, x=0.02, ha="left", fontsize=15.5)

    note_ko = ("데이터 · 좌: Chain 19 Exp 11 · SP-4g · L1 + N identical NRx · MPS pct=30 (SP 최선 튜닝). "
               "우: Chain 19 Exp 5 · CP · L1@4g 단독 + N diverse AI@3g + MPS on AI. "
               "좌는 같은 워크로드 (NRx) 만 · N만 변수 · MPS 다중화 오버헤드가 N에 순수하게 비례하는 게 보임. "
               "우는 L1이 MPS 밖에 있으니 AI 파티션의 MPS 부하가 L1에 전파 안 됨.")
    note_en = ("Data · left: Chain 19 Exp 11 · SP-4g · L1 + N identical NRx · MPS pct=30 (best SP tuning). "
               "Right: Chain 19 Exp 5 · CP · L1@4g alone + N diverse AI@3g + MPS on AI. "
               "Left holds workload constant (identical NRx) — only N varies — so MPS coordination overhead scales purely with N. "
               "Right lifts L1 out of MPS · AI-side MPS pressure does not propagate to L1.")
    fig.text(0.02, 0.008, note_ko if lang=="ko" else note_en,
             fontsize=10.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_MPS_BOTTLENECK_v2{suffix}.png"); plt.close()
    print(f"F_MPS_BOTTLENECK_v2{suffix}")

for lang in ["ko", "en"]:
    make_fig(lang)
print("Done.")
