#!/usr/bin/env python3
"""MIG-alone problem evidence figure.

Two panels showing why MIG by itself is insufficient:

Panel A · Static SM budget makes contention WORSE, not better.
  MIG 4g partition (SM 56) with L1+AI colocated is 8× worse than
  the same workload on Full GPU (SM 108) where more SMs = more headroom.
  Data:
    · SP-4g + MIG + MPS pct=100 · N=6 diverse: L1 p99 411 ms (Exp 11)
    · Full GPU + MPS · N=6 diverse:             L1 p99 54 ms (Exp 1)
    · Baseline (L1 alone):                       L1 p99 42 ms

Panel B · MIG partition boundary alone doesn't fix within-partition
  multi-process contention. Without MPS, processes on the same
  partition context-switch and L1 kernel-gap tail explodes.
  Data (Chain 17 · Config A MIG 4g SP):
    · MPS OFF gap p99 grows 5.4 → 13.5 ms as N=1→8
    · MPS ON  gap p99 stays  0.7 → 1.8  ms
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
COL_MPS_OFF="#b91c1c"; COL_MPS_ON="#059669"

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

# Load Chain 19 realL1 for panel A
def collect(exp, prefix_exact):
    trials = []
    for f in glob.glob(f"{BASE_19}/{exp}/realL1_{prefix_exact}_t*.json"):
        try:
            d = json.load(open(f))
            trials.append(d["p99_ms"])
        except: pass
    return trials

# Load Chain 17 gap stats for panel B
ch17 = json.load(open(f"{BASE_17}/chain17_all_stats.json"))
def ch17_gap(cfg, N, mps):
    return [ch17[k]["gap_p99"]/1e6 for k in ch17
            if re.match(f"cfg{cfg}_A_nrxN{N}_MPS{mps}_t\\d+", k)]

def make_fig(lang):
    apply_font(lang)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6.5),
                                     gridspec_kw={"width_ratios":[1, 1.1]})

    # ---- Panel A · SM budget paradox
    baseline = collect("chain19_exp5", "e5_baseline")
    full_gpu = collect("chain19_exp1", "e1_cfgB_diverseN6")
    sp_mig   = collect("chain19_exp11", "e11_pct100_N6")

    labels_A = (["Baseline\n(L1 단독\nno AI)", "Full GPU + MPS\n(SM 108 · MIG 없음)\nN=6 diverse AI", "MIG SP-4g + MPS\n(SM 56 · MIG on)\nN=6 diverse AI"] if lang=="ko"
                else ["Baseline\n(L1 alone\nno AI)", "Full GPU + MPS\n(SM 108 · no MIG)\nN=6 diverse AI", "MIG SP-4g + MPS\n(SM 56 · MIG on)\nN=6 diverse AI"])
    means_A = [np.mean(baseline), np.mean(full_gpu), np.mean(sp_mig)]
    maxes_A = [max(baseline), max(full_gpu), max(sp_mig)]
    mins_A  = [min(baseline), min(full_gpu), min(sp_mig)]
    colors_A = [COL_BASE, COL_WARN, COL_BAD]

    xs = np.arange(3)
    axL.bar(xs, means_A, color=colors_A, alpha=0.85, edgecolor="white", linewidth=1.8, width=0.55)
    for i, (mn, mx, m) in enumerate(zip(mins_A, maxes_A, means_A)):
        axL.plot([xs[i], xs[i]], [mn, mx], color=INK, linewidth=2, alpha=0.85)
        axL.plot([xs[i]-0.08, xs[i]+0.08], [mn, mn], color=INK, linewidth=2, alpha=0.85)
        axL.plot([xs[i]-0.08, xs[i]+0.08], [mx, mx], color=INK, linewidth=2, alpha=0.85)
        offset = m * 0.15
        axL.text(xs[i], mx + offset,
                 f"mean {m:.0f}\nworst {mx:.0f}",
                 ha="center", fontsize=11, color=colors_A[i], fontweight="bold")

    axL.axhline(50, color=INK, linestyle="--", linewidth=1.8, alpha=0.75)
    axL.text(2.4, 55, "5G L1 SLA 50 ms", ha="right",
             color=INK, fontsize=11, style="italic")

    # Annotation showing the paradox
    axL.annotate("" if True else "", xy=(2, means_A[2]*0.4), xytext=(1, means_A[1]),
                 arrowprops=dict(arrowstyle="->", color=COL_BAD, lw=2.5))
    axL.text(1.5, means_A[2]*0.55,
             ("MIG를 켰는데 · 8× 더 나빠짐" if lang=="ko" else "MIG turned on · 8× worse"),
             ha="center", fontsize=13, color=COL_BAD, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=COL_BAD, lw=1.5))

    axL.set_yscale("log")
    axL.set_ylim(20, 1500)
    axL.set_xticks(xs)
    axL.set_xticklabels(labels_A, fontsize=10.5)
    axL.set_ylabel("L1 p99 지연 (ms · 로그)" if lang=="ko" else "L1 p99 latency (ms · log)")
    axL.set_title("(a) SM 예산 역설 — 작은 파티션 = 경쟁 더 심함"
                   if lang=="ko" else
                   "(a) SM budget paradox — smaller partition = more contention",
                   fontweight="bold", loc="left", color=INK_SEC)
    axL.grid(axis="y", alpha=0.5, which="both")

    # ---- Panel B · MIG alone (MPS off) fails within partition
    Ns = [1, 2, 3, 4, 6, 8]
    off = [np.mean(ch17_gap("A", N, "off")) for N in Ns]
    on  = [np.mean(ch17_gap("A", N, "on"))  for N in Ns]

    axR.plot(Ns, off, "s-", color=COL_MPS_OFF, linewidth=3, markersize=12,
             markerfacecolor="white", markeredgewidth=2.5,
             label=("MIG on + MPS OFF (파티션 안 컨텍스트 스위칭)" if lang=="ko"
                    else "MIG on + MPS OFF (in-partition context switch)"))
    axR.plot(Ns, on,  "o-", color=COL_MPS_ON,  linewidth=3, markersize=12,
             markerfacecolor="white", markeredgewidth=2.5,
             label=("MIG on + MPS ON (참조)" if lang=="ko" else "MIG on + MPS ON (reference)"))
    for N, v in zip(Ns, off):
        axR.text(N, v+0.5, f"{v:.1f}", ha="center", fontsize=10, color=COL_MPS_OFF, fontweight="bold")
    for N, v in zip(Ns, on):
        axR.text(N, v-0.6, f"{v:.1f}", ha="center", fontsize=10, color=COL_MPS_ON, fontweight="bold")

    axR.annotate("" if True else "", xy=(8, off[-1]*0.7), xytext=(4, off[3]*0.5),
                 arrowprops=dict(arrowstyle="->", color=COL_BAD, lw=2.5))
    axR.text(6, 15,
             ("MIG 파티션 안에서도 · MPS 없으면 컨텍스트 시분할 · gap 폭발"
              if lang=="ko" else
              "Even inside a MIG partition · without MPS · time-slicing blows the gap"),
             ha="center", fontsize=11, color=COL_BAD, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=COL_BAD, lw=1.5))

    axR.set_xticks(Ns)
    axR.set_xlabel("N (같은 MIG 4g 파티션 위 동일 NRx 프로세스 수)"
                    if lang=="ko" else
                    "N (identical NRx processes on the same MIG 4g partition)")
    axR.set_ylabel("L1 kernel gap p99 (ms · Chain 17)")
    axR.set_title("(b) MIG만으로는 파티션 내부 문제 해결 못 함"
                   if lang=="ko" else
                   "(b) MIG alone does not fix within-partition scheduling",
                   fontweight="bold", loc="left", color=INK_SEC)
    axR.legend(frameon=True, loc="upper left", fontsize=11)
    axR.grid(alpha=0.5)
    axR.set_ylim(0, 20)

    # ---- Suptitle
    fig.suptitle("Fig · MIG 단독의 두 문제 — (a) 작은 파티션이 오히려 경쟁 심화 · (b) 파티션 내부 컨텍스트 시분할"
                 if lang=="ko" else
                 "Fig · Two problems with MIG alone — (a) small partition worsens contention · (b) within-partition context switching",
                 fontweight="bold", y=1.01, x=0.02, ha="left", fontsize=16)

    note_ko = ("(a) MIG 4g SP · N=6 diverse에서 L1 p99 = 411 ms · 같은 워크로드가 Full GPU에서는 54 ms. 파티션 자체가 SM 예산을 반토막내서 경쟁 심화. "
               "(b) MIG 파티션은 경계에서만 격리 · 파티션 안에서 N개 프로세스는 여전히 CUDA context 시분할 · MPS 없으면 gap p99 5→13 ms. "
               "결론: MIG는 하드웨어 경계만 제공 · 다중 프로세스 조정 도구 (MPS) 가 반드시 필요.")
    note_en = ("(a) MIG 4g SP with N=6 diverse: L1 p99 = 411 ms; the same workload on Full GPU is 54 ms. The partition halves the SM budget → contention worsens. "
               "(b) MIG isolates only at partition boundaries · within a partition, N processes still time-slice the CUDA context · without MPS, gap p99 grows 5→13 ms. "
               "Conclusion: MIG provides only the hardware boundary · a multi-process coordination tool (MPS) is still required.")
    fig.text(0.02, 0.008, note_ko if lang=="ko" else note_en,
             fontsize=10.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_MIG_PROBLEM{suffix}.png"); plt.close()
    print(f"F_MIG_PROBLEM{suffix}")

for lang in ["ko", "en"]:
    make_fig(lang)
print("Done.")
