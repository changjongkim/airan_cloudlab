#!/usr/bin/env python3
"""MIG-alone problem · context-switch angle · no MPS or Full GPU comparison.

Shows MIG's pure limitation: partition creates a HARDWARE boundary at the
partition edge, but WITHIN a partition multiple processes still each hold
their own CUDA context. These contexts time-slice the partition's SMs
→ each additional process on the same MIG partition costs L1 latency
proportional to context-switch overhead.

Metric: L1 kernel gap p99 (tail interval between L1 kernels) grows with
N processes co-located on the same MIG partition. Also L1 launch rate
drops because L1 waits for its context turn.

Panels:
  (a) gap p99 vs N on MIG 4g (SM 56) and MIG 3g (SM 42) — both grow
      with N, smaller partition = worse
  (b) L1 launch rate vs N on same partitions — throughput collapse
      shows L1 kernels not getting scheduled on time
"""
import json, re
import numpy as np
import matplotlib.pyplot as plt

BASE_17 = "/Users/changjongkim/New_research/cloudlab_results/results/20260725"
BASE_19 = "/Users/changjongkim/New_research/cloudlab_results/results/20260803"
FIG     = f"{BASE_19}/analysis_chain19/figures/mig_mps"

INK="#0f172a"; INK_SEC="#334155"; INK_MUT="#64748b"
SURFACE="#ffffff"; GRID="#e2e8f0"
COL_4g="#7c3aed"; COL_3g="#0284c7"; COL_BAD="#b91c1c"

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

# Load Chain 17 stats — only MPS off data (pure MIG condition)
ch17 = json.load(open(f"{BASE_17}/chain17_all_stats.json"))
def gap_trials(cfg, N):
    return [ch17[k]["gap_p99"]/1e6 for k in ch17
            if re.match(f"cfg{cfg}_A_nrxN{N}_MPSoff_t\\d+", k)]
def launch_trials(cfg, N):
    return [ch17[k]["launch_rate"]/1000 for k in ch17
            if re.match(f"cfg{cfg}_A_nrxN{N}_MPSoff_t\\d+", k)]

def make_fig(lang):
    apply_font(lang)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6.5))

    Ns = [1, 2, 3, 4, 6, 8]

    # ---- Panel A · gap p99 grows with N (context switch tail)
    for cfg, name, color in [("A", "MIG 4g (SM 56)", COL_4g), ("C", "MIG 3g (SM 42)", COL_3g)]:
        means = [np.mean(gap_trials(cfg, N)) for N in Ns]
        maxes = [max(gap_trials(cfg, N)) for N in Ns]
        mins  = [min(gap_trials(cfg, N)) for N in Ns]
        axL.plot(Ns, means, "o-", color=color, linewidth=3, markersize=12,
                 markerfacecolor="white", markeredgewidth=2.5, label=name)
        axL.fill_between(Ns, mins, maxes, color=color, alpha=0.12)
        for N, v in zip(Ns, means):
            axL.text(N, v+0.6, f"{v:.1f}", ha="center", fontsize=9.5,
                     color=color, fontweight="bold")

    axL.set_xticks(Ns)
    axL.set_xlabel(("N (같은 MIG 파티션 위 동일 프로세스 수)"
                    if lang=="ko" else
                    "N (identical processes on the same MIG partition)"))
    axL.set_ylabel("L1 kernel gap p99 (ms)")
    axL.set_title("(a) 컨텍스트 스위칭 오버헤드 · gap tail이 N에 따라 상승"
                   if lang=="ko" else
                   "(a) Context-switch overhead · gap tail grows with N",
                   fontweight="bold", loc="left", color=INK_SEC)
    axL.legend(frameon=True, loc="upper left")
    axL.grid(alpha=0.5)
    axL.set_ylim(0, 20)

    # Annotation on Panel A
    axL.annotate(("N 증가 = 컨텍스트 스위치 증가 = L1 wait 시간 증가"
                  if lang=="ko" else
                  "More N = more context switches = longer L1 waits"),
                 xy=(6.5, 13), xytext=(3, 17),
                 fontsize=11, color=COL_BAD, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=COL_BAD, lw=1.5),
                 arrowprops=dict(arrowstyle="->", color=COL_BAD, lw=2))

    # ---- Panel B · launch rate drops with N
    for cfg, name, color in [("A", "MIG 4g (SM 56)", COL_4g), ("C", "MIG 3g (SM 42)", COL_3g)]:
        means = [np.mean(launch_trials(cfg, N)) for N in Ns]
        axR.plot(Ns, means, "o-", color=color, linewidth=3, markersize=12,
                 markerfacecolor="white", markeredgewidth=2.5, label=name)
        for N, v in zip(Ns, means):
            axR.text(N, v+0.4, f"{v:.1f}k", ha="center", fontsize=9.5,
                     color=color, fontweight="bold")

    axR.set_xticks(Ns)
    axR.set_xlabel(("N (같은 MIG 파티션 위 동일 프로세스 수)"
                    if lang=="ko" else
                    "N (identical processes on the same MIG partition)"))
    axR.set_ylabel("L1 launch rate (k kernels/s)")
    axR.set_title("(b) L1 발사 속도 붕괴 · 자기 차례 대기하느라 처리량 급감"
                   if lang=="ko" else
                   "(b) L1 launch-rate collapse · waits for context turn, throughput drops",
                   fontweight="bold", loc="left", color=INK_SEC)
    axR.legend(frameon=True, loc="upper right")
    axR.grid(alpha=0.5)
    axR.set_ylim(0, 14)

    axR.annotate(("MIG는 파티션 경계에서만 격리 · 파티션 안에서는 N개 CUDA context가 시분할"
                  if lang=="ko" else
                  "MIG isolates only at partition edge · inside, N contexts time-slice"),
                 xy=(6.5, 2), xytext=(2, 12),
                 fontsize=11, color=COL_BAD, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=COL_BAD, lw=1.5),
                 arrowprops=dict(arrowstyle="->", color=COL_BAD, lw=2))

    fig.suptitle(("Fig · MIG 단독의 근본 문제 (context-switch 관점) — 파티션 안에서 N개 프로세스는 여전히 CUDA context 시분할"
                 if lang=="ko" else
                 "Fig · MIG's fundamental problem (context-switch angle) — N processes still time-slice the CUDA context inside a partition"),
                 fontweight="bold", y=1.01, x=0.02, ha="left", fontsize=16)

    note_ko = ("데이터: Chain 17 · Config A(4g) / Config C(3g) · N개 동일 NRx 프로세스가 하나의 MIG 파티션에 배치 · MPS 없음. "
               "(a) N=1에선 gap 5ms 대 · N=8에선 4g 13ms · 3g 17ms · 파티션 안 CUDA context 시분할이 L1 kernel을 뒤로 밀어냄. "
               "(b) L1 launch rate가 N=1의 12k에서 N=8에서 1k로 붕괴 · L1이 자기 컨텍스트 차례 오기를 기다리는 시간이 지배적. "
               "결론: MIG의 하드웨어 파티션 경계는 파티션 간 격리만 제공 · 파티션 내부의 다중 프로세스 조정 부재 · N개 프로세스가 자연스레 시분할되면서 L1 SLA 파괴.")
    note_en = ("Data: Chain 17 · Config A(4g) / Config C(3g) · N identical NRx processes on one MIG partition · no MPS. "
               "(a) N=1 gives gap ~5ms; N=8 gives 13ms on 4g, 17ms on 3g — in-partition context switching pushes L1 kernels back. "
               "(b) L1 launch rate collapses from 12k (N=1) to ~1k (N=8) — L1 spends most of its time waiting for its context turn. "
               "Takeaway: MIG's hardware partition boundary provides only inter-partition isolation · no in-partition coordination · N processes naturally time-slice and destroy L1 SLA.")
    fig.text(0.02, 0.008, note_ko if lang=="ko" else note_en,
             fontsize=10.5, color=INK_SEC, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_MIG_PURE_PROBLEM{suffix}.png"); plt.close()
    print(f"F_MIG_PURE_PROBLEM{suffix}")

for lang in ["ko", "en"]:
    make_fig(lang)
print("Done.")
