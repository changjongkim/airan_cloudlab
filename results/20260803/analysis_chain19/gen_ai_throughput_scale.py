#!/usr/bin/env python3
"""AI throughput scaling with N · SP + MPS on/off.

Data: Chain 18 p3 memcpy workload (memory-bandwidth AI proxy)
  SP · MIG Config A · L1 + N memcpy containers · MPS on vs off
  · N=1, 2, 4, 6, 8

Purpose: pair with F_MIG_PROBLEM_v2 (L1 latency growth) to expose
the SP trade-off. AI throughput scales with N when MPS is on
(coordination enables sharing), stalls when MPS is off (context
switch serialization).
"""
import glob, re, os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

BASE_18 = "/Users/changjongkim/New_research/cloudlab_results/results/20260725/chain18"
FIG     = "/Users/changjongkim/New_research/cloudlab_results/results/20260803/analysis_chain19/figures/mig_mps"

INK="#0f172a"; INK_SEC="#334155"; INK_MUT="#64748b"
SURFACE="#ffffff"; GRID="#e2e8f0"
COL_MPS_ON="#059669"; COL_MPS_OFF="#b91c1c"

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

def collect_rate(pattern):
    """Sum per-container rate= per trial."""
    trials = defaultdict(int)
    for f in sorted(glob.glob(pattern)):
        b = os.path.basename(f)
        m = re.search(r"_(t\d+)_", b)
        if not m: continue
        trial = m.group(1)
        rates = []
        for line in open(f, errors="ignore"):
            mm = re.search(r"rate=(\d+)/s", line)
            if mm: rates.append(int(mm.group(1)))
        if rates: trials[trial] += max(rates)
    return list(trials.values())

def make_fig(lang):
    apply_font(lang)
    Ns = [1, 2, 4, 6, 8]
    on = {N: collect_rate(f"{BASE_18}/p3_memcpyN{N}_MPSon_same_t*_c*.log") for N in Ns}
    off = {N: collect_rate(f"{BASE_18}/p3_memcpyN{N}_MPSoff_same_t*_c*.log") for N in Ns}

    fig, ax = plt.subplots(figsize=(13, 6.5))

    xs = np.arange(len(Ns))
    w = 0.35

    on_means = [np.mean(on[N])/1000 for N in Ns]
    on_max = [max(on[N])/1000 for N in Ns]
    on_min = [min(on[N])/1000 for N in Ns]
    off_means = [np.mean(off[N])/1000 for N in Ns]
    off_max = [max(off[N])/1000 for N in Ns]
    off_min = [min(off[N])/1000 for N in Ns]

    ax.bar(xs - w/2, on_means, w, color=COL_MPS_ON, alpha=0.85, edgecolor="white", linewidth=1.5,
           label=("SP + MPS on (조정 도구 활성)" if lang=="ko" else "SP + MPS on (coordination active)"))
    for i, (mn, mx, m) in enumerate(zip(on_min, on_max, on_means)):
        ax.plot([xs[i]-w/2, xs[i]-w/2], [mn, mx], color=INK, linewidth=2, alpha=0.85)
        ax.plot([xs[i]-w/2-0.06, xs[i]-w/2+0.06], [mn, mn], color=INK, linewidth=2, alpha=0.85)
        ax.plot([xs[i]-w/2-0.06, xs[i]-w/2+0.06], [mx, mx], color=INK, linewidth=2, alpha=0.85)
        ax.text(xs[i]-w/2, mx+25, f"{m:.0f}k", ha="center", fontsize=11,
                color=COL_MPS_ON, fontweight="bold")

    ax.bar(xs + w/2, off_means, w, color=COL_MPS_OFF, alpha=0.85, edgecolor="white", linewidth=1.5,
           label=("SP + MPS off (컨텍스트 시분할)" if lang=="ko" else "SP + MPS off (context time-slicing)"))
    for i, (mn, mx, m) in enumerate(zip(off_min, off_max, off_means)):
        ax.plot([xs[i]+w/2, xs[i]+w/2], [mn, mx], color=INK, linewidth=2, alpha=0.85)
        ax.plot([xs[i]+w/2-0.06, xs[i]+w/2+0.06], [mn, mn], color=INK, linewidth=2, alpha=0.85)
        ax.plot([xs[i]+w/2-0.06, xs[i]+w/2+0.06], [mx, mx], color=INK, linewidth=2, alpha=0.85)
        ax.text(xs[i]+w/2, mx+25, f"{m:.0f}k", ha="center", fontsize=11,
                color=COL_MPS_OFF, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_xlabel(("N (같은 4g 파티션 위 동일 AI 프로세스 수)"
                    if lang=="ko" else
                    "N (identical AI processes on 4g partition)"))
    ax.set_ylabel(("AI 집계 처리량 (k iter/s · memcpy 프록시)"
                    if lang=="ko" else
                    "AI aggregate throughput (k iter/s · memcpy proxy)"))
    ax.set_title(("Fig · SP + MPS · AI 처리량은 N에 따라 스케일 (MPS off는 시분할로 정체)"
                   if lang=="ko" else
                   "Fig · SP + MPS · AI throughput scales with N (MPS off saturates from time-slicing)"),
                 fontweight="bold", pad=16, loc="left")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(axis="y", alpha=0.5)
    ax.set_ylim(0, max(on_max)*1.15)

    note_ko = ("데이터: Chain 18 p3 · MIG Config A (4g SP) · L1 + N memcpy 컨테이너 · MPS on/off. "
               "MPS on: N 늘수록 aggregate throughput 선형 증가 (130k→987k). 조정 도구가 컨테이너 사이 SM 공유를 활성화. "
               "MPS off: N=2부터 정체 (~230k) · 컨텍스트 시분할로 순수 시간 지분 나눠갖기 · 처리량 총합이 성장 못 함. "
               "결론: SP 셋업에서 · MPS는 AI 처리량 확장의 필수 조건. 그러나 이 처리량 이득은 F_MIG_PROBLEM_v2의 L1 지연 붕괴와 동시에 발생 · trade-off.")
    note_en = ("Data: Chain 18 p3 · MIG Config A (4g SP) · L1 + N memcpy containers · MPS on/off. "
               "MPS on: linear growth 130k → 987k as N goes 1→8 (coordination enables inter-container SM sharing). "
               "MPS off: saturates at ~230k from N=2 (context time-slicing means processes just share wall-clock). "
               "Takeaway: In SP, MPS is required for AI throughput scaling — but this AI gain coincides with L1 latency collapse (F_MIG_PROBLEM_v2). Trade-off.")
    fig.text(0.02, 0.008, note_ko if lang=="ko" else note_en,
             fontsize=10.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_AI_THROUGHPUT_SCALE{suffix}.png"); plt.close()
    print(f"F_AI_THROUGHPUT_SCALE{suffix}")

for lang in ["ko", "en"]:
    make_fig(lang)
print("Done.")
