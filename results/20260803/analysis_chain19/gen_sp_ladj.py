#!/usr/bin/env python3
"""Generate SP L1-adjacent (identical NRx) evidence figures.

Data source: Chain 17 identical-NRx grid experiment (2026-07)
Purpose:  the SP failure in Chain 19 Exp 11 was caused by mixing HEAVY
          general-purpose AI (Qwen/Whisper) into the L1 partition. When we
          restrict SP to L1-adjacent workloads (NRx-style small kernels),
          the gap_p99 stays under ~2 ms across N=1..8 — suggesting SP is
          feasible for the actual co-work use case.
Caveat:   Chain 17 measured gap_p99 only, not per-iter latency.
          Per-iter realL1 measurement for identical-NRx SP is the
          next-experiment candidate.
"""
import os, json, re
import numpy as np
import matplotlib.pyplot as plt

BASE_17 = "/Users/changjongkim/New_research/cloudlab_results/results/20260725"
BASE_19 = "/Users/changjongkim/New_research/cloudlab_results/results/20260803"
FIG     = f"{BASE_19}/analysis_chain19/figures/mig_mps"

INK="#0f172a"; INK_SEC="#334155"; INK_MUT="#64748b"
SURFACE="#ffffff"; GRID="#e2e8f0"
COL_GOOD="#059669"; COL_WARN="#d97706"; COL_BAD="#b91c1c"

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

ch17 = json.load(open(f"{BASE_17}/chain17_all_stats.json"))
def ch17_agg(cfg, N, mps, key):
    vals = [ch17[k][key] for k in ch17 if re.match(f"cfg{cfg}_A_nrxN{N}_MPS{mps}_t\\d+", k)]
    return np.mean(vals) if vals else None

def make_ladj(lang="ko"):
    """Split-panel: (a) identical-NRx SP gap_p99 vs N · (b) mixed workload comparison bar."""
    Ns = [1, 2, 3, 4, 6, 8]
    # Panel A: SP MIG 4g + identical NRx, gap_p99 across N
    ladj_A_on = [ch17_agg("A", N, "on",  "gap_p99")/1e6 for N in Ns]
    ladj_C_on = [ch17_agg("C", N, "on",  "gap_p99")/1e6 for N in Ns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={"width_ratios":[1.1, 1]})

    # ---- Panel A: gap_p99 line (identical NRx in SP)
    ax1.plot(Ns, ladj_A_on, "o-", color=COL_GOOD, linewidth=3, markersize=13,
             markerfacecolor="white", markeredgewidth=2.5,
             label="Config A · MIG 4g (SP · MPS on)")
    ax1.plot(Ns, ladj_C_on, "s-", color="#0369a1", linewidth=3, markersize=12,
             markerfacecolor="white", markeredgewidth=2.5,
             label="Config C · MIG 3g (SP · MPS on)")
    for N, v in zip(Ns, ladj_A_on):
        ax1.text(N, v+0.15, f"{v:.1f} ms", ha="center", fontsize=10, color=COL_GOOD, fontweight="bold")
    ax1.axhline(2.0, color=INK_MUT, linestyle=":", linewidth=1.5, alpha=0.6)
    if lang == "ko":
        ax1.text(0.02, 2.05, "gap p99 = 2 ms 참조선", transform=ax1.get_yaxis_transform(),
                 color=INK_MUT, fontsize=10, style="italic")
        ax1.set_xlabel("N (동일 NRx replica · MIG SP 파티션)")
        ax1.set_ylabel("L1 kernel gap p99 (ms)")
        ax1.set_title("(a) SP · L1 + identical NRx (small L1-adjacent workload) — gap p99 ≤ 2 ms",
                      fontweight="bold", loc="left", color=INK_SEC)
    else:
        ax1.text(0.02, 2.05, "gap p99 = 2 ms reference", transform=ax1.get_yaxis_transform(),
                 color=INK_MUT, fontsize=10, style="italic")
        ax1.set_xlabel("N (identical NRx replicas · MIG SP partition)")
        ax1.set_ylabel("L1 inter-kernel gap p99 (ms)")
        ax1.set_title("(a) SP · L1 + identical NRx (small L1-adjacent workload) — gap p99 stays ≤ 2 ms",
                      fontweight="bold", loc="left", color=INK_SEC)
    ax1.set_xticks(Ns); ax1.legend(frameon=True, loc="upper left")
    ax1.set_ylim(0, 6)
    ax1.grid(alpha=0.5)

    # ---- Panel B: comparison bar — identical NRx vs diverse AI in same SP topology
    if lang == "ko":
        labels = ["SP + identical NRx\n(gap p99 · Chain 17)",
                  "SP + diverse AI mix\n(L1 p99 · Chain 19 Exp 11)"]
    else:
        labels = ["SP + identical NRx\n(gap p99 · Chain 17)",
                  "SP + diverse AI mix\n(L1 p99 · Chain 19 Exp 11)"]
    vals = [ladj_A_on[4], 146]  # N=6 for both metrics — different metrics but qualitative comparison
    colors = [COL_GOOD, COL_BAD]
    xs = np.arange(len(labels))
    ax2.bar(xs, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=2, width=0.55)
    for i, v in enumerate(vals):
        ax2.text(i, v+3, f"{v:.1f} ms" + ("*" if i == 0 else ""),
                 ha="center", fontsize=13, color=colors[i], fontweight="bold")
    ax2.set_xticks(xs); ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel("latency-related metric (ms) · N=6" if lang == "en" else "지연 지표 (ms) · N=6")
    ax2.grid(axis="y", alpha=0.5)
    ax2.set_ylim(0, 170)
    if lang == "ko":
        ax2.set_title("(b) 같은 SP 위상, 워크로드만 다름 — 100× 이상 차이",
                      fontweight="bold", loc="left", color=INK_SEC)
    else:
        ax2.set_title("(b) Same SP topology, different workload — 100× gap",
                      fontweight="bold", loc="left", color=INK_SEC)

    if lang == "ko":
        fig.suptitle("SP는 워크로드 종류가 결정 — L1-adjacent (NRx-only) 이면 gap p99 ≤ 2 ms, LLM 섞이면 100 ms+ 붕괴",
                     fontweight="bold", y=1.02, x=0.02, ha="left", fontsize=17)
        fig.text(0.02, 0.008,
                 "* (a)는 gap p99 (kernel 사이 tail 간격) · (b) 좌측은 gap p99, 우측은 L1 p99 iteration latency — 다른 지표지만 심각도 대비 뚜렷. "
                 "SP + identical NRx 조건의 realL1 per-iter 측정은 다음 실험 후보.",
                 fontsize=10.5, color=INK_SEC, style="italic")
    else:
        fig.suptitle("SP is workload-dependent — L1-adjacent (NRx-only) yields gap p99 ≤ 2 ms; heavy LLMs cause 100 ms+ collapse",
                     fontweight="bold", y=1.02, x=0.02, ha="left", fontsize=17)
        fig.text(0.02, 0.008,
                 "* (a) is gap p99 (inter-kernel tail) · (b) left = gap p99, right = L1 per-iter p99 — different metrics, "
                 "but the severity gap is unambiguous. Per-iter measurement of SP + identical NRx is the next-experiment candidate.",
                 fontsize=10.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_G07_SP_LADJ{suffix}.png"); plt.close()
    print(f"F_G07_SP_LADJ{suffix}")

for lang in ["ko", "en"]:
    make_ladj(lang)
print("Done.")
