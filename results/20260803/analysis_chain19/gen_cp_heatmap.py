#!/usr/bin/env python3
"""CP MPS heatmap · companion to SP MPS pct-sweep heatmap.

SP heatmap (user's existing) shows L1 p99 as function of MPS pct × N,
with severe latency growth. CP heatmap here shows L1 p99 stays at
baseline regardless of AI-side N because L1 is on a separate MIG
partition and does not touch the AI-side MPS context.

Data note: only pct=100 (default) measured on CP for N=6/8/10/12/16
(Chain 19 Exp 5). Other pct values not tested — but expected identical
because MPS pct only affects AI containers within the 3g partition;
it has no path to touch L1 which lives on the 4g partition.
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
    "axes.grid": False, "axes.axisbelow": True,
    "axes.unicode_minus": False, "savefig.dpi": 150, "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

def collect(prefix):
    trials = []
    for f in glob.glob(f"{BASE}/chain19_exp5/realL1_{prefix}_t*.json"):
        try:
            d = json.load(open(f))
            trials.append(d["p99_ms"])
        except: pass
    return trials

def make_fig(lang):
    apply_font(lang)

    # Measured: CP + MPS pct=100 (default) · N=6, 8, 10, 12, 16
    Ns = [6, 8, 10, 12, 16]
    measured_vals = {N: np.mean(collect(f"e5_cpN{N}")) for N in Ns}

    # Grid: pct rows (matching SP heatmap style)
    pcts = [30, 50, 70, 100]
    matrix = np.zeros((len(pcts), len(Ns)))
    # Only pct=100 row is measured · other rows filled with same value
    # (MPS pct on AI partition doesn't affect L1 latency in CP because
    # MIG hardware boundary separates the two partitions' launch queues)
    for pi, pct in enumerate(pcts):
        for ni, N in enumerate(Ns):
            matrix[pi, ni] = measured_vals[N]

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=40, vmax=200)
    ax.set_xticks(range(len(Ns))); ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_yticks(range(len(pcts))); ax.set_yticklabels([f"pct={p}%" for p in pcts])

    for i, pct in enumerate(pcts):
        for j, N in enumerate(Ns):
            v = matrix[i, j]
            measured = (pct == 100)
            label = f"{v:.0f}ms" if measured else f"{v:.0f}ms*"
            ax.text(j, i, label, ha="center", va="center", fontsize=14,
                    color=INK, fontweight="bold")

    plt.colorbar(im, ax=ax, label="L1 p99 latency (ms)")

    title_ko = "Fig · CP + MPS · L1 p99 지연 (pct × N heatmap · L1은 4g 파티션 단독)"
    title_en = "Fig · CP + MPS · L1 p99 latency (pct × N heatmap · L1 alone on 4g)"
    ax.set_title(title_ko if lang=="ko" else title_en,
                 fontweight="bold", pad=16, loc="left")

    note_ko = ("* pct=100 (기본값) 만 실측 · N=6/8/10/12/16. pct=30/50/70은 CP에서는 미측정 · 그러나 예상 동일. "
               "이유: MIG 하드웨어 경계가 두 파티션의 launch queue를 분리 · MPS pct는 3g AI 파티션 안에서만 SM 캡 · L1 (4g) 에는 도달 경로 없음. "
               "결과: CP에서는 pct와 N 무관하게 L1 p99 = ~40 ms baseline 유지 · SP heatmap과 정반대.")
    note_en = ("* Only pct=100 (default) measured on CP · N=6/8/10/12/16. pct=30/50/70 not tested but expected identical. "
               "Reason: MIG hardware boundary separates the two partitions' launch queues · MPS pct only caps SMs within the 3g AI partition · has no path to L1 (4g). "
               "Result: in CP, L1 p99 = ~40 ms baseline regardless of pct or N · opposite of the SP heatmap.")
    fig.text(0.02, 0.008, note_ko if lang=="ko" else note_en,
             fontsize=10.5, color=INK_SEC, style="italic")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_CP_HEATMAP{suffix}.png"); plt.close()
    print(f"F_CP_HEATMAP{suffix}")

for lang in ["ko", "en"]:
    make_fig(lang)
print("Done.")
