#!/usr/bin/env python3
"""Generate duty-cycle counterpart figures for slides that need (a)/(b) pairing.

Purpose: the presentation makes the point "duty cycle is misleading vs L1 latency".
For that argument to land, each latency figure needs its duty-cycle twin side by side.

Pairs generated:
  F09b — Full GPU + MPS on, duty cycle (companion to F09 latency)
  F13b — MIG CP + MPS, duty cycle across N (companion to F13 latency invariance)
  F28b — Duty ranking vs latency ranking mismatch (for verdict slide 10)
"""
import os, json, glob
import numpy as np
import matplotlib.pyplot as plt

BASE_19 = "/Users/changjongkim/New_research/cloudlab_results/results/20260803"
FIG     = f"{BASE_19}/analysis_chain19/figures/mig_mps"
os.makedirs(FIG, exist_ok=True)

INK="#0f172a"; INK_SEC="#334155"; INK_MUT="#64748b"
SURFACE="#ffffff"; GRID="#e2e8f0"
COL_GOOD="#059669"; COL_WARN="#d97706"; COL_BAD="#b91c1c"
COL_NOMIG="#dc6803"

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

# Load gap stats and per-iter latency
ch19_gap = {}
for f in glob.glob(f"{BASE_19}/chain19_gapstats/*.stats.json"):
    label = os.path.basename(f).replace(".stats.json", "")
    try: ch19_gap[label] = json.load(open(f))
    except: pass

ch19_l1 = {}
for f in glob.glob(f"{BASE_19}/chain19_exp*/realL1_*.json"):
    try:
        d = json.load(open(f))
        ch19_l1[d["label"]] = d
    except: pass

def duty_mean(cond_prefix):
    vals = [d.get("duty", 0) for label, d in ch19_gap.items()
            if label.startswith(cond_prefix + "_t") or label == cond_prefix]
    return np.mean(vals) if vals else None

def l1_p99(cond_prefix):
    vals = [d["p99_ms"] for label, d in ch19_l1.items()
            if label.startswith(cond_prefix + "_t") or label == cond_prefix]
    return np.mean(vals) if vals else None

# =========================
# F09b — Full GPU duty cycle (companion to F09 latency)
# =========================
def fig09b():
    """Duty cycle for Full GPU + MPS on, N sweep — BAR chart (each N discrete)."""
    Ns = [1, 3, 6, 8, 10, 12]
    duties = [duty_mean(f"e1_cfgB_diverseN{N}") or 0 for N in Ns]
    baseline = duty_mean("e1_baseline") or 25
    fig, ax = plt.subplots(figsize=(13, 6.5))
    xs = np.arange(len(Ns))
    ax.bar(xs, duties, color=COL_NOMIG, alpha=0.85, edgecolor="white", linewidth=2, width=0.65)
    ax.axhline(baseline, color=INK_MUT, linestyle="--", linewidth=2, alpha=0.7)
    ax.text(len(Ns)-0.5, baseline+1.5, f"L1 alone baseline: {baseline:.0f}%",
            color=INK_SEC, fontsize=11, style="italic", ha="right")
    for i, d in enumerate(duties):
        ax.text(i, d+1.2, f"{d:.0f}%", ha="center", fontsize=12, color=COL_NOMIG, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_xlabel("다이버스 AI 컨테이너 수 (각 막대는 독립 조건)")
    ax.set_ylabel("L1 duty cycle (%)")
    ax.grid(axis="y", alpha=0.5)
    ax.set_title("F09b · Full GPU + MPS on — duty cycle (건강해 '보이는' 지표, 조건별 이산 측정)",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "각 막대는 독립된 N 조건. Duty가 baseline 이상으로 유지 → GPU가 바쁨. "
             "하지만 이 지표는 L1 SLA를 보장하지 않는다 (실제 지연은 옆 그림 참조).",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/F09b_duty_full_gpu.png"); plt.close()
    print("F09b")

# =========================
# F13b — CP + MPS duty across N (companion to F13 latency invariance)
# =========================
def fig13b():
    """CP + MPS duty across N — BAR chart. Each N is a discrete condition."""
    Ns = [0, 6, 8, 10, 12, 16]
    duties = []
    labels = []
    for N in Ns:
        cond = "e5_baseline" if N == 0 else f"e5_cpN{N}"
        duties.append(duty_mean(cond) or 0)
        labels.append("baseline\n(L1 alone)" if N == 0 else f"N={N}")
    fig, ax = plt.subplots(figsize=(13, 6.5))
    xs = np.arange(len(Ns))
    colors = [INK_MUT] + [COL_GOOD]*(len(Ns)-1)
    ax.bar(xs, duties, color=colors, alpha=0.85, edgecolor="white", linewidth=2, width=0.65)
    for i, d in enumerate(duties):
        ax.text(i, d+0.8, f"{d:.0f}%", ha="center", fontsize=12, color=colors[i], fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel("AI on 3g 파티션 (각 막대는 독립 조건)")
    ax.set_ylabel("L1 duty cycle (%, L1 on 4g 파티션)")
    ax.grid(axis="y", alpha=0.5)
    ax.set_title("F13b · MIG CP + MPS — duty cycle 조건별 이산 측정",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "각 막대는 독립된 N 조건. Duty는 안정적으로 보이지만 이것만으로 L1 SLA를 보장하지 않는다. "
             "옆 그림의 L1 p99 지연이 실제 근거.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/F13b_duty_cp.png"); plt.close()
    print("F13b")

# =========================
# F04b — Duty ranking vs latency ranking mismatch (for verdict slide)
# =========================
def fig04b():
    """Show explicit mismatch: same conditions, different rankings."""
    conditions = [
        ("Full GPU + MPS on\n(N=1)",  "e1_cfgB_diverseN1"),
        ("Full GPU + MPS on\n(N=6)",  "e1_cfgB_diverseN6"),
        ("SP + MPS pct=100\n(N=6)",  "e11_pct100_N6"),
        ("SP + MPS pct=30\n(N=6)",   "e11_pct30_N6"),
        ("MIG CP + MPS\n(N=6)",       "e5_cpN6"),
        ("MIG CP + MPS\n(N=16)",      "e5_cpN16"),
    ]
    labels = [c[0] for c in conditions]
    duties = [duty_mean(c[1]) or 0 for c in conditions]
    lats   = [l1_p99(c[1]) or 0 for c in conditions]
    baseline_lat = l1_p99("e5_baseline") or 40

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    # (left) duty ranking
    order_duty = np.argsort(duties)[::-1]
    xs = np.arange(len(labels))
    colors_d = [COL_BAD if l1_p99(conditions[i][1]) and l1_p99(conditions[i][1]) > 50 else COL_GOOD
                for i in order_duty]
    ax1.barh(xs, [duties[i] for i in order_duty], color=colors_d, alpha=0.85,
             edgecolor="white", linewidth=2)
    ax1.set_yticks(xs)
    ax1.set_yticklabels([labels[i] for i in order_duty], fontsize=11)
    ax1.invert_yaxis()
    ax1.set_xlabel("Duty cycle (%)")
    ax1.set_title("(좌) Duty cycle 순위 — '바쁘게 보이는' 순서",
                  fontweight="bold", loc="left", color=INK_SEC)
    for i, idx in enumerate(order_duty):
        ax1.text(duties[idx]+0.5, i, f"{duties[idx]:.0f}%", va="center", fontsize=11,
                 color=colors_d[i], fontweight="bold")

    # (right) latency ranking
    order_lat = np.argsort(lats)
    colors_l = [COL_BAD if lats[i] > 50 else COL_GOOD for i in order_lat]
    ax2.barh(xs, [lats[i] for i in order_lat], color=colors_l, alpha=0.85,
             edgecolor="white", linewidth=2)
    ax2.set_yticks(xs)
    ax2.set_yticklabels([labels[i] for i in order_lat], fontsize=11)
    ax2.invert_yaxis()
    ax2.axvline(baseline_lat, color=INK, linestyle="--", linewidth=1.5, alpha=0.6)
    ax2.text(baseline_lat, len(labels)-0.5, f" baseline {baseline_lat:.0f}ms",
             color=INK, fontsize=10, style="italic", va="bottom")
    ax2.set_xlabel("L1 p99 지연 (ms)")
    ax2.set_title("(우) 실제 L1 p99 지연 순위 — SLA 순서",
                  fontweight="bold", loc="left", color=INK_SEC)
    for i, idx in enumerate(order_lat):
        ax2.text(lats[idx]+2, i, f"{lats[idx]:.0f}ms", va="center", fontsize=11,
                 color=colors_l[i], fontweight="bold")

    fig.suptitle("F04b · Duty 순위 ≠ 지연 순위. Duty로 판단하면 SLA 실패 조건을 '건강'으로 오독한다.",
                 fontweight="bold", y=1.02, x=0.02, ha="left", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f"{FIG}/F04b_duty_vs_latency_ranking.png"); plt.close()
    print("F04b")

for fn in [fig09b, fig13b, fig04b]:
    try: fn()
    except Exception as e: print(f"ERR {fn.__name__}: {e}")

print("Done.")
