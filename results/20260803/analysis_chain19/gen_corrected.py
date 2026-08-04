#!/usr/bin/env python3
"""Corrected slide figures based on verified data.

Fixes:
  - Chain 17 figures used dur_med+gap_med which underreports MPS OFF cost.
    Switched to gap_p99 (tail interval) and launch_rate (starvation proxy).
  - Chain 19 hardcoded "SP+MPS pct=30 = 45ms" was wrong (real=146ms). Removed.
  - Reveals that MIG SP (4g) is WORSE than Full GPU because 4g has only 56 SMs.

Generates:
  F_G01_FULL_MPS   — Full GPU MPS off/on (Chain 17 gap_p99 · Config B)
  F_G02_SP_MIG_MPS — MIG SP MPS off/on (Chain 17 gap_p99 · Config A/C)
  F_G03_STARVE     — launch rate starvation under MPS off across configs
  F_G04_SP_PARADOX — MIG SP-4g worse than Full GPU (Chain 19 real L1)
  F_G05_CP_WIN     — CP + MPS L1 baseline preserved (Chain 19 real L1)
  F_G06_VERDICT    — corrected decision matrix
"""
import os, json, glob, re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_17 = "/Users/changjongkim/New_research/cloudlab_results/results/20260725"
BASE_19 = "/Users/changjongkim/New_research/cloudlab_results/results/20260803"
FIG     = f"{BASE_19}/analysis_chain19/figures/mig_mps"

INK="#0f172a"; INK_SEC="#334155"; INK_MUT="#64748b"
SURFACE="#ffffff"; GRID="#e2e8f0"
COL_GOOD="#059669"; COL_WARN="#d97706"; COL_BAD="#b91c1c"
COL_MPS_ON="#059669"; COL_MPS_OFF="#b91c1c"; COL_BASELINE="#0f172a"

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

# =========================
# Load data
# =========================
ch17 = json.load(open(f"{BASE_17}/chain17_all_stats.json"))

def ch17_agg(cfg, N, mps, key):
    vals = [ch17[k][key] for k in ch17 if re.match(f"cfg{cfg}_A_nrxN{N}_MPS{mps}_t\\d+", k)]
    return np.mean(vals) if vals else None

ch19_l1 = {}
for f in glob.glob(f"{BASE_19}/chain19_exp*/realL1_*.json"):
    try:
        d = json.load(open(f))
        ch19_l1[d["label"]] = d
    except: pass

def l1_trials(prefix):
    return [d["p99_ms"] for label, d in ch19_l1.items()
            if label.startswith(prefix + "_t") or label == prefix]

def l1_mean(prefix):
    t = l1_trials(prefix)
    return np.mean(t) if t else None

def l1_max(prefix):
    t = l1_trials(prefix)
    return max(t) if t else None

# =========================
# F_G01 · Full GPU (Config B) — gap_p99 as SLA proxy
# =========================
def fig_g01():
    Ns = [1, 2, 3, 4, 6, 8]
    off_ms = [ch17_agg("B", N, "off", "gap_p99")/1e6 for N in Ns]
    on_ms  = [ch17_agg("B", N, "on",  "gap_p99")/1e6 for N in Ns]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(Ns, off_ms, "s-", color=COL_MPS_OFF, linewidth=3, markersize=12,
            markerfacecolor="white", markeredgewidth=2.5, label="MPS OFF")
    ax.plot(Ns, on_ms,  "o-", color=COL_MPS_ON,  linewidth=3, markersize=12,
            markerfacecolor="white", markeredgewidth=2.5, label="MPS ON")
    for N, v in zip(Ns, off_ms):
        ax.text(N, v+0.5, f"{v:.1f}", ha="center", fontsize=10, color=COL_MPS_OFF, fontweight="bold")
    for N, v in zip(Ns, on_ms):
        ax.text(N, v-0.8, f"{v:.1f}", ha="center", fontsize=10, color=COL_MPS_ON, fontweight="bold")
    ax.set_xlabel("N (Full GPU · identical NRx replicas)")
    ax.set_ylabel("L1 kernel 사이 gap p99 (ms)")
    ax.set_xticks(Ns); ax.grid(alpha=0.5); ax.legend(frameon=True, loc="upper left")
    ax.set_title("F_G01 · Full GPU MPS off/on — 실제 SLA 지표(gap_p99)로 재측정",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "gap_p99가 L1 kernel 사이의 tail 대기 시간(SLA 위험). MPS OFF는 최대 13ms 대기, MPS ON은 ≤1ms.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/F_G01_FULL_MPS.png"); plt.close()
    print("F_G01")

# =========================
# F_G02 · MIG SP (Config A/C) — gap_p99
# =========================
def fig_g02():
    Ns = [1, 2, 3, 4, 6, 8]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, cfg, title in [(axes[0], "A", "Config A · MIG 4g (SP)"),
                             (axes[1], "C", "Config C · MIG 3g (SP)")]:
        off_ms = [ch17_agg(cfg, N, "off", "gap_p99")/1e6 for N in Ns]
        on_ms  = [ch17_agg(cfg, N, "on",  "gap_p99")/1e6 for N in Ns]
        ax.plot(Ns, off_ms, "s-", color=COL_MPS_OFF, linewidth=3, markersize=11,
                markerfacecolor="white", markeredgewidth=2.5, label="MPS OFF")
        ax.plot(Ns, on_ms,  "o-", color=COL_MPS_ON,  linewidth=3, markersize=11,
                markerfacecolor="white", markeredgewidth=2.5, label="MPS ON")
        for N, v in zip(Ns, off_ms):
            ax.text(N, v+0.5, f"{v:.1f}", ha="center", fontsize=9, color=COL_MPS_OFF, fontweight="bold")
        for N, v in zip(Ns, on_ms):
            ax.text(N, v-0.8, f"{v:.1f}", ha="center", fontsize=9, color=COL_MPS_ON, fontweight="bold")
        ax.set_xlabel("N (concurrent NRx on same MIG partition)")
        ax.set_ylabel("gap p99 (ms)")
        ax.set_title(title, fontweight="bold", loc="left", color=INK_SEC)
        ax.set_xticks(Ns); ax.grid(alpha=0.5)
    axes[0].legend(loc="upper left", frameon=True)
    fig.suptitle("F_G02 · MIG same-partition — MPS OFF에서 L1 gap p99 심각히 상승 (파티션 안에서도 시분할)",
                 fontweight="bold", y=1.01, x=0.02, ha="left", fontsize=17)
    fig.text(0.02, 0.008,
             "MIG 파티션을 켜도 L1과 AI가 같은 파티션에 있으면 launch queue 공유. MPS OFF면 kernel gap tail이 급증.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(f"{FIG}/F_G02_SP_MIG_MPS.png"); plt.close()
    print("F_G02")

# =========================
# F_G03 · Launch rate starvation (MPS OFF vs ON) — grouped bar
# =========================
def fig_g03():
    Ns = [1, 2, 3, 4, 6, 8]
    configs = [("A", "MIG 4g"), ("B", "Full GPU"), ("C", "MIG 3g")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for i, (cfg, name) in enumerate(configs):
        ax = axes[i]
        off = [ch17_agg(cfg, N, "off", "launch_rate")/1000 for N in Ns]
        on  = [ch17_agg(cfg, N, "on",  "launch_rate")/1000 for N in Ns]
        xs = np.arange(len(Ns)); w = 0.38
        ax.bar(xs - w/2, off, w, color=COL_MPS_OFF, alpha=0.85, edgecolor="white", linewidth=1.5, label="MPS OFF")
        ax.bar(xs + w/2, on,  w, color=COL_MPS_ON,  alpha=0.85, edgecolor="white", linewidth=1.5, label="MPS ON")
        ax.set_xticks(xs); ax.set_xticklabels([f"N={n}" for n in Ns])
        ax.set_xlabel("N"); ax.set_ylabel("L1 launch rate (k kernels/s)")
        ax.set_title(name, fontweight="bold", loc="left", color=INK_SEC)
        ax.grid(axis="y", alpha=0.5)
    axes[0].legend(loc="upper right", frameon=True)
    fig.suptitle("F_G03 · L1 launch rate — MPS OFF는 L1을 굶긴다 (kernel 처리량 급감)",
                 fontweight="bold", y=1.02, x=0.02, ha="left", fontsize=17)
    fig.text(0.02, 0.008,
             "MPS OFF: L1 launch rate 1000~2000/s로 붕괴. MPS ON: 2000~12000/s 유지. 왜 MPS가 필수인지 정직한 지표.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f"{FIG}/F_G03_STARVE.png"); plt.close()
    print("F_G03")

# =========================
# F_G04 · SP paradox — MIG 4g는 오히려 Full GPU보다 나쁨 (Chain 19 real L1)
# =========================
def fig_g04():
    """Direct comparison at N=6 diverse AI, MPS on."""
    # Full GPU + MPS on (Exp 1)
    full_trials = l1_trials("e1_cfgB_diverseN6")
    # MIG SP-4g + MPS pct=100 (Exp 11) — analogous "MPS on default" case
    sp_pct100 = l1_trials("e11_pct100_N6")
    sp_pct30  = l1_trials("e11_pct30_N6")
    # CP (Exp 5) for reference
    cp        = l1_trials("e5_cpN6")

    conditions = [
        ("Full GPU\n+ MPS on", full_trials, COL_WARN),
        ("MIG SP-4g\n+ MPS pct=100", sp_pct100, COL_BAD),
        ("MIG SP-4g\n+ MPS pct=30\n(best SP tuning)", sp_pct30, COL_BAD),
        ("MIG CP\n(4g L1 / 3g AI)", cp, COL_GOOD),
    ]
    baseline = 38.5

    fig, ax = plt.subplots(figsize=(14, 6.8))
    xs = np.arange(len(conditions))
    means = [np.mean(t) for _, t, _ in conditions]
    maxes = [max(t) for _, t, _ in conditions]
    mins  = [min(t) for _, t, _ in conditions]
    colors = [c for _, _, c in conditions]

    # Bar = mean, error = min/max range
    ax.bar(xs, means, color=colors, alpha=0.85, edgecolor="white", linewidth=2, width=0.55)
    for i, (mn, mx, m) in enumerate(zip(mins, maxes, means)):
        # Range indicator
        ax.plot([xs[i], xs[i]], [mn, mx], color=INK, linewidth=2, alpha=0.7)
        ax.plot([xs[i]-0.1, xs[i]+0.1], [mn, mn], color=INK, linewidth=2, alpha=0.7)
        ax.plot([xs[i]-0.1, xs[i]+0.1], [mx, mx], color=INK, linewidth=2, alpha=0.7)
        ax.text(xs[i], m+15, f"평균 {m:.0f}ms\n최악 {mx:.0f}ms",
                ha="center", fontsize=10.5, color=colors[i], fontweight="bold")

    ax.axhline(50, color=INK, linestyle="--", linewidth=2, alpha=0.75)
    ax.text(len(conditions)-0.5, 53, "5G L1 SLA proxy 50 ms", ha="right",
            color=INK, fontsize=11, style="italic")
    ax.axhline(baseline, color=INK_MUT, linestyle=":", linewidth=1.5, alpha=0.6)
    ax.text(-0.4, baseline-4, f"L1 단독 baseline {baseline:.0f}ms",
            color=INK_MUT, fontsize=10, style="italic")

    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in conditions], fontsize=11)
    ax.set_ylabel("L1 p99 지연 (ms) · N=6 다이버스 AI · 3 trial 범위")
    ax.grid(axis="y", alpha=0.5)
    ax.set_ylim(0, max(maxes)*1.15)
    ax.set_title("F_G04 · SP 역설 — MIG 4g (SM 56) 는 Full GPU (SM 108) 보다 오히려 나쁨",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "MIG SP는 파티션이 SM을 반토막내서 압박 심화. Full GPU는 SM 여유. CP는 완전 격리로 baseline 유지.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/F_G04_SP_PARADOX.png"); plt.close()
    print("F_G04")

# =========================
# F_G05 · CP + MPS L1 invariance (real numbers, corrected)
# =========================
def fig_g05():
    Ns = [0, 6, 8, 10, 12, 16]
    means, mins, maxes = [], [], []
    for N in Ns:
        cond = "e5_baseline" if N == 0 else f"e5_cpN{N}"
        trials = l1_trials(cond)
        means.append(np.mean(trials))
        mins.append(min(trials))
        maxes.append(max(trials))
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(Ns, means, "o-", color=COL_GOOD, linewidth=3, markersize=13,
            markerfacecolor="white", markeredgewidth=2.5, label="mean of 3 trials")
    ax.fill_between(Ns, mins, maxes, color=COL_GOOD, alpha=0.15, label="min–max range")
    for N, m, mx in zip(Ns, means, maxes):
        ax.text(N, mx+1, f"{m:.1f} ms", ha="center", fontsize=11, color=COL_GOOD, fontweight="bold")
    ax.axhline(50, color=INK, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(0.02, 51, "5G L1 SLA proxy 50 ms", transform=ax.get_yaxis_transform(),
            color=INK, fontsize=11, style="italic")
    ax.set_xlabel("N (AI 컨테이너 수 on 3g 파티션 · MPS on)")
    ax.set_ylabel("L1 p99 지연 (ms · L1 on 4g 파티션)")
    ax.set_xticks(Ns); ax.legend(frameon=True); ax.grid(alpha=0.5)
    ax.set_ylim(0, 60)
    ax.set_title("F_G05 · MIG CP + MPS on AI — L1 p99가 N=16까지 baseline 유지 (실측)",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "3 trial min–max 범위 포함. 모든 N에서 SLA 50ms 아래. 유일하게 검증된 co-tenancy 답.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/F_G05_CP_WIN.png"); plt.close()
    print("F_G05")

# =========================
# F_G06 · Corrected decision matrix
# =========================
def fig_g06():
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.axis('off')
    header = ["Topology", "L1 p99 mean", "L1 p99 worst", "N=16 지원?", "SLA 통과?"]
    rows = [
        ("Multi-GPU (별도 GPU)",             "40 ms",  "40 ms",  "✓",   "✓"),
        ("MIG CP (4g L1 / 3g AI) + MPS on", "42 ms",  "44 ms",  "✓",   "✓"),
        ("Full GPU + MPS on (bimodal)",     "54 ms",  "62 ms",  "?",   "worst-case 실패"),
        ("MIG SP-4g + MPS pct=30",          "146 ms", "160 ms", "✗",   "✗"),
        ("MIG SP-4g + MPS pct=100 (기본)",  "411 ms", "420 ms", "✗",   "✗"),
        ("MPS off (모든 topology)",         ">300 ms","catastrophic","✗","✗"),
    ]
    colors = [None, COL_GOOD, COL_GOOD, COL_WARN, COL_BAD, COL_BAD, COL_BAD]

    cell_h = 0.7; cell_w = [3.7, 1.7, 1.7, 1.3, 1.9]
    xs = [sum(cell_w[:i]) for i in range(len(cell_w)+1)]
    y_top = 6.5

    for j, h in enumerate(header):
        ax.add_patch(plt.Rectangle((xs[j], y_top), cell_w[j], cell_h, facecolor=INK, alpha=0.15, edgecolor="white"))
        ax.text(xs[j] + cell_w[j]/2, y_top + cell_h/2, h, ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
    for i, row in enumerate(rows):
        y = y_top - (i+1) * cell_h
        row_col = colors[i+1]
        for j, cell in enumerate(row):
            ax.add_patch(plt.Rectangle((xs[j], y), cell_w[j], cell_h, facecolor=row_col if j>0 else INK,
                                        alpha=0.1, edgecolor="white"))
            ax.text(xs[j] + cell_w[j]/2, y + cell_h/2, cell, ha="center", va="center",
                    fontsize=11, color=row_col if j>0 else INK,
                    fontweight="bold" if j==0 else "normal")
    ax.set_xlim(0, sum(cell_w))
    ax.set_ylim(-0.5, 7.5)
    ax.text(sum(cell_w)/2, 7.3, "F_G06 · Corrected verdict — 실측 데이터 기준",
            ha="center", fontsize=17, fontweight="bold")
    ax.text(0, -0.3,
            "SLA proxy 50ms. L1 p99 worst = 3 trial 중 최댓값. Full GPU + MPS는 bimodal (좋을 때 39ms, 나쁠 때 62ms) — worst-case로 판단해야 SLA 위험 안 놓침.",
            fontsize=10.5, style="italic", color=INK_SEC)
    plt.tight_layout()
    plt.savefig(f"{FIG}/F_G06_VERDICT.png"); plt.close()
    print("F_G06")

for fn in [fig_g01, fig_g02, fig_g03, fig_g04, fig_g05, fig_g06]:
    try: fn()
    except Exception as e: print(f"ERR {fn.__name__}: {e}")
print("Done.")
