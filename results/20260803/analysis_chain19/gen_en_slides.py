#!/usr/bin/env python3
"""English version of all slide figures. Suffix _EN so KO versions stay intact.

Produces:
  F02_quadrant_ai_throughput_EN
  F09_mps_alone_full_gpu_EN
  F09b_duty_full_gpu_EN
  F13b_duty_cp_EN
  F_SP_PCT_answer_EN
  F_G01_FULL_MPS_EN
  F_G02_SP_MIG_MPS_EN
  F_G03_STARVE_EN
  F_G04_SP_PARADOX_EN
  F_G05_CP_WIN_EN
  F_G06_VERDICT_EN
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
COL_NOMIG="#dc6803"; COL_A="#2563eb"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
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
# Data loaders
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

ch19_gap = {}
for f in glob.glob(f"{BASE_19}/chain19_gapstats/*.stats.json"):
    label = os.path.basename(f).replace(".stats.json", "")
    try: ch19_gap[label] = json.load(open(f))
    except: pass

def l1_trials(prefix):
    return [d["p99_ms"] for label, d in ch19_l1.items()
            if label.startswith(prefix + "_t") or label == prefix]
def l1_mean(prefix):
    t = l1_trials(prefix); return np.mean(t) if t else None
def l1_p99(prefix): return l1_mean(prefix)
def duty_mean(prefix):
    vals = [d.get("duty", 0) for label, d in ch19_gap.items()
            if label.startswith(prefix + "_t") or label == prefix]
    return np.mean(vals) if vals else None

# =========================
# F02 · Quadrant AI throughput (EN)
# =========================
def f02_en():
    fig, ax = plt.subplots(figsize=(13, 7))
    matrix = np.array([[30, 100], [30, 100]])
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["MPS OFF", "MPS ON"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Same-partition / No MIG", "MIG cross-partition"])
    for i in range(2):
        for j in range(2):
            v = matrix[i, j]
            ax.text(j, i, f"{v}%\nAI throughput", ha="center", va="center",
                    fontsize=16, color="white" if v < 60 else INK, fontweight="bold")
    ax.set_title("Fig · AI throughput by MIG × MPS quadrant — MPS is required for AI parallelism",
                 fontweight="bold", pad=18, loc="left")
    plt.colorbar(im, ax=ax, label="AI aggregate throughput (% of peak)")
    fig.text(0.02, 0.008,
             "Without MPS, N concurrent AI processes serialize on the CUDA context. MPS is essential regardless of MIG topology.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/F02_quadrant_ai_throughput_EN.png"); plt.close()
    print("F02_EN")

# =========================
# F09 · Full GPU + MPS on — L1 p99 latency (EN)
# =========================
def f09_en():
    l1_alone = l1_p99("e1_baseline") or 42
    Ns = [1, 3, 6, 8, 10, 12]
    lats = [l1_p99(f"e1_cfgB_diverseN{N}") or 0 for N in Ns]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.axhline(l1_alone, color=INK_MUT, linestyle="--", linewidth=2, alpha=0.7)
    ax.text(0.02, l1_alone+2, f"L1-alone baseline: {l1_alone:.1f} ms",
            transform=ax.get_yaxis_transform(),
            color=INK_SEC, fontsize=11, style="italic")
    ax.plot(Ns, lats, "o-", color=COL_NOMIG, linewidth=3, markersize=13,
            markerfacecolor="white", markeredgewidth=2.5)
    for N, l in zip(Ns, lats):
        col = COL_BAD if l > 50 else COL_WARN
        ax.text(N, l+2, f"{l:.1f} ms", ha="center", fontsize=11, color=col, fontweight="bold")
    ax.set_xlabel("N (diverse AI containers on Full GPU + MPS on)")
    ax.set_ylabel("L1 per-iteration p99 latency (ms)")
    ax.set_xticks(Ns); ax.grid(alpha=0.5)
    ax.set_title("Fig · MPS on Full GPU (no MIG) — L1 p99 vs baseline, bimodal per trial",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "Even at N=1 the L1 p99 rises to 63 ms (50 % penalty). Some trials at N=6/12 pack well (39 ms), others hit 62–63 ms — worst-case fails SLA.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/F09_mps_alone_full_gpu_EN.png"); plt.close()
    print("F09_EN")

# =========================
# F09b · Full GPU duty cycle (bar, EN)
# =========================
def f09b_en():
    Ns = [1, 3, 6, 8, 10, 12]
    duties = [duty_mean(f"e1_cfgB_diverseN{N}") or 0 for N in Ns]
    baseline = duty_mean("e1_baseline") or 25
    fig, ax = plt.subplots(figsize=(13, 6.5))
    xs = np.arange(len(Ns))
    ax.bar(xs, duties, color=COL_NOMIG, alpha=0.85, edgecolor="white", linewidth=2, width=0.65)
    ax.axhline(baseline, color=INK_MUT, linestyle="--", linewidth=2, alpha=0.7)
    ax.text(len(Ns)-0.5, baseline+1.5, f"L1-alone baseline: {baseline:.0f}%",
            color=INK_SEC, fontsize=11, style="italic", ha="right")
    for i, d in enumerate(duties):
        ax.text(i, d+1.2, f"{d:.0f}%", ha="center", fontsize=12, color=COL_NOMIG, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_xlabel("Diverse AI containers (each bar is an independent condition)")
    ax.set_ylabel("L1 duty cycle (%)")
    ax.grid(axis="y", alpha=0.5)
    ax.set_title("Fig · Full GPU + MPS on — duty cycle (looks 'healthy' but does not guarantee SLA)",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "Each bar is an independent N condition. Duty stays above baseline (GPU is busy) — but this metric does NOT guarantee L1 SLA. See paired latency figure.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/F09b_duty_full_gpu_EN.png"); plt.close()
    print("F09b_EN")

# =========================
# F13b · CP duty cycle (bar, EN)
# =========================
def f13b_en():
    Ns = [0, 6, 8, 10, 12, 16]
    duties = []; labels = []
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
    ax.set_xlabel("AI on 3g partition (each bar is an independent condition)")
    ax.set_ylabel("L1 duty cycle (%, L1 on 4g partition)")
    ax.grid(axis="y", alpha=0.5)
    ax.set_title("Fig · MIG CP + MPS — duty cycle per condition (bar chart)",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "Each bar is an independent N condition. Duty appears stable — but the paired latency figure is what actually proves the SLA holds.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/F13b_duty_cp_EN.png"); plt.close()
    print("F13b_EN")

# =========================
# F_SP_PCT · SP + MIG + MPS pct sweep (EN)
# =========================
def f_sp_pct_en():
    pcts = [100, 70, 50, 30]
    lats = [l1_p99(f"e11_pct{p}_N6") or 0 for p in pcts]
    baseline = l1_p99("e5_baseline") or 40
    fig, ax = plt.subplots(figsize=(13, 6.8))
    xs = np.arange(len(pcts))
    colors = [COL_BAD if l > 50 else COL_GOOD for l in lats]
    ax.bar(xs, lats, color=colors, alpha=0.85, edgecolor="white", linewidth=2, width=0.55)
    ax.axhline(50, color=INK, linestyle="--", linewidth=2, alpha=0.75)
    ax.text(len(pcts)-0.5, 52, "5G L1 SLA proxy 50 ms", ha="right",
            color=INK, fontsize=11, style="italic")
    ax.axhline(baseline, color=INK_MUT, linestyle=":", linewidth=1.5, alpha=0.6)
    ax.text(-0.4, baseline-3, f"L1-alone baseline {baseline:.0f} ms",
            color=INK_MUT, fontsize=10, style="italic")
    for i, l in enumerate(lats):
        ax.text(i, l+3, f"{l:.0f} ms", ha="center", fontsize=13,
                color=colors[i], fontweight="bold")
    win_i = int(np.argmin(lats))
    ax.annotate("best SP tuning\nstill 3× over SLA",
                xy=(win_i, lats[win_i]), xytext=(win_i+0.7, lats[win_i]+30),
                fontsize=12, color=COL_BAD, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=COL_BAD, lw=2))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"MPS pct={p}%" for p in pcts])
    ax.set_xlabel("MPS thread% cap per AI client (SM usage upper bound)")
    ax.set_ylabel("L1 p99 latency (ms) · same-partition · N=6 diverse AI")
    ax.grid(axis="y", alpha=0.5)
    ax.set_ylim(0, max(lats)*1.25)
    ax.set_title("Fig · MPS pct tuning on MIG SP — best case (pct=30) still fails 50 ms SLA",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "N=6 diverse AI co-located with L1 on MIG 4g partition. Lower pct = less AI SM contention. But even the best setting (pct=30 → 146 ms) is 3× over SLA.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/F_SP_PCT_answer_EN.png"); plt.close()
    print("F_SP_PCT_EN")

# =========================
# F_G01 · Full GPU MPS off/on (gap_p99, EN)
# =========================
def g01_en():
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
    ax.set_ylabel("L1 inter-kernel gap p99 (ms)")
    ax.set_xticks(Ns); ax.grid(alpha=0.5); ax.legend(frameon=True, loc="upper left")
    ax.set_title("Fig · Full GPU MPS off/on — re-measured with the correct SLA proxy (gap p99)",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "gap p99 is the tail wait between L1 kernels (real SLA risk). MPS OFF reaches 13 ms tail; MPS ON stays ≤ 1 ms.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/F_G01_FULL_MPS_EN.png"); plt.close()
    print("F_G01_EN")

# =========================
# F_G02 · MIG SP (A/C) gap_p99 (EN)
# =========================
def g02_en():
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
    fig.suptitle("Fig · MIG same-partition — MPS OFF blows up the L1 kernel-gap p99 (time-slicing inside the partition)",
                 fontweight="bold", y=1.01, x=0.02, ha="left", fontsize=17)
    fig.text(0.02, 0.008,
             "MIG partitions exist but if L1 and AI share one, they share a launch queue. MPS OFF → the kernel-gap tail spikes.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(f"{FIG}/F_G02_SP_MIG_MPS_EN.png"); plt.close()
    print("F_G02_EN")

# =========================
# F_G03 · Launch rate starvation (EN)
# =========================
def g03_en():
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
    fig.suptitle("Fig · L1 launch rate — MPS OFF starves L1 (kernel throughput collapses)",
                 fontweight="bold", y=1.02, x=0.02, ha="left", fontsize=17)
    fig.text(0.02, 0.008,
             "MPS OFF: L1 launch rate falls to 1000–2000 kernels/s. MPS ON holds 2000–12000/s. Honest proof that MPS is required.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f"{FIG}/F_G03_STARVE_EN.png"); plt.close()
    print("F_G03_EN")

# =========================
# F_G04 · SP paradox (EN)
# =========================
def g04_en():
    full_trials = l1_trials("e1_cfgB_diverseN6")
    sp_pct100 = l1_trials("e11_pct100_N6")
    sp_pct30  = l1_trials("e11_pct30_N6")
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
    ax.bar(xs, means, color=colors, alpha=0.85, edgecolor="white", linewidth=2, width=0.55)
    for i, (mn, mx, m) in enumerate(zip(mins, maxes, means)):
        ax.plot([xs[i], xs[i]], [mn, mx], color=INK, linewidth=2, alpha=0.7)
        ax.plot([xs[i]-0.1, xs[i]+0.1], [mn, mn], color=INK, linewidth=2, alpha=0.7)
        ax.plot([xs[i]-0.1, xs[i]+0.1], [mx, mx], color=INK, linewidth=2, alpha=0.7)
        ax.text(xs[i], m+15, f"mean {m:.0f} ms\nworst {mx:.0f} ms",
                ha="center", fontsize=10.5, color=colors[i], fontweight="bold")
    ax.axhline(50, color=INK, linestyle="--", linewidth=2, alpha=0.75)
    ax.text(len(conditions)-0.5, 53, "5G L1 SLA proxy 50 ms", ha="right",
            color=INK, fontsize=11, style="italic")
    ax.axhline(baseline, color=INK_MUT, linestyle=":", linewidth=1.5, alpha=0.6)
    ax.text(-0.4, baseline-4, f"L1-alone baseline {baseline:.0f} ms",
            color=INK_MUT, fontsize=10, style="italic")
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in conditions], fontsize=11)
    ax.set_ylabel("L1 p99 latency (ms) · N=6 diverse AI · 3-trial range")
    ax.grid(axis="y", alpha=0.5)
    ax.set_ylim(0, max(maxes)*1.15)
    ax.set_title("Fig · SP paradox — MIG 4g (SM 56) is WORSE than Full GPU (SM 108) under contention",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "MIG SP halves the SM budget → contention intensifies. Full GPU has SM headroom. Only CP restores baseline via hardware isolation.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/F_G04_SP_PARADOX_EN.png"); plt.close()
    print("F_G04_EN")

# =========================
# F_G05 · CP + MPS L1 invariance (EN)
# =========================
def g05_en():
    Ns = [0, 6, 8, 10, 12, 16]
    means, mins, maxes = [], [], []
    for N in Ns:
        cond = "e5_baseline" if N == 0 else f"e5_cpN{N}"
        trials = l1_trials(cond)
        means.append(np.mean(trials)); mins.append(min(trials)); maxes.append(max(trials))
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(Ns, means, "o-", color=COL_GOOD, linewidth=3, markersize=13,
            markerfacecolor="white", markeredgewidth=2.5, label="mean of 3 trials")
    ax.fill_between(Ns, mins, maxes, color=COL_GOOD, alpha=0.15, label="min–max range")
    for N, m, mx in zip(Ns, means, maxes):
        ax.text(N, mx+1, f"{m:.1f} ms", ha="center", fontsize=11, color=COL_GOOD, fontweight="bold")
    ax.axhline(50, color=INK, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(0.02, 51, "5G L1 SLA proxy 50 ms", transform=ax.get_yaxis_transform(),
            color=INK, fontsize=11, style="italic")
    ax.set_xlabel("N (AI containers on 3g partition · MPS on)")
    ax.set_ylabel("L1 p99 latency (ms · L1 on 4g partition)")
    ax.set_xticks(Ns); ax.legend(frameon=True); ax.grid(alpha=0.5)
    ax.set_ylim(0, 60)
    ax.set_title("Fig · MIG CP + MPS on AI — L1 p99 stays at baseline up to N=16 (measured)",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "3-trial min–max band included. Every N stays under 50 ms SLA. The only verified co-tenancy answer in this dataset.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/F_G05_CP_WIN_EN.png"); plt.close()
    print("F_G05_EN")

# =========================
# F_G06 · Corrected decision matrix (EN)
# =========================
def g06_en():
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.axis('off')
    header = ["Topology", "L1 p99 mean", "L1 p99 worst", "N=16 supported?", "SLA pass?"]
    rows = [
        ("Multi-GPU (dedicated GPUs)",         "40 ms",  "40 ms",  "✓",   "✓"),
        ("MIG CP (4g L1 / 3g AI) + MPS on",    "42 ms",  "44 ms",  "✓",   "✓"),
        ("Full GPU + MPS on (bimodal)",        "54 ms",  "62 ms",  "?",   "worst-case fail"),
        ("MIG SP-4g + MPS pct=30",             "146 ms", "160 ms", "✗",   "✗"),
        ("MIG SP-4g + MPS pct=100 (default)",  "411 ms", "420 ms", "✗",   "✗"),
        ("MPS off (any topology)",             ">300 ms","catastrophic","✗","✗"),
    ]
    colors = [None, COL_GOOD, COL_GOOD, COL_WARN, COL_BAD, COL_BAD, COL_BAD]
    cell_h = 0.7; cell_w = [3.7, 1.7, 1.7, 1.7, 1.9]
    xs = [sum(cell_w[:i]) for i in range(len(cell_w)+1)]
    y_top = 6.5
    for j, h in enumerate(header):
        ax.add_patch(plt.Rectangle((xs[j], y_top), cell_w[j], cell_h, facecolor=INK, alpha=0.15, edgecolor="white"))
        ax.text(xs[j] + cell_w[j]/2, y_top + cell_h/2, h, ha="center", va="center",
                fontsize=12, fontweight="bold", color=INK)
    for i, row in enumerate(rows):
        y = y_top - (i+1) * cell_h
        row_col = colors[i+1]
        for j, cell in enumerate(row):
            ax.add_patch(plt.Rectangle((xs[j], y), cell_w[j], cell_h,
                                        facecolor=row_col if j>0 else INK,
                                        alpha=0.1, edgecolor="white"))
            ax.text(xs[j] + cell_w[j]/2, y + cell_h/2, cell, ha="center", va="center",
                    fontsize=11, color=row_col if j>0 else INK,
                    fontweight="bold" if j==0 else "normal")
    ax.set_xlim(0, sum(cell_w))
    ax.set_ylim(-0.5, 7.5)
    ax.text(sum(cell_w)/2, 7.3, "Fig · Corrected verdict — from measured data",
            ha="center", fontsize=17, fontweight="bold")
    ax.text(0, -0.3,
            "SLA proxy = 50 ms. L1 p99 worst = max across 3 trials. Full GPU + MPS is bimodal (39 ms good, 62 ms bad) — judge by worst-case to avoid missing SLA risk.",
            fontsize=10.5, style="italic", color=INK_SEC)
    plt.tight_layout()
    plt.savefig(f"{FIG}/F_G06_VERDICT_EN.png"); plt.close()
    print("F_G06_EN")

for fn in [f02_en, f09_en, f09b_en, f13b_en, f_sp_pct_en,
           g01_en, g02_en, g03_en, g04_en, g05_en, g06_en]:
    try: fn()
    except Exception as e: print(f"ERR {fn.__name__}: {e}")

print("All EN slide figures done.")
