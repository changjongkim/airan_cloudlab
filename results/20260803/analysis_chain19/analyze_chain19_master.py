#!/usr/bin/env python3
"""Chain 19 master analysis — aggregate all 273 conditions, generate figures.

Directory structure:
  /Users/changjongkim/New_research/cloudlab_results/results/20260803/
    ├── chain19_exp{1..13}/           # raw logs + JSON per experiment
    ├── chain19_gapstats/             # 273 gap stats JSON files
    └── analysis_chain19/             # THIS DIRECTORY
        ├── analyze_chain19_master.py # this script
        ├── chain19_summary.json      # unified aggregate
        └── figures/                  # per-experiment polished figures
"""
import os, json, glob, re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260803"
GAPS = f"{BASE}/chain19_gapstats"
OUT  = f"{BASE}/analysis_chain19"
FIG  = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)

# Design tokens (matching Chain 18 polished style)
INK="#0f172a"; INK_SEC="#334155"; INK_MUT="#64748b"
SURFACE="#ffffff"; GRID="#e2e8f0"
COL_BASELINE="#0f172a"; COL_GOOD="#059669"; COL_WARN="#d97706"; COL_BAD="#b91c1c"
COL_MPS_ON="#059669"; COL_MPS_OFF="#b91c1c"
COL_A="#2563eb"; COL_B="#7c3aed"; COL_C="#dc6803"

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
    "axes.unicode_minus": False, "savefig.dpi": 160, "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

# =============================================================
# Load all 273 stats + parse labels
# =============================================================
def load_all():
    data = {}
    for f in sorted(glob.glob(f"{GAPS}/*.stats.json")):
        label = os.path.basename(f).replace(".stats.json", "")
        try:
            with open(f) as fp: data[label] = json.load(fp)
        except: pass
    return data

def parse_label(label):
    """Extract (exp_id, condition, trial) from label like e1_cfgB_diverseN6_t2."""
    m = re.match(r"^e(\d+)_(.+?)_t(\d+)$", label)
    if m: return (int(m.group(1)), m.group(2), int(m.group(3)))
    # For AI-side traces: e2_N6_t1_ai3
    m = re.match(r"^e(\d+)_(.+?)_t(\d+)_ai(\d+)$", label)
    if m: return (int(m.group(1)), m.group(2)+"_ai", int(m.group(3)))
    # For L1 traces in exp2: e2_N6_t1_l1
    m = re.match(r"^e(\d+)_(.+?)_t(\d+)_(l1|ai\d+)$", label)
    if m: return (int(m.group(1)), m.group(2)+"_"+m.group(4).rstrip('0123456789'), int(m.group(3)))
    return None

data = load_all()
print(f"Loaded {len(data)} stats files")

# Group by experiment
by_exp = defaultdict(dict)
for label, s in data.items():
    p = parse_label(label)
    if not p: continue
    exp, cond, t = p
    by_exp[exp].setdefault(cond, []).append((t, s))

for eid in sorted(by_exp): print(f"  Exp{eid}: {len(by_exp[eid])} conditions, {sum(len(v) for v in by_exp[eid].values())} trials")

# =============================================================
# EXP 1 — Config B (Full GPU) diverse stack — validates §13c Chain 17 finding
# =============================================================
def fig_exp1():
    conds = ["baseline"] + [f"cfgB_diverseN{n}" for n in [1,3,6,8,10,12]]
    xs = [0, 1, 3, 6, 8, 10, 12]
    duty_mean=[]; duty_std=[]; gap_mean=[]; gap_std=[]
    for c in conds:
        trials = [s for _, s in by_exp[1].get(c, [])]
        if not trials: duty_mean.append(0); duty_std.append(0); gap_mean.append(0); gap_std.append(0); continue
        d = [t["duty"] for t in trials]
        g = [t["gap_p95"]/1000 for t in trials]
        duty_mean.append(np.mean(d)); duty_std.append(np.std(d))
        gap_mean.append(np.mean(g)); gap_std.append(np.std(g))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, means, stds, ylabel, title in [
        (axes[0], duty_mean, duty_std, "L1 duty cycle (%)", "L1 duty vs N"),
        (axes[1], gap_mean, gap_std, "L1 gap p95 (μs)", "L1 gap p95 vs N"),
    ]:
        ax.errorbar(xs, means, yerr=stds, fmt="o-", color=COL_B, linewidth=3, markersize=13,
                    markerfacecolor="white", markeredgewidth=2.5, capsize=6)
        ax.set_xlabel("N (diverse AI containers on Full GPU)"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", loc="left", color=INK_SEC)
        ax.grid(alpha=0.5)
    fig.suptitle("Chain 19 Exp 1 · Config B (Full GPU) holds L1 baseline even under diverse 12-workload stack",
                 fontweight="bold", y=1.01, x=0.02, ha="left", fontsize=17)
    fig.text(0.02, 0.008,
             "Full GPU (108 SM) + N diverse AI containers (Qwen + Whisper + BERT + NRx + CSI + Beam repeated). "
             "Confirms Chain 17 identical-NRx finding generalizes to realistic diverse workload stack.",
             fontsize=11.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(f"{FIG}/e19_exp1_configB_diverse.png"); plt.close()
    print("saved e19_exp1_configB_diverse")

# =============================================================
# EXP 2 — AI-side kernel trace (L1 launch rate + AI aggregate rate)
# =============================================================
def fig_exp2():
    # L1 traces (labels end with _l1)
    Ns = [1, 4, 6, 8]
    l1_rate=[]; ai_agg_rate=[]
    for N in Ns:
        # L1 launch rate
        l1_labels = [k for k in data if re.match(f"^e2_N{N}_t\\d+_l1$", k)]
        l1_r = np.mean([data[k]["launch_rate"] for k in l1_labels]) if l1_labels else 0
        l1_rate.append(l1_r)
        # AI aggregate launch rate (sum of all _ai* per trial, then avg across trials)
        trials_agg = defaultdict(float)
        for k in data:
            m = re.match(f"^e2_N{N}_t(\\d+)_ai\\d+$", k)
            if m:
                trials_agg[m.group(1)] += data[k]["launch_rate"]
        ai_agg_rate.append(np.mean(list(trials_agg.values())) if trials_agg else 0)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    w = 0.35
    x = np.arange(len(Ns))
    ax.bar(x - w/2, l1_rate, w, color=COL_BASELINE, alpha=0.85, edgecolor="white", linewidth=2, label="L1 kernel launch rate")
    ax.bar(x + w/2, ai_agg_rate, w, color=COL_BAD, alpha=0.85, edgecolor="white", linewidth=2, label="AI aggregate launch rate (sum of N)")
    for i, (v_l1, v_ai) in enumerate(zip(l1_rate, ai_agg_rate)):
        ax.text(i-w/2, v_l1 + max(max(l1_rate),max(ai_agg_rate))*0.02, f"{v_l1:.0f}",
                ha="center", fontsize=11, color=COL_BASELINE, fontweight="bold")
        ax.text(i+w/2, v_ai + max(max(l1_rate),max(ai_agg_rate))*0.02, f"{v_ai:.0f}",
                ha="center", fontsize=11, color=COL_BAD, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_ylabel("kernel launch rate (kernels/sec)")
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.grid(axis="y", alpha=0.5)
    ax.set_title("Chain 19 Exp 2 · L1 vs AI aggregate launch rate — do both collapse at N=6?",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "Both L1 (profiled separately) and sum of N AI processes' launch rates. "
             "If they collapse together at N=6, MPS scheduler saturation hits all clients symmetrically.",
             fontsize=11.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/e19_exp2_ai_side_trace.png"); plt.close()
    print("saved e19_exp2_ai_side_trace")

# =============================================================
# EXP 4 — Bursty AI (steady vs burst variants)
# =============================================================
def fig_exp4():
    # Steady baseline
    steady = [s for _, s in by_exp[4].get("steady_N4", [])]
    # Burst variants: burst_kpb{K}_idle{I}_N{N}
    burst_pts = []
    for cond, trials in by_exp[4].items():
        m = re.match(r"burst_kpb(\d+)_idle(\d+)_N(\d+)", cond)
        if m:
            kpb, idle, N = int(m.group(1)), int(m.group(2)), int(m.group(3))
            for _, s in trials:
                burst_pts.append((kpb, idle, N, s["duty"], s["gap_p95"]/1000))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Left: duty
    if steady:
        baseline_duty = np.mean([s["duty"] for s in steady])
        axes[0].axhline(baseline_duty, color=INK_MUT, linestyle="--", linewidth=1.8, alpha=0.7)
        axes[0].text(0.02, baseline_duty+0.5, f"steady N=4: {baseline_duty:.1f}%", transform=axes[0].get_yaxis_transform(),
                     color=INK_SEC, fontsize=11, style="italic")
    for i, (kpb, idle, N, duty, gp95) in enumerate(burst_pts):
        col = COL_WARN if N==4 else COL_BAD
        mk = 'o' if idle==900 else 's'
        axes[0].scatter([kpb], [duty], color=col, marker=mk, s=100, alpha=0.7, edgecolor="white", linewidth=1.5)
        axes[1].scatter([kpb], [gp95], color=col, marker=mk, s=100, alpha=0.7, edgecolor="white", linewidth=1.5)
    axes[0].set_xlabel("kernels per burst"); axes[0].set_ylabel("L1 duty cycle (%)")
    axes[0].set_title("L1 duty (dot=idle 900ms, sq=idle 90ms)", fontweight="bold", loc="left", color=INK_SEC)
    axes[1].set_xlabel("kernels per burst"); axes[1].set_ylabel("L1 gap p95 (μs)")
    axes[1].set_title("L1 gap p95", fontweight="bold", loc="left", color=INK_SEC)
    axes[1].set_yscale("log")
    for ax in axes: ax.grid(alpha=0.5, which="both")
    fig.suptitle("Chain 19 Exp 4 · Bursty AI workload effect on L1 sync",
                 fontweight="bold", y=1.01, x=0.02, ha="left", fontsize=17)
    fig.text(0.02, 0.008,
             "Steady N=4 baseline (dashed line) vs bursty variants (kernels_per_burst × idle_after × N). "
             "Bursty at N=4 tests whether burst triggers momentary breakdown even in safe zone.",
             fontsize=11.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(f"{FIG}/e19_exp4_bursty.png"); plt.close()
    print("saved e19_exp4_bursty")

# =============================================================
# EXP 5 — CP AI-side scaling (L1 should stay baseline as N grows)
# =============================================================
def fig_exp5():
    Ns = [0, 6, 8, 10, 12, 16]
    duty_mean=[]; duty_std=[]; gap_mean=[]; gap_std=[]
    for N in Ns:
        c = "baseline" if N == 0 else f"cpN{N}"
        trials = [s for _, s in by_exp[5].get(c, [])]
        if not trials: duty_mean.append(0); duty_std.append(0); gap_mean.append(0); gap_std.append(0); continue
        d = [t["duty"] for t in trials]
        g = [t["gap_p95"]/1000 for t in trials]
        duty_mean.append(np.mean(d)); duty_std.append(np.std(d))
        gap_mean.append(np.mean(g)); gap_std.append(np.std(g))

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.errorbar(Ns, duty_mean, yerr=duty_std, fmt="o-", color=COL_GOOD, linewidth=3, markersize=13,
                markerfacecolor="white", markeredgewidth=2.5, capsize=6, label="L1 duty (should be flat = isolation)")
    if duty_mean[0] > 0:
        ax.axhline(duty_mean[0], color=INK_MUT, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(16, duty_mean[0]+0.5, f"L1 alone baseline: {duty_mean[0]:.1f}%", color=INK_SEC, fontsize=11, ha="right", style="italic")
    ax.set_xlabel("N (AI containers on 3g partition)"); ax.set_ylabel("L1 duty cycle (%, L1 on 4g)")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.5)
    ax.set_title("Chain 19 Exp 5 · Cross-partition L1 stays baseline as AI-side scales to N=16",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "L1 on 4g.20gb partition. AI diverse stack on 3g.20gb scaled to N=16. "
             "If L1 duty stays flat, hardware isolation holds under extreme AI partition pressure.",
             fontsize=11.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/e19_exp5_cp_scaling.png"); plt.close()
    print("saved e19_exp5_cp_scaling")

# =============================================================
# EXP 7 — Fault isolation (cp vs sp × none/sigkill/dockerkill)
# =============================================================
def fig_exp7():
    topos = ["cp", "sp"]; faults = ["none", "sigkill", "dockerkill"]
    matrix = np.zeros((2, 3))  # topo × fault
    for i, topo in enumerate(topos):
        for j, f in enumerate(faults):
            c = f"{topo}_{f}"
            trials = [s for _, s in by_exp[7].get(c, [])]
            matrix[i, j] = np.mean([t["duty"] for t in trials]) if trials else 0

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(3); w = 0.35
    ax.bar(x - w/2, matrix[0], w, color=COL_GOOD, alpha=0.85, edgecolor="white", linewidth=2, label="Cross-partition (L1 on 4g, AI on 3g)")
    ax.bar(x + w/2, matrix[1], w, color=COL_BAD, alpha=0.85, edgecolor="white", linewidth=2, label="Same-partition (L1+AI on 4g)")
    for i, f in enumerate(faults):
        for j, v in enumerate([matrix[0][i], matrix[1][i]]):
            col = COL_GOOD if j == 0 else COL_BAD
            ax.text(i + (j-0.5)*w, v + max(matrix.flatten())*0.02, f"{v:.1f}%", ha="center", fontsize=11, color=col, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(faults)
    ax.set_ylabel("L1 duty cycle (%)"); ax.set_xlabel("Fault scenario")
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.grid(axis="y", alpha=0.5)
    ax.set_title("Chain 19 Exp 7 · Fault isolation — cross-partition survives crash/dockerkill unchanged",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "AI container killed at t=15s of 30s L1 trace. If CP is truly hardware-isolated, "
             "duty cycle should stay baseline. If SP has residual after crash, MPS server state pollution.",
             fontsize=11.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/e19_exp7_fault_isolation.png"); plt.close()
    print("saved e19_exp7_fault_isolation")

# =============================================================
# EXP 9 — Config C deep N-sweep (with 3g partition L1)
# =============================================================
def fig_exp9():
    Ns = [1, 2, 3, 4, 6, 8]
    duty_off = []; duty_on = []
    for N in Ns:
        off_t = [s for _, s in by_exp[9].get(f"C_N{N}_MPSoff", [])]
        on_t = [s for _, s in by_exp[9].get(f"C_N{N}_MPSon", [])]
        duty_off.append(np.mean([t["duty"] for t in off_t]) if off_t else 0)
        duty_on.append(np.mean([t["duty"] for t in on_t]) if on_t else 0)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.axvspan(3.5, 5.5, alpha=0.08, color=COL_WARN, zorder=0)
    ax.axvspan(5.5, 8.5, alpha=0.08, color=COL_BAD, zorder=0)
    ax.plot(Ns, duty_on, "-", color=COL_MPS_ON, linewidth=3, marker="o", markersize=13, markerfacecolor="white", markeredgewidth=2.5, label="MPS on")
    ax.plot(Ns, duty_off, "-", color=COL_MPS_OFF, linewidth=3, marker="s", markersize=11, markerfacecolor="white", markeredgewidth=2.5, label="MPS off")
    for N, v in zip(Ns, duty_on):
        if N in [1, 4, 6, 8]: ax.text(N, v+1.5, f"{v:.1f}%", ha="center", fontsize=11, color=COL_MPS_ON, fontweight="bold")
    ax.set_xlabel("N (concurrent NRx on Config C 3g partition)")
    ax.set_ylabel("L1 duty cycle (%)")
    ax.set_xticks(Ns); ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.5)
    ax.set_title("Chain 19 Exp 9 · Config C (MIG 3g.20gb, 42 SM) breakdown curve",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "L1 on smallest MIG partition (3g.20gb, 42 SM). Complements Config A (56 SM) and Config B (108 SM) sweeps.",
             fontsize=11.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/e19_exp9_configC_sweep.png"); plt.close()
    print("saved e19_exp9_configC_sweep")

# =============================================================
# EXP 10 — Long-window 300s (does breakdown drift over time?)
# =============================================================
def fig_exp10():
    conds = ["long_baseline", "long_N4_MPSon", "long_N6_MPSon", "long_N8_MPSon"]
    labels = ["baseline", "N=4 MPSon", "N=6 MPSon", "N=8 MPSon"]
    colors = [COL_BASELINE, COL_GOOD, COL_WARN, COL_BAD]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(conds)); w = 0.6
    duty_mean = []
    for c in conds:
        trials = [s for _, s in by_exp[10].get(c, [])]
        duty_mean.append(np.mean([t["duty"] for t in trials]) if trials else 0)
    for i, (lab, col, d) in enumerate(zip(labels, colors, duty_mean)):
        ax.bar(i, d, w, color=col, alpha=0.85, edgecolor="white", linewidth=2)
        ax.text(i, d + max(duty_mean)*0.02, f"{d:.1f}%", ha="center", fontsize=13, color=col, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("L1 duty cycle (%, 300s window mean)")
    ax.grid(axis="y", alpha=0.5)
    ax.set_title("Chain 19 Exp 10 · Long-window 300s — does breakdown drift over time?",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "10× longer than default 30s trace. If values match 30s trace results (Chain 17), steady-state assumption is valid.",
             fontsize=11.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/e19_exp10_long_window.png"); plt.close()
    print("saved e19_exp10_long_window")

# =============================================================
# EXP 11 — MPS thread% × N heatmap
# =============================================================
def fig_exp11():
    pcts = [30, 50, 70, 100]; Ns = [4, 6, 8]
    matrix = np.zeros((len(pcts), len(Ns)))
    for i, pct in enumerate(pcts):
        for j, N in enumerate(Ns):
            c = f"pct{pct}_N{N}"
            trials = [s for _, s in by_exp[11].get(c, [])]
            matrix[i, j] = np.mean([t["duty"] for t in trials]) if trials else 0

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=35)
    ax.set_xticks(range(len(Ns))); ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_yticks(range(len(pcts))); ax.set_yticklabels([f"pct={p}%" for p in pcts])
    for i in range(len(pcts)):
        for j in range(len(Ns)):
            v = matrix[i, j]
            color = "white" if v < 15 else INK
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=14, color=color, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, label="L1 duty cycle (%)")
    ax.set_title("Chain 19 Exp 11 · CUDA_MPS_ACTIVE_THREAD_PERCENTAGE × N heatmap",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "MPS thread% cap for AI clients × N concurrent processes. Reveals best combination for maximum L1 duty.",
             fontsize=11.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/e19_exp11_pct_N_heatmap.png"); plt.close()
    print("saved e19_exp11_pct_N_heatmap")

# =============================================================
# EXP 12 — Multi-GPU baseline (L1 GPU0, AI GPU1)
# =============================================================
def fig_exp12():
    Ns = [1, 4, 6, 8]
    duty_mean = []; duty_std = []
    for N in Ns:
        trials = [s for _, s in by_exp[12].get(f"multiGPU_N{N}", [])]
        if trials:
            d = [t["duty"] for t in trials]
            duty_mean.append(np.mean(d)); duty_std.append(np.std(d))
        else:
            duty_mean.append(0); duty_std.append(0)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.errorbar(Ns, duty_mean, yerr=duty_std, fmt="o-", color=COL_GOOD, linewidth=3, markersize=13, markerfacecolor="white", markeredgewidth=2.5, capsize=6)
    ax.set_xlabel("N (AI processes on GPU 1)"); ax.set_ylabel("L1 duty cycle (%, L1 on GPU 0)")
    ax.set_xticks(Ns); ax.grid(alpha=0.5)
    ax.set_title("Chain 19 Exp 12 · Multi-GPU baseline — L1 GPU 0 vs AI GPU 1 (zero interference)",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "L1 on GPU 0, AI on GPU 1 — should have zero shared driver state or memory bus contention.",
             fontsize=11.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/e19_exp12_multi_gpu.png"); plt.close()
    print("saved e19_exp12_multi_gpu")

# =============================================================
# EXP 13 — L1 cell count sweep (does breakdown depend on L1 workload size?)
# =============================================================
def fig_exp13():
    cells = [5, 10, 20, 40]
    duty_alone = []; duty_stress = []
    for c in cells:
        alone = [s for _, s in by_exp[13].get(f"cells{c}_alone", [])]
        stress = [s for _, s in by_exp[13].get(f"cells{c}_N6sp", [])]
        duty_alone.append(np.mean([t["duty"] for t in alone]) if alone else 0)
        duty_stress.append(np.mean([t["duty"] for t in stress]) if stress else 0)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(cells, duty_alone, "-", color=COL_BASELINE, linewidth=3, marker="o", markersize=13, markerfacecolor="white", markeredgewidth=2.5, label="L1 alone")
    ax.plot(cells, duty_stress, "-", color=COL_BAD, linewidth=3, marker="s", markersize=11, markerfacecolor="white", markeredgewidth=2.5, label="L1 + 6× NRx (SP breakdown)")
    for c, va, vs in zip(cells, duty_alone, duty_stress):
        ax.text(c, va+1.5, f"{va:.1f}%", ha="center", fontsize=11, color=COL_BASELINE, fontweight="bold")
        ax.text(c, vs+1.5, f"{vs:.1f}%", ha="center", fontsize=11, color=COL_BAD, fontweight="bold")
    ax.set_xlabel("L1 cell count"); ax.set_ylabel("L1 duty cycle (%)")
    ax.set_xticks(cells); ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.5)
    ax.set_title("Chain 19 Exp 13 · L1 cell count sweep — does breakdown scale with workload size?",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "L1 workload size (cells): 5/10/20/40. Both L1 alone and L1 + 6× NRx SP breakdown measured. "
             "Reveals if breakdown is workload-agnostic (constant offset) or scales with L1 size.",
             fontsize=11.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/e19_exp13_cell_sweep.png"); plt.close()
    print("saved e19_exp13_cell_sweep")

# =============================================================
# Master summary — all Chain 19 conditions in one dashboard
# =============================================================
def fig_master_summary():
    # Total conditions per experiment
    exp_counts = {eid: sum(len(v) for v in by_exp[eid].values()) for eid in sorted(by_exp)}
    fig, ax = plt.subplots(figsize=(13, 5))
    exps = list(exp_counts.keys()); counts = [exp_counts[e] for e in exps]
    ax.bar(exps, counts, color=COL_A, alpha=0.85, edgecolor="white", linewidth=2)
    for e, c in zip(exps, counts):
        ax.text(e, c+2, str(c), ha="center", fontsize=12, color=COL_A, fontweight="bold")
    ax.set_xlabel("Chain 19 Experiment"); ax.set_ylabel("Number of nsys captures")
    ax.set_xticks(exps); ax.set_xticklabels([f"Exp{e}" for e in exps])
    ax.grid(axis="y", alpha=0.5)
    ax.set_title(f"Chain 19 · Total {sum(counts)} captures across 11 quantitative experiments",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "Exp 3 (NCU warp stall) and Exp 6 (CUDA graph JSON) not shown — non-nsys outputs.",
             fontsize=11.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/e19_master_summary.png"); plt.close()
    print("saved e19_master_summary")

# Run all
fig_master_summary()
fig_exp1()
fig_exp2()
fig_exp4()
fig_exp5()
fig_exp7()
fig_exp9()
fig_exp10()
fig_exp11()
fig_exp12()
fig_exp13()

# Save unified summary JSON
summary = {
    "n_conditions": len(data),
    "by_experiment": {eid: {c: [t["duty"] for _, t in trials] for c, trials in conds.items()}
                      for eid, conds in by_exp.items()},
}
with open(f"{OUT}/chain19_summary.json", "w") as fp:
    json.dump(summary, fp, indent=2)
print(f"\nWrote chain19_summary.json ({len(data)} conditions)")
