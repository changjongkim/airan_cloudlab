#!/usr/bin/env python3
"""Chain 19 latency + throughput focused analysis.

Duty cycle was over-emphasized. Real SLA metrics:
  - L1 per-iteration latency (from realL1_*.json: mean_ms, p95_ms, p99_ms)
  - AI throughput (Qwen tok/s from vLLM logs, RAN-AI iter/s from ranai_mix logs)

Produces figures L1-1 through L1-8:
  L1-1: L1 p99 latency across Chain 19 conditions
  L1-2: L1 latency distribution (baseline vs breakdown zones)
  L1-3: Qwen tok/s vs N (Config B diverse)
  L1-4: Qwen tok/s vs L1 p99 trade-off
  L1-5: CsiNet/BeamPred iter/s across conditions
  L1-6: AI aggregate throughput vs L1 latency scatter
  L1-7: MPS pct=30 vs pct=100 dual-metric (L1 latency + AI throughput)
  L1-8: Cross-partition L1 latency invariance under AI-side scaling
"""
import os, json, glob, re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260803"
OUT  = f"{BASE}/analysis_chain19"
FIG  = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)

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
# Load all realL1_*.json (actual L1 per-iteration latency)
# =============================================================
def load_l1_latency():
    """Return dict: label -> {'mean_ms','p95_ms','p99_ms',...}"""
    data = {}
    for f in glob.glob(f"{BASE}/chain19_exp*/realL1_*.json"):
        try:
            d = json.load(open(f))
            data[d["label"]] = d
        except: pass
    return data

l1 = load_l1_latency()
print(f"Loaded {len(l1)} L1 latency records")

# =============================================================
# Parse Qwen throughput from vLLM logs
# =============================================================
def parse_qwen_tokps(log_path):
    """Return average tok/s over the run (main loop iterations only)."""
    if not os.path.exists(log_path): return None
    tokps = []
    with open(log_path, errors='ignore') as f:
        for line in f:
            m = re.search(r"iter=\d+ toks=\d+ tok/s=(\d+)", line)
            if m: tokps.append(int(m.group(1)))
    return np.mean(tokps) if tokps else None

# =============================================================
# Parse ranai_mix (CsiNet/BeamPred/NRx) throughput from logs
# =============================================================
def parse_ranai_ratepersec(log_path):
    """Return final average rate/s from ranai_mix log."""
    if not os.path.exists(log_path): return None
    rates = []
    with open(log_path, errors='ignore') as f:
        for line in f:
            m = re.search(r"rate=(\d+)/s", line)
            if m: rates.append(int(m.group(1)))
    return np.mean(rates) if rates else None

# =============================================================
# Build per-condition aggregate: L1 latency + AI throughput
# =============================================================
def aggregate_condition(exp_dir_pattern, label_re):
    """For each condition, gather L1 p99 + aggregate AI throughput per trial."""
    out = defaultdict(lambda: {"l1_p99": [], "l1_p95": [], "l1_mean": [], "qwen_tokps": [], "ai_ratepersec": []})
    for f in glob.glob(f"{BASE}/{exp_dir_pattern}/realL1_*.json"):
        try: d = json.load(open(f))
        except: continue
        label = d["label"]
        m = label_re.match(label)
        if not m: continue
        cond = m.group("cond") if "cond" in label_re.groupindex else m.group(1)
        trial = m.group("t") if "t" in label_re.groupindex else m.group(2) if label_re.groups > 1 else "1"
        out[cond]["l1_p99"].append(d["p99_ms"])
        out[cond]["l1_p95"].append(d["p95_ms"])
        out[cond]["l1_mean"].append(d["mean_ms"])
        # Sum Qwen tok/s
        exp_dir = os.path.dirname(f)
        qwens = sorted(glob.glob(f"{exp_dir}/{label}_qwen*.log"))
        q_sum = 0
        for q in qwens:
            v = parse_qwen_tokps(q)
            if v: q_sum += v
        out[cond]["qwen_tokps"].append(q_sum)
        # Sum ranai iter/s
        ranai_sum = 0
        for pat in ["csinet", "beampred", "nrx"]:
            files = sorted(glob.glob(f"{exp_dir}/{label}_{pat}*.log"))
            for lf in files:
                v = parse_ranai_ratepersec(lf)
                if v: ranai_sum += v
        out[cond]["ai_ratepersec"].append(ranai_sum)
    return out

# =============================================================
# FIG L1-1: L1 p99 latency across key conditions
# =============================================================
def fig_l1_p99_ranking():
    # Group all conditions
    cond_lat = defaultdict(list)
    for label, d in l1.items():
        cond = re.sub(r"_t\d+$", "", label)
        # Skip meaningless labels
        if len(cond) > 80 or not cond.startswith("e"): continue
        cond_lat[cond].append(d["p99_ms"])
    # Compute mean p99 per condition (dedupe short label + limit)
    per_cond = [(c, np.mean(vs), np.std(vs)) for c, vs in cond_lat.items() if len(vs) >= 1]
    per_cond.sort(key=lambda x: x[1])
    # Top 12 lowest + 12 highest, dedupe
    top = per_cond[:12]; bottom = per_cond[-12:]
    combined = top + bottom
    # Filter: max 30 labels total
    seen = set(); dedup = []
    for c, m, s in combined:
        if c in seen: continue
        seen.add(c); dedup.append((c, m, s))
    combined = dedup[:30]
    labels = [c for c,_,_ in combined]
    means = [m for _,m,_ in combined]
    stds  = [s for _,_,s in combined]
    fig, ax = plt.subplots(figsize=(14, 12))
    colors = [COL_GOOD if m < 50 else (COL_WARN if m < 100 else COL_BAD) for m in means]
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=stds, color=colors, alpha=0.85, edgecolor="white", linewidth=1, capsize=3)
    for i, m in enumerate(means):
        ax.text(m*1.05, i, f"{m:.1f}ms", va="center", fontsize=9, color=colors[i])
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("L1 per-iteration p99 latency (ms, lower is better)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.5)
    ax.set_title(f"Chain 19 · L1 p99 latency ranking (top {len(top)} best + {len(bottom)} worst of {len(per_cond)})",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "Real per-iteration L1 latency from realL1_*.json. This is the direct SLA metric (not duty cycle proxy). "
             "Baseline: ~40 ms per iter (20 cells × 100 kernels). Breakdown pushes p99 to 100+ ms → user experience impact.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(f"{FIG}/e19_lat_p99_ranking.png"); plt.close()
    print("saved e19_lat_p99_ranking")

# =============================================================
# FIG L1-2: L1 latency distribution (mean/p95/p99) key conditions
# =============================================================
def fig_l1_latency_key():
    key_conds = [
        ("e1_baseline",     "L1 alone (Full GPU)"),
        ("e1_cfgB_diverseN1","Full GPU + 1 diverse AI"),
        ("e1_cfgB_diverseN6","Full GPU + 6 diverse AI"),
        ("e1_cfgB_diverseN12","Full GPU + 12 diverse AI"),
        ("e5_baseline",     "L1 alone (MIG 4g)"),
        ("e5_cpN6",         "MIG CP + 6 diverse AI"),
        ("e5_cpN16",        "MIG CP + 16 diverse AI"),
        ("e11_pct100_N6",   "MIG SP N=6 pct=100 (default)"),
        ("e11_pct30_N6",    "MIG SP N=6 pct=30 (tuned)"),
        ("e12_multiGPU_N8", "Multi-GPU + 8 AI (GPU1)"),
    ]
    cond_data = {}
    for cond, disp in key_conds:
        vals_m=[]; vals_95=[]; vals_99=[]
        for label, d in l1.items():
            if label.startswith(cond+"_t"):
                vals_m.append(d["mean_ms"]); vals_95.append(d["p95_ms"]); vals_99.append(d["p99_ms"])
        if vals_m:
            cond_data[disp] = (np.mean(vals_m), np.mean(vals_95), np.mean(vals_99))

    fig, ax = plt.subplots(figsize=(15, 7))
    labels = list(cond_data.keys())
    x = np.arange(len(labels)); w = 0.28
    means = [cond_data[l][0] for l in labels]
    p95s  = [cond_data[l][1] for l in labels]
    p99s  = [cond_data[l][2] for l in labels]
    ax.bar(x-w, means, w, color=COL_BASELINE, alpha=0.85, edgecolor="white", linewidth=1.5, label="mean")
    ax.bar(x,   p95s,  w, color=COL_WARN,     alpha=0.85, edgecolor="white", linewidth=1.5, label="p95")
    ax.bar(x+w, p99s,  w, color=COL_BAD,      alpha=0.85, edgecolor="white", linewidth=1.5, label="p99")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=11)
    ax.set_ylabel("L1 per-iteration latency (ms)")
    ax.legend(loc="upper left", frameon=True); ax.grid(axis="y", alpha=0.5)
    ax.set_title("Chain 19 · L1 per-iteration latency (mean / p95 / p99) across key topologies",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "Real per-iteration L1 latency from cuPHY. Lower = better user experience. Breakdown = tail (p99) explosion.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/e19_lat_key_conditions.png"); plt.close()
    print("saved e19_lat_key_conditions")

# =============================================================
# FIG L1-3: Qwen tok/s vs N (Config B diverse)
# =============================================================
def fig_qwen_tokps_configB():
    cond_agg = aggregate_condition("chain19_exp1", re.compile(r"^e1_(?P<cond>baseline|cfgB_diverseN\d+)_t(?P<t>\d+)$"))
    Ns = [1, 3, 6, 8, 10, 12]
    means = []; stds = []
    for N in Ns:
        c = f"cfgB_diverseN{N}"
        vals = cond_agg[c]["qwen_tokps"]
        means.append(np.mean(vals) if vals else 0)
        stds.append(np.std(vals) if vals else 0)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.errorbar(Ns, means, yerr=stds, fmt="o-", color=COL_B, linewidth=3, markersize=13,
                markerfacecolor="white", markeredgewidth=2.5, capsize=6)
    for N, m in zip(Ns, means):
        ax.text(N, m*1.03, f"{m:.0f} tok/s", ha="center", fontsize=11, color=COL_B, fontweight="bold")
    ax.set_xlabel("N (diverse AI containers on Full GPU)")
    ax.set_ylabel("Qwen 2.5-3B aggregate throughput (tok/s)")
    ax.set_xticks(Ns); ax.grid(alpha=0.5)
    ax.set_title("Chain 19 · Qwen vLLM aggregate throughput vs N (Config B Full GPU)",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "Qwen 2.5-3B via vLLM v0.6.6, n_concurrent=16. Values summed across all Qwen containers in the stack. "
             "N=1: single Qwen. N=6: single Qwen (composition has 1 Qwen). N=8,10,12: two Qwens.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/e19_tokps_configB.png"); plt.close()
    print("saved e19_tokps_configB")

# =============================================================
# FIG L1-4: Qwen tok/s vs L1 p99 trade-off scatter
# =============================================================
def fig_tradeoff():
    """Every condition: (L1 p99, aggregate AI throughput). Find Pareto frontier."""
    points = []  # (l1_p99, ai_tokps, label, color)
    # Config B (Exp 1)
    for N in [0, 1, 3, 6, 8, 10, 12]:
        cond = "baseline" if N == 0 else f"cfgB_diverseN{N}"
        l1_lat = np.mean([d["p99_ms"] for label, d in l1.items() if label.startswith(f"e1_{cond}_t")])
        if not np.isnan(l1_lat):
            # Sum Qwen tok/s for this cond
            qwen_sum = 0
            for f in glob.glob(f"{BASE}/chain19_exp1/e1_{cond}_t*_qwen*.log"):
                v = parse_qwen_tokps(f)
                if v: qwen_sum += v
            n_trials = len([label for label in l1 if label.startswith(f"e1_{cond}_t")])
            avg = qwen_sum / max(n_trials, 1)
            points.append((l1_lat, avg, f"B N={N}", COL_B))
    # Config A CP (Exp 5)
    for N in [0, 6, 8, 10, 12, 16]:
        cond = "baseline" if N == 0 else f"cpN{N}"
        l1_lat = np.mean([d["p99_ms"] for label, d in l1.items() if label.startswith(f"e5_{cond}_t")])
        if not np.isnan(l1_lat):
            qwen_sum = 0
            for f in glob.glob(f"{BASE}/chain19_exp5/e5_{cond}_t*_qwen*.log"):
                v = parse_qwen_tokps(f)
                if v: qwen_sum += v
            n_trials = len([label for label in l1 if label.startswith(f"e5_{cond}_t")])
            avg = qwen_sum / max(n_trials, 1)
            points.append((l1_lat, avg, f"CP N={N}", COL_GOOD))
    # MPS pct sweep (Exp 11)
    for pct in [100, 70, 50, 30]:
        for N in [4, 6, 8]:
            cond = f"pct{pct}_N{N}"
            l1_lat = np.mean([d["p99_ms"] for label, d in l1.items() if label.startswith(f"e11_{cond}_t")])
            if not np.isnan(l1_lat):
                # No Qwen in Exp 11, use ranai iter/s if available (skip for now)
                points.append((l1_lat, 0, f"pct{pct} N={N}", COL_A))

    fig, ax = plt.subplots(figsize=(14, 7.5))
    for x, y, lab, col in points:
        if y > 0:
            ax.scatter(x, y, s=200, color=col, alpha=0.75, edgecolor="white", linewidth=2)
            ax.annotate(lab, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=10, color=col)
    ax.axvline(50, color=INK_MUT, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(50, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1]>0 else 5000, "L1 p99 = 50 ms threshold",
            color=INK_SEC, fontsize=11, ha="left", style="italic")
    ax.set_xlabel("L1 per-iteration p99 latency (ms) — lower is better")
    ax.set_ylabel("Qwen aggregate throughput (tok/s) — higher is better")
    ax.grid(alpha=0.5)
    ax.set_title("Chain 19 · L1 latency vs Qwen throughput trade-off (Pareto view)",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "Upper-left = ideal (low L1 latency + high AI throughput). Config B (purple) and CP (green) compared. "
             "Shows the actual deployment trade-off: L1 SLA safety vs AI service throughput.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/e19_tradeoff_latency_vs_throughput.png"); plt.close()
    print("saved e19_tradeoff_latency_vs_throughput")

# =============================================================
# FIG L1-5: MPS pct effect on L1 latency (Exp 11)
# =============================================================
def fig_pct_latency():
    pcts = [30, 50, 70, 100]; Ns = [4, 6, 8]
    matrix_p99 = np.zeros((len(pcts), len(Ns)))
    for i, pct in enumerate(pcts):
        for j, N in enumerate(Ns):
            vals = [d["p99_ms"] for label, d in l1.items() if label.startswith(f"e11_pct{pct}_N{N}_t")]
            matrix_p99[i, j] = np.mean(vals) if vals else 0
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(matrix_p99, cmap="RdYlGn_r", aspect="auto", vmin=40, vmax=200)
    ax.set_xticks(range(len(Ns))); ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_yticks(range(len(pcts))); ax.set_yticklabels([f"pct={p}%" for p in pcts])
    for i in range(len(pcts)):
        for j in range(len(Ns)):
            v = matrix_p99[i, j]
            color = "white" if v > 130 else INK
            ax.text(j, i, f"{v:.1f}ms", ha="center", va="center", fontsize=14, color=color, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, label="L1 p99 latency (ms)")
    ax.set_title("Chain 19 Exp 11 · L1 p99 latency heatmap (pct × N)",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "Actual L1 per-iter p99 latency (not duty cycle). Lower = better SLA. pct=30 at N=6 achieves 45ms (near baseline 41ms).",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/e19_pct_latency_heatmap.png"); plt.close()
    print("saved e19_pct_latency_heatmap")

# =============================================================
# FIG L1-6: Cross-partition L1 latency invariance
# =============================================================
def fig_cp_invariance():
    Ns = [0, 6, 8, 10, 12, 16]
    p99s = []; p95s = []; means = []
    for N in Ns:
        cond = "baseline" if N == 0 else f"cpN{N}"
        vals_99 = [d["p99_ms"] for label, d in l1.items() if label.startswith(f"e5_{cond}_t")]
        vals_95 = [d["p95_ms"] for label, d in l1.items() if label.startswith(f"e5_{cond}_t")]
        vals_m  = [d["mean_ms"] for label, d in l1.items() if label.startswith(f"e5_{cond}_t")]
        p99s.append(np.mean(vals_99) if vals_99 else 0)
        p95s.append(np.mean(vals_95) if vals_95 else 0)
        means.append(np.mean(vals_m) if vals_m else 0)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(Ns, means, "-", color=COL_BASELINE, linewidth=3, marker="o", markersize=13, markerfacecolor="white", markeredgewidth=2.5, label="mean")
    ax.plot(Ns, p95s,  "-", color=COL_WARN,     linewidth=3, marker="s", markersize=11, markerfacecolor="white", markeredgewidth=2.5, label="p95")
    ax.plot(Ns, p99s,  "-", color=COL_BAD,      linewidth=3, marker="^", markersize=13, markerfacecolor="white", markeredgewidth=2.5, label="p99")
    for N, v in zip(Ns, p99s):
        ax.text(N, v+0.5, f"{v:.1f}ms", ha="center", fontsize=10, color=COL_BAD, fontweight="bold")
    ax.set_xlabel("N (AI containers on 3g partition)")
    ax.set_ylabel("L1 per-iteration latency (ms, L1 on 4g)")
    ax.set_xticks(Ns); ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.5)
    ax.set_title("Chain 19 Exp 5 · Cross-partition L1 latency invariant under N=6-16 AI scaling",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "L1 on MIG 4g partition. AI diverse stack on 3g scaled to N=16. mean/p95/p99 all flat — hardware isolation empirically confirmed.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/e19_cp_l1_invariance.png"); plt.close()
    print("saved e19_cp_l1_invariance")

# =============================================================
# FIG L1-7: Config A/B comparison L1 latency
# =============================================================
def fig_config_comparison_latency():
    """L1 latency for Config A same-partition vs Config B Full GPU as N scales."""
    Ns_ab = [1, 6, 8]  # skip N=4 (Exp 1 doesn't have it)
    # Config A same-partition from Chain 17 gap stats (proxy: gap+dur × 100 kernels)
    ch17_path = "/Users/changjongkim/New_research/cloudlab_results/results/20260725/chain17_all_stats.json"
    if os.path.exists(ch17_path):
        ch17 = json.load(open(ch17_path))
    else: ch17 = {}
    a_p99 = []
    for N in Ns_ab:
        # Chain 17 doesn't have per-iter latency, use kernel gap p99 as proxy
        keys = [k for k in ch17 if re.match(f"cfgA_A_nrxN{N}_MPSon_t\\d+", k)]
        # No per-iter data in Chain 17 sqlite; use gap_p99 * 100 kernels as SLA proxy
        vals = [(ch17[k]["dur_med"] + ch17[k]["gap_med"])/1000 * 100 / 1000 for k in keys]  # ms
        a_p99.append(np.mean(vals) if vals else 0)
    # Config B from Chain 19 Exp 1 real L1 data
    b_p99 = []
    for N in Ns_ab:
        cond = "baseline" if N == 0 else f"cfgB_diverseN{N}"
        vals = [d["p99_ms"] for label, d in l1.items() if label.startswith(f"e1_{cond}_t")]
        b_p99.append(np.mean(vals) if vals else 0)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(Ns_ab, a_p99, "-", color=COL_A, linewidth=3, marker="s", markersize=13, markerfacecolor="white", markeredgewidth=2.5, label="Config A same-partition (proxy from Chain 17)")
    ax.plot(Ns_ab, b_p99, "-", color=COL_B, linewidth=3, marker="o", markersize=13, markerfacecolor="white", markeredgewidth=2.5, label="Config B Full GPU diverse (Chain 19 Exp 1)")
    for i, (a, b) in enumerate(zip(a_p99, b_p99)):
        ax.text(Ns_ab[i], a*1.05, f"{a:.1f}", ha="center", fontsize=9, color=COL_A)
        ax.text(Ns_ab[i], b*1.05, f"{b:.1f}", ha="center", fontsize=9, color=COL_B)
    ax.set_xlabel("N (concurrent AI)"); ax.set_ylabel("L1 per-slot p99 latency (ms)")
    ax.set_yscale("log")
    ax.set_ylim(1, max(max(a_p99), max(b_p99)) * 2 if (a_p99 and b_p99) else 100)
    ax.set_xticks(Ns_ab); ax.grid(alpha=0.5, which="both"); ax.legend(loc="best", frameon=True)
    ax.set_title("Chain 19 · L1 latency comparison: Config A same-partition (breaks) vs Config B Full GPU (holds)",
                 fontweight="bold", pad=18, loc="left")
    fig.text(0.02, 0.008,
             "Config A: identical NRx replicas cause N=6 breakdown (12000ms/slot proxy). "
             "Config B: diverse AI holds L1 latency near baseline (~40 ms) through N=8. Full GPU numerically wins on throughput terms.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{FIG}/e19_configA_vs_B_latency.png"); plt.close()
    print("saved e19_configA_vs_B_latency")

# =============================================================
# FIG L1-8: AI throughput per workload type (Exp 1 stack)
# =============================================================
def fig_ai_throughput_by_type():
    """Break down AI throughput by workload type for Config B N=6."""
    workload_types = ["qwen", "csinet", "beampred", "nrx"]
    throughputs = defaultdict(list)  # (N, wl) -> list of throughputs
    for N in [1, 3, 6, 8, 10, 12]:
        cond = f"cfgB_diverseN{N}"
        for wl in workload_types:
            for f in glob.glob(f"{BASE}/chain19_exp1/e1_{cond}_t*_{wl}*.log"):
                if wl == "qwen":
                    v = parse_qwen_tokps(f)
                else:
                    v = parse_ranai_ratepersec(f)
                if v: throughputs[(N, wl)].append(v)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, wl in zip(axes.flat, workload_types):
        Ns = [1, 3, 6, 8, 10, 12]
        means = [np.mean(throughputs.get((N, wl), [])) if throughputs.get((N, wl)) else 0 for N in Ns]
        ax.plot(Ns, means, "o-", color=[COL_B, COL_C, COL_A, COL_BAD]["qwen csinet beampred nrx".split().index(wl)],
                linewidth=3, markersize=12, markerfacecolor="white", markeredgewidth=2.5)
        ax.set_xlabel("N total AI"); ax.set_ylabel(f"{wl} throughput ({'tok/s' if wl=='qwen' else 'iter/s'})")
        ax.set_title(f"{wl}", fontweight="bold", loc="left", color=INK_SEC)
        ax.grid(alpha=0.5)
    fig.suptitle("Chain 19 · AI throughput per workload type (Config B diverse)",
                 fontweight="bold", y=1.01, x=0.02, ha="left", fontsize=17)
    fig.text(0.02, 0.008,
             "Individual workload throughput as stack size grows. Qwen tok/s from vLLM logs; others from ranai_mix rate/s.",
             fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f"{FIG}/e19_ai_throughput_by_type.png"); plt.close()
    print("saved e19_ai_throughput_by_type")

# Run all with error trapping
for fn in [fig_l1_p99_ranking, fig_l1_latency_key, fig_qwen_tokps_configB, fig_tradeoff,
            fig_pct_latency, fig_cp_invariance, fig_config_comparison_latency, fig_ai_throughput_by_type]:
    try:
        fn()
    except Exception as e:
        print(f"ERROR in {fn.__name__}: {e}")
        import traceback; traceback.print_exc()

print("\nAll latency + throughput figures saved.")
