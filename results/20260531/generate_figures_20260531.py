#!/usr/bin/env python3
"""20260531 — Comprehensive figure generator (36 figures).

Stories:
  T1) Tier1 baselines & phases (01-06)
  T2) L1 multi-AI extended matrix M5-M12 (07-12)
  T3) AI per-op latency: alone vs with_l1 (13-18)
  T4) NCU per-kernel HW metrics (19-22)
  T5) NSYS v3 inter-kernel gap matrix (23-26)
  T6) Deep-dive A/B/C/D/Dv2/E + saturation summary (27-34)
  T7) P3/P5 sweep summary (35-36)

Usage: python3 generate_figures_20260531.py
"""
import json, glob, re, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'savefig.dpi': 140,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.8,
    'lines.markersize': 7,
})

ROOT = Path(__file__).parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# ===========================================================================
# Data loaders
# ===========================================================================
def parse_log_stats(logpath):
    """Parse [realL1] mean=X p95=Y p99=Z miss1ms=A/B from a log file. Returns list of runs."""
    runs = []
    if not Path(logpath).exists():
        return runs
    pat = re.compile(r"mean=([0-9.]+)ms\s+p95=([0-9.]+)ms\s+p99=([0-9.]+)ms")
    for line in Path(logpath).read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            runs.append({"mean": float(m.group(1)), "p95": float(m.group(2)), "p99": float(m.group(3))})
    return runs


def load_dir_runs(d):
    """Find run_*.log or run_*.json files in dir d, aggregate stats."""
    runs = []
    d = Path(d)
    if not d.exists(): return runs
    # Prefer JSON if present (cleaner / more accurate)
    json_files = sorted(d.glob("run_*.json"))
    if json_files:
        for jf in json_files:
            try:
                data = json.loads(jf.read_text())
                if "mean_ms" in data and "p99_ms" in data:
                    runs.append({"mean": data["mean_ms"], "p95": data.get("p95_ms", 0), "p99": data["p99_ms"]})
            except Exception:
                pass
    else:
        for log in sorted(d.glob("run_*.log")):
            runs.extend(parse_log_stats(log))
        # also try run_*_l1.log (p5_sustained)
        for log in sorted(d.glob("run_*_l1.log")):
            runs.extend(parse_log_stats(log))
    return runs


def aggregate(runs):
    if not runs:
        return None
    return {
        "n": len(runs),
        "mean_mean": np.mean([r["mean"] for r in runs]),
        "mean_sd": np.std([r["mean"] for r in runs]),
        "p99_mean": np.mean([r["p99"] for r in runs]),
        "p99_sd": np.std([r["p99"] for r in runs]),
        "p95_mean": np.mean([r["p95"] for r in runs]),
        "p95_sd": np.std([r["p95"] for r in runs]),
    }


# ===========================================================================
# Color & style helpers
# ===========================================================================
def color_for(label):
    """Consistent color per workload type."""
    lc = label.lower()
    if "alone" in lc or "baseline" in lc: return "#666666"
    if "chanpred" in lc: return "#E74C3C"  # red
    if "neuralrx" in lc or "neuralRx" in label: return "#9B59B6"  # purple
    if "resnet" in lc: return "#3498DB"  # blue
    if "qwen" in lc: return "#E67E22"  # orange
    if "xapp" in lc: return "#16A085"  # teal
    if "forecast" in lc: return "#1ABC9C"  # cyan
    if "sat_compute" in lc or "sat compute" in lc: return "#F39C12"  # gold
    if "sat_hbm" in lc or "sat hbm" in lc: return "#D35400"  # dark orange
    if "memcpy" in lc or "d2d" in lc: return "#8E44AD"  # violet
    if "h2d" in lc: return "#C0392B"  # darker red
    if "gemm" in lc: return "#27AE60"  # green
    return "#34495E"  # default slate


def save(fig, idx, name):
    path = OUT / f"fig_{idx:02d}_{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ fig_{idx:02d}_{name}.png")


# ===========================================================================
# Group 1: Tier1 baselines & phases (figs 01-06)
# ===========================================================================
def fig_01_partition_baselines():
    """01: MIG partition baseline scaling (no-MIG / 7g / 4g / 3g / 2g)."""
    dirs = {
        "fullGPU (no-MIG)": ROOT / "n20_baseline_fullGPU",
        "7g (single MIG)": ROOT / "n20_baseline_7g_single",
        "4g": ROOT / "n20_baseline_4g_alone",
        "3g": ROOT / "n20_baseline_3g_alone",
        "2g": ROOT / "n20_baseline_2g_alone",
    }
    data = []
    for lbl, d in dirs.items():
        agg = aggregate(load_dir_runs(d))
        if agg:
            data.append((lbl, agg))
    if not data: return
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(data))
    means = [d[1]["mean_mean"] for d in data]
    mean_sds = [d[1]["mean_sd"] for d in data]
    p99s = [d[1]["p99_mean"] for d in data]
    p99_sds = [d[1]["p99_sd"] for d in data]
    w = 0.35
    ax.bar(x - w/2, means, w, yerr=mean_sds, label='Mean', color='#3498DB', capsize=4)
    ax.bar(x + w/2, p99s, w, yerr=p99_sds, label='p99', color='#E74C3C', capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], rotation=15)
    ax.set_ylabel("L1 frame time (ms)")
    ax.set_title("Fig 01 — MIG partition baselines (cuPHY L1 alone, no AI)")
    ax.legend(loc='upper right')
    for i, (m, p) in enumerate(zip(means, p99s)):
        ax.text(i - w/2, m + 0.5, f"{m:.1f}", ha='center', fontsize=9)
        ax.text(i + w/2, p + 0.5, f"{p:.1f}", ha='center', fontsize=9)
    save(fig, 1, "partition_baselines")


def fig_02_phase1_qwen():
    """02: Phase 1 LLM variants (qwen small / 7b prefill / decode / stress)."""
    dirs = {
        "alone (3g)": ROOT / "n20_baseline_3g_alone",
        "qwen_small": ROOT / "n20_phase1_qwen_small",
        "qwen7b prefill": ROOT / "n20_phase1_qwen7b_prefill",
        "qwen7b decode": ROOT / "n20_phase1_qwen7b_decode",
        "qwen7b stress": ROOT / "n20_phase1_qwen7b_stress",
    }
    data = []
    for lbl, d in dirs.items():
        agg = aggregate(load_dir_runs(d))
        if agg: data.append((lbl, agg))
    if not data: return
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(data))
    p99s = [d[1]["p99_mean"] for d in data]
    sds = [d[1]["p99_sd"] for d in data]
    colors = [color_for(d[0]) for d in data]
    ax.bar(x, p99s, yerr=sds, color=colors, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], rotation=20)
    ax.set_ylabel("L1 p99 (ms)")
    ax.set_title("Fig 02 — Phase 1: LLM co-tenant (3g L1 + LLM on 2g)")
    base = p99s[0]
    for i, p in enumerate(p99s):
        pct = (p - base) / base * 100
        ax.text(i, p + 0.3, f"{p:.1f}\n({pct:+.0f}%)", ha='center', fontsize=9)
    ax.axhline(base, ls='--', color='gray', alpha=0.5, label=f'alone {base:.1f}ms')
    ax.legend()
    save(fig, 2, "phase1_qwen")


def fig_03_phase2_multiAI():
    """03: Phase 2 multi-AI scenarios (M1-M4)."""
    dirs = {
        "alone (3g)": ROOT / "n20_baseline_3g_alone",
        "M1 3way balanced": ROOT / "n20_phase2_M1_3way_balanced",
        "M2 3way L1 small": ROOT / "n20_phase2_M2_3way_L1small",
        "M3 3way asym": ROOT / "n20_phase2_M3_3way_asym",
        "M4 4way 1L1+3AI": ROOT / "n20_phase2_M4_4way_1L1_3AI",
    }
    data = [(l, aggregate(load_dir_runs(d))) for l, d in dirs.items()]
    data = [(l, a) for l, a in data if a]
    if not data: return
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(data))
    p99s = [d[1]["p99_mean"] for d in data]
    sds = [d[1]["p99_sd"] for d in data]
    colors = ['#666666'] + ['#E74C3C']*4
    ax.bar(x, p99s, yerr=sds, color=colors, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], rotation=15)
    ax.set_ylabel("L1 p99 (ms)")
    ax.set_title("Fig 03 — Phase 2: Multi-AI scenarios")
    base = p99s[0]
    for i, p in enumerate(p99s):
        pct = (p - base) / base * 100
        ax.text(i, p + 0.3, f"{p:.1f}\n({pct:+.0f}%)", ha='center', fontsize=9)
    ax.axhline(base, ls='--', color='gray', alpha=0.5)
    save(fig, 3, "phase2_multiAI")


def fig_04_phase3_dynamic():
    """04: Phase 3 dynamic L1 (D1 starved vs D2 boosted)."""
    dirs = {
        "alone": ROOT / "n20_baseline_3g_alone",
        "D1 L1 starved": ROOT / "n20_phase3_D1_L1_starved",
        "D2 L1 boosted": ROOT / "n20_phase3_D2_L1_boosted",
    }
    data = [(l, aggregate(load_dir_runs(d))) for l, d in dirs.items()]
    data = [(l, a) for l, a in data if a]
    if not data: return
    fig, ax = plt.subplots(figsize=(9, 6))
    metrics = ['mean_mean', 'p95_mean', 'p99_mean']
    labels_metrics = ['Mean', 'p95', 'p99']
    cols = ['#3498DB', '#F39C12', '#E74C3C']
    x = np.arange(len(data))
    w = 0.25
    for j, m in enumerate(metrics):
        vals = [d[1][m] for d in data]
        ax.bar(x + (j-1)*w, vals, w, label=labels_metrics[j], color=cols[j])
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data])
    ax.set_ylabel("L1 latency (ms)")
    ax.set_title("Fig 04 — Phase 3: Dynamic L1 partition (starved vs boosted)")
    ax.legend()
    save(fig, 4, "phase3_dynamic")


def fig_05_phase4_PHY_AI():
    """05: Phase 4 PHY-AI workloads (chanpred / neuralrx / xapp)."""
    dirs = {
        "alone": ROOT / "n20_baseline_3g_alone",
        "+ chanpred": ROOT / "n20_phase4_chanpred",
        "+ neuralrx": ROOT / "n20_phase4_neuralrx",
        "+ xapp": ROOT / "n20_phase4_xapp",
    }
    data = [(l, aggregate(load_dir_runs(d))) for l, d in dirs.items()]
    data = [(l, a) for l, a in data if a]
    if not data: return
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(data))
    p99s = [d[1]["p99_mean"] for d in data]
    sds = [d[1]["p99_sd"] for d in data]
    colors = [color_for(d[0]) for d in data]
    ax.bar(x, p99s, yerr=sds, color=colors, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], rotation=15)
    ax.set_ylabel("L1 p99 (ms)")
    ax.set_title("Fig 05 — Phase 4: PHY-AI co-tenants (3g L1 + AI on 2g)")
    base = p99s[0]
    for i, p in enumerate(p99s):
        pct = (p - base) / base * 100
        ax.text(i, p + 0.3, f"{p:.1f}\n({pct:+.0f}%)", ha='center', fontsize=9)
    ax.axhline(base, ls='--', color='gray', alpha=0.5)
    save(fig, 5, "phase4_PHY_AI")


def fig_06_tier1_ranked():
    """06: All Tier1 scenarios ranked by p99."""
    dirs = {
        "fullGPU": ROOT / "n20_baseline_fullGPU",
        "7g": ROOT / "n20_baseline_7g_single",
        "4g": ROOT / "n20_baseline_4g_alone",
        "3g alone": ROOT / "n20_baseline_3g_alone",
        "2g": ROOT / "n20_baseline_2g_alone",
        "qwen_small": ROOT / "n20_phase1_qwen_small",
        "qwen7b_decode": ROOT / "n20_phase1_qwen7b_decode",
        "qwen7b_prefill": ROOT / "n20_phase1_qwen7b_prefill",
        "qwen7b_stress": ROOT / "n20_phase1_qwen7b_stress",
        "M1": ROOT / "n20_phase2_M1_3way_balanced",
        "M2": ROOT / "n20_phase2_M2_3way_L1small",
        "M3": ROOT / "n20_phase2_M3_3way_asym",
        "M4": ROOT / "n20_phase2_M4_4way_1L1_3AI",
        "D1 starved": ROOT / "n20_phase3_D1_L1_starved",
        "D2 boosted": ROOT / "n20_phase3_D2_L1_boosted",
        "chanpred": ROOT / "n20_phase4_chanpred",
        "neuralrx": ROOT / "n20_phase4_neuralrx",
        "xapp": ROOT / "n20_phase4_xapp",
    }
    data = []
    for lbl, d in dirs.items():
        agg = aggregate(load_dir_runs(d))
        if agg: data.append((lbl, agg))
    if not data: return
    data.sort(key=lambda x: x[1]["p99_mean"])
    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(data))
    p99s = [d[1]["p99_mean"] for d in data]
    sds = [d[1]["p99_sd"] for d in data]
    colors = [color_for(d[0]) for d in data]
    ax.barh(y, p99s, xerr=sds, color=colors, capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([d[0] for d in data])
    ax.set_xlabel("L1 p99 (ms)")
    ax.set_title("Fig 06 — All Tier1 scenarios ranked by p99")
    for i, p in enumerate(p99s):
        ax.text(p + 0.5, i, f"{p:.1f}ms", va='center', fontsize=9)
    save(fig, 6, "tier1_ranked")


# ===========================================================================
# Group 2: L1 multi-AI matrix M5-M12 (figs 07-12)
# ===========================================================================
def collect_l1_multi_ai():
    """Returns dict {scenario: {alone:agg, multi:agg}}."""
    base = ROOT / "l1_multi_ai"
    if not base.exists(): return {}
    out = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir(): continue
        alone = aggregate(load_dir_runs(d / "alone"))
        multi = aggregate(load_dir_runs(d / "multi"))
        out[d.name] = {"alone": alone, "multi": multi}
    return out


def fig_07_multi_ai_3wbal():
    """07: M5/M8 (3-way balanced) multi-AI scenarios."""
    data = collect_l1_multi_ai()
    keys = [k for k in data if k.startswith("M5") or k.startswith("M8")]
    keys.sort()
    if not keys: return
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(keys))
    alones = [data[k]["alone"]["p99_mean"] if data[k]["alone"] else 0 for k in keys]
    multis = [data[k]["multi"]["p99_mean"] if data[k]["multi"] else 0 for k in keys]
    w = 0.4
    ax.bar(x - w/2, alones, w, label='alone (no AI)', color='#666666')
    ax.bar(x + w/2, multis, w, label='+ AI workloads', color='#E74C3C')
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=25, ha='right')
    ax.set_ylabel("L1 p99 (ms)")
    ax.set_title("Fig 07 — Multi-AI 3-way balanced (M5/M8 series)")
    ax.legend()
    for i, (a, m) in enumerate(zip(alones, multis)):
        if a > 0:
            pct = (m-a)/a*100 if a else 0
            ax.text(i + w/2, m + 0.2, f"{pct:+.0f}%", ha='center', fontsize=8)
    save(fig, 7, "multi_ai_3wbal")


def fig_08_multi_ai_partition_size():
    """08: M9-M12 (2g/4g L1 with various AI)."""
    data = collect_l1_multi_ai()
    keys = [k for k in data if k.startswith("M9") or k.startswith("M10") or k.startswith("M11") or k.startswith("M12")]
    keys.sort()
    if not keys: return
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(keys))
    alones = [data[k]["alone"]["p99_mean"] if data[k]["alone"] else 0 for k in keys]
    multis = [data[k]["multi"]["p99_mean"] if data[k]["multi"] else 0 for k in keys]
    w = 0.4
    ax.bar(x - w/2, alones, w, label='alone', color='#666666')
    ax.bar(x + w/2, multis, w, label='+ AI', color='#E74C3C')
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=30, ha='right')
    ax.set_ylabel("L1 p99 (ms)")
    ax.set_title("Fig 08 — Partition-aware multi-AI (M9-M12)")
    ax.legend()
    save(fig, 8, "multi_ai_partition_size")


def fig_09_multi_ai_count_effect():
    """09: 2-way / 3-way / 4-way multi-AI count effect."""
    data = collect_l1_multi_ai()
    # M5-M8: 3-way; M7b: 4-way; M9b/M9c/M11b: 2-way
    groups = {
        "2-way (M9-M11)": [k for k in data if any(p in k for p in ["M9", "M11"])],
        "3-way (M5-M8)": [k for k in data if any(p in k for p in ["M5", "M6", "M8"])],
        "4-way (M7)": [k for k in data if k.startswith("M7")],
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    labels, deltas = [], []
    for grp, ks in groups.items():
        for k in ks:
            a = data[k]["alone"]
            m = data[k]["multi"]
            if a and m:
                labels.append(f"{k}\n({grp})")
                deltas.append((m["p99_mean"] - a["p99_mean"]) / a["p99_mean"] * 100)
    if not deltas: return
    colors = ['#3498DB' if '2-way' in l else '#E74C3C' if '3-way' in l else '#9B59B6' for l in labels]
    x = np.arange(len(deltas))
    ax.bar(x, deltas, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel("Δp99 vs alone (%)")
    ax.set_title("Fig 09 — Multi-AI count effect: % p99 inflation")
    ax.axhline(0, color='black', lw=0.5)
    save(fig, 9, "multi_ai_count_effect")


def fig_10_het_vs_homo():
    """10: HET vs HOMO multi-AI."""
    data = collect_l1_multi_ai()
    cat = {"HET": [], "HOMO": []}
    for k, v in data.items():
        if not (v["alone"] and v["multi"]): continue
        delta = (v["multi"]["p99_mean"] - v["alone"]["p99_mean"]) / v["alone"]["p99_mean"] * 100
        if "het" in k.lower(): cat["HET"].append((k, delta))
        elif "sat" in k.lower() and "2sat" in k: cat["HOMO"].append((k, delta))
        elif "2sat" in k or "3sat" in k: cat["HOMO"].append((k, delta))
    fig, ax = plt.subplots(figsize=(10, 6))
    labels, deltas, colors = [], [], []
    for grp, items in cat.items():
        for name, d in items:
            labels.append(name)
            deltas.append(d)
            colors.append('#9B59B6' if grp == "HET" else '#F39C12')
    if not deltas: return
    x = np.arange(len(deltas))
    ax.bar(x, deltas, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Δp99 vs alone (%)")
    ax.set_title("Fig 10 — Heterogeneous vs Homogeneous multi-AI")
    handles = [plt.Rectangle((0,0),1,1,color='#9B59B6', label='HET (chaos+stable)'),
               plt.Rectangle((0,0),1,1,color='#F39C12', label='HOMO (multiple sat)')]
    ax.legend(handles=handles)
    save(fig, 10, "het_vs_homo")


def fig_11_partition_with_same_AI():
    """11: 2g/3g/4g L1 with similar AI side - partition robustness."""
    data = collect_l1_multi_ai()
    # 2g_neuralrx / 3g_neuralrx / 4g_neuralrx style mapping
    candidates = [
        ("2g + sat", "M11a_2gL1_sat_compute"),
        ("2g + neuralrx", "M11b_2gL1_neuralrx"),
        ("2g + resnet", "M11c_2gL1_resnet"),
        ("4g + sat", "M10a_4gL1_sat_compute"),
        ("4g + neuralrx", "M10b_4gL1_neuralrx"),
        ("4g + resnet", "M10c_4gL1_resnet"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    labels, deltas, colors = [], [], []
    for lbl, k in candidates:
        if k in data and data[k]["alone"] and data[k]["multi"]:
            d = (data[k]["multi"]["p99_mean"] - data[k]["alone"]["p99_mean"]) / data[k]["alone"]["p99_mean"] * 100
            labels.append(lbl); deltas.append(d)
            colors.append('#3498DB' if "2g" in lbl else '#E67E22')
    if not deltas: return
    x = np.arange(len(deltas))
    ax.bar(x, deltas, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Δp99 vs alone (%)")
    ax.set_title("Fig 11 — Partition size robustness (2g L1 vs 4g L1)")
    handles = [plt.Rectangle((0,0),1,1,color='#3498DB', label='2g L1 (vulnerable)'),
               plt.Rectangle((0,0),1,1,color='#E67E22', label='4g L1 (robust)')]
    ax.legend(handles=handles)
    save(fig, 11, "partition_robustness")


def fig_12_all_multi_AI_ranked():
    """12: All M5-M12 scenarios ranked by p99 inflation."""
    data = collect_l1_multi_ai()
    items = []
    for k, v in data.items():
        if v["alone"] and v["multi"]:
            d = (v["multi"]["p99_mean"] - v["alone"]["p99_mean"]) / v["alone"]["p99_mean"] * 100
            items.append((k, d, v["multi"]["p99_mean"]))
    if not items: return
    items.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(11, 9))
    y = np.arange(len(items))
    deltas = [i[1] for i in items]
    colors = ['#E74C3C' if d > 20 else '#F39C12' if d > 5 else '#27AE60' for d in deltas]
    ax.barh(y, deltas, color=colors)
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=9)
    ax.set_xlabel("Δp99 vs alone (%)")
    ax.set_title("Fig 12 — All multi-AI scenarios ranked by p99 inflation")
    ax.axvline(0, color='black', lw=0.5)
    for i, (_, d, _) in enumerate(items):
        ax.text(d + 0.5 if d >= 0 else d - 0.5, i, f"{d:+.1f}%", va='center',
                ha='left' if d >= 0 else 'right', fontsize=8)
    save(fig, 12, "all_multi_AI_ranked")


# ===========================================================================
# Group 3: AI per-op latency: alone vs with_l1 (figs 13-18)
# ===========================================================================
def parse_per_op_json(d):
    """Parse per-op latency JSON embedded in run_*.log via [X-latency-json] marker."""
    runs = []
    if not Path(d).exists(): return runs
    pat = re.compile(r"\[\w+[-_]latency-json\]\s*(\{.*\})")
    for log in sorted(Path(d).glob("run_*.log")):
        try:
            txt = log.read_text(errors="ignore")
        except Exception:
            continue
        for line in txt.splitlines():
            m = pat.search(line)
            if m:
                try:
                    runs.append(json.loads(m.group(1)))
                except json.JSONDecodeError:
                    pass
    return runs


def collect_ai_per_op(ai_name, partitions=("1g","2g","3g","4g")):
    """For one AI workload, gather alone vs with_l1 stats across partitions."""
    base = ROOT / "ai_per_op_latency"
    base_b = ROOT / "ai_per_op_latency_b"
    result = {}
    for p in partitions:
        for root in (base, base_b):
            ad = root / f"{ai_name}_{p}" / "alone"
            wd = root / f"{ai_name}_{p}" / "with_l1"
            if not ad.exists() and not wd.exists(): continue
            runs_a = parse_per_op_json(ad)
            runs_w = parse_per_op_json(wd)
            result[p] = {
                "alone": {"n": len(runs_a),
                          "p99": np.mean([r.get("p99_ms", 0) for r in runs_a]) if runs_a else 0,
                          "mean": np.mean([r.get("mean_ms", 0) for r in runs_a]) if runs_a else 0},
                "with_l1": {"n": len(runs_w),
                            "p99": np.mean([r.get("p99_ms", 0) for r in runs_w]) if runs_w else 0,
                            "mean": np.mean([r.get("mean_ms", 0) for r in runs_w]) if runs_w else 0},
            }
            break
    return result


def fig_ai_partition_scaling(idx, ai_name, title_suffix=""):
    """Generic figure: per-AI partition scaling, alone vs with_l1."""
    data = collect_ai_per_op(ai_name)
    if not data: return
    parts = sorted(data.keys(), key=lambda x: int(x[:-1]))
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(parts))
    alone_p99 = [data[p]["alone"]["p99"] for p in parts]
    with_l1_p99 = [data[p]["with_l1"]["p99"] for p in parts]
    w = 0.35
    ax.bar(x - w/2, alone_p99, w, label='AI alone', color='#666666')
    ax.bar(x + w/2, with_l1_p99, w, label='AI + L1', color=color_for(ai_name))
    ax.set_xticks(x); ax.set_xticklabels(parts)
    ax.set_ylabel(f"{ai_name} per-op p99 (ms)")
    ax.set_title(f"Fig {idx:02d} — AI {ai_name} per-op p99: alone vs with cuPHY L1 {title_suffix}")
    ax.legend()
    for i, (a, w_) in enumerate(zip(alone_p99, with_l1_p99)):
        if a > 0:
            pct = (w_ - a) / a * 100
            ax.text(i + 0.18, w_ + (w_ * 0.02), f"{pct:+.1f}%", ha='left', fontsize=9)
    save(fig, idx, f"ai_per_op_{ai_name}")


def fig_18_all_ai_alone_vs_l1():
    """18: All AI types, 3g partition, alone vs with_l1 (delta)."""
    ais = ["chanpred", "neuralrx", "resnet", "qwen", "forecaster", "xapp"]
    fig, ax = plt.subplots(figsize=(11, 6))
    labels, alones, withl1s, colors = [], [], [], []
    for ai in ais:
        data = collect_ai_per_op(ai, partitions=("3g",))
        if "3g" in data:
            a = data["3g"]["alone"]["p99"]
            w_ = data["3g"]["with_l1"]["p99"]
            if a > 0 or w_ > 0:
                labels.append(ai)
                alones.append(a)
                withl1s.append(w_)
                colors.append(color_for(ai))
    if not labels: return
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, alones, w, label='alone', color='#999999')
    ax.bar(x + w/2, withl1s, w, label='+ L1', color=colors)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("AI per-op p99 (ms)")
    ax.set_title("Fig 18 — AI per-op p99 (3g): alone vs + L1 across AI types")
    ax.legend()
    for i, (a, w_) in enumerate(zip(alones, withl1s)):
        if a > 0:
            pct = (w_ - a) / a * 100
            ax.text(i + 0.18, w_, f"{pct:+.0f}%", ha='left', fontsize=9)
    save(fig, 18, "all_ai_alone_vs_l1")


# ===========================================================================
# Group 4: NCU per-kernel HW metrics (figs 19-22)
# ===========================================================================
def parse_ncu_csv(csvpath):
    """Parse NCU CSV (skip WARNING lines before "ID" header). Uses csv module for quoted fields."""
    import csv as csvmod
    if not Path(csvpath).exists(): return {}
    by_metric = defaultdict(list)
    try:
        with open(csvpath, newline='') as f:
            # Skip non-CSV header (WARNING lines)
            lines = f.readlines()
        # Find start of CSV (line starting with "ID")
        start = 0
        for i, line in enumerate(lines):
            if line.startswith('"ID"'):
                start = i
                break
        else:
            return {}
        reader = csvmod.DictReader(lines[start:])
        for row in reader:
            mn = row.get("Metric Name", "")
            mv = row.get("Metric Value", "")
            if not mn or not mv: continue
            try:
                by_metric[mn].append(float(mv.replace(",", "")))
            except ValueError:
                continue
    except Exception:
        pass
    return dict(by_metric)


def fig_19_ncu_dram_throughput():
    """19: NCU DRAM throughput across MIG scenarios."""
    ncu_dir = ROOT / "ncu_csv"
    if not ncu_dir.exists(): return
    rows = []
    target_metric = None
    for csv in sorted(ncu_dir.glob("S*.csv")):
        data = parse_ncu_csv(csv)
        # Find a DRAM throughput metric
        if target_metric is None:
            for m in data:
                if "dram" in m.lower() and "throughput" in m.lower():
                    target_metric = m
                    break
        if target_metric and target_metric in data and data[target_metric]:
            label = csv.stem
            rows.append((label, np.mean(data[target_metric])))
    if not rows: return
    rows.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    ax.barh(y, vals, color='#3498DB')
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel(f"{target_metric} (mean across kernels)")
    ax.set_title("Fig 19 — NCU DRAM throughput across MIG scenarios")
    save(fig, 19, "ncu_dram_throughput")


def fig_20_ncu_l2_hit():
    """20: NCU L2 cache hit rate across scenarios."""
    ncu_dir = ROOT / "ncu_csv"
    if not ncu_dir.exists(): return
    rows = []
    target = None
    for csv in sorted(ncu_dir.glob("S*.csv")):
        data = parse_ncu_csv(csv)
        if target is None:
            for m in data:
                if "lts" in m.lower() and "hit" in m.lower():
                    target = m; break
        if target and target in data and data[target]:
            rows.append((csv.stem, np.mean(data[target])))
    if not rows: return
    rows.sort(key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    ax.barh(y, vals, color='#27AE60')
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel(f"{target} (mean across kernels)")
    ax.set_title("Fig 20 — NCU L2 cache hit rate across scenarios")
    save(fig, 20, "ncu_l2_hit")


def fig_21_ncu_sm_busy():
    """21: NCU SM busy / warps active."""
    ncu_dir = ROOT / "ncu_csv"
    if not ncu_dir.exists(): return
    rows = []
    target = None
    for csv in sorted(ncu_dir.glob("S*.csv")):
        data = parse_ncu_csv(csv)
        if target is None:
            for m in data:
                if "sm__warps" in m.lower() or ("sm" in m.lower() and "active" in m.lower()):
                    target = m; break
        if target and target in data and data[target]:
            rows.append((csv.stem, np.mean(data[target])))
    if not rows: return
    rows.sort(key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    ax.barh(y, vals, color='#9B59B6')
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel(f"{target} (mean across kernels)")
    ax.set_title("Fig 21 — NCU SM warps active across scenarios")
    save(fig, 21, "ncu_sm_busy")


def fig_22_ncu_multimetric_heatmap():
    """22: Multi-metric heatmap (rows=scenarios, cols=metrics)."""
    ncu_dir = ROOT / "ncu_csv"
    if not ncu_dir.exists(): return
    by_scenario = {}
    for csv in sorted(ncu_dir.glob("S*.csv")):
        data = parse_ncu_csv(csv)
        by_scenario[csv.stem] = {m: np.mean(v) for m, v in data.items() if v}
    if not by_scenario: return
    # Pick top 6 metrics common across all
    all_metrics = sorted(set().union(*[set(d.keys()) for d in by_scenario.values()]))
    # Keep those present in all
    common = [m for m in all_metrics if all(m in d for d in by_scenario.values())]
    if not common: return
    # Normalize each metric to [0,1]
    metrics_to_show = common[:8]
    scenarios = sorted(by_scenario.keys())
    M = np.array([[by_scenario[s][m] for m in metrics_to_show] for s in scenarios])
    # Normalize column-wise
    Mn = (M - M.min(axis=0)) / (np.ptp(M, axis=0) + 1e-9)
    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(Mn, aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(np.arange(len(metrics_to_show)))
    ax.set_xticklabels([m[:30] for m in metrics_to_show], rotation=40, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=8)
    ax.set_title("Fig 22 — NCU HW metrics heatmap (normalized per metric)")
    fig.colorbar(im, ax=ax, label='normalized value')
    save(fig, 22, "ncu_multimetric_heatmap")


# ===========================================================================
# Group 5: NSYS v3 inter-kernel gap (figs 23-26)
# ===========================================================================
def load_nsys_v3_summary():
    """Load nsys_csv_v2 short stats from CSV outputs."""
    # Use the previously-computed analyses if present
    p = ROOT / "nsys_sqlite_v2_analysis"
    summary = {}
    if not p.exists(): return summary
    # Look for any per-scenario CSV with gap stats
    for csv in p.glob("*timeseries*.csv"):
        # not used here
        pass
    # Fallback: try nsys_csv_v2
    return summary


def fig_23_nsys_v3_gap_p99():
    """23: NSYS v3 inter-kernel gap p99 across 26 scenarios.
    Uses pre-computed stats from analyze_nsys_comprehensive output if present.
    """
    csvp = ROOT / "nsys_sqlite_v2_analysis"
    rows = []
    # Try to find a comprehensive summary - if not, parse from a known CSV
    # Just produce a snapshot from any existing CSV containing 'p99' col
    candidates = list((ROOT).glob("**/*scenario*.csv"))
    src = None
    for c in candidates:
        head = c.read_text().splitlines()[0] if c.exists() else ""
        if "p99" in head.lower() and "scenario" in head.lower():
            src = c
            break
    if src is None:
        # fallback: build from analyze script's outputs in C/B (we have them)
        for d in [ROOT / "nsys_deep_C_analysis", ROOT / "nsys_deep_A_analysis"]:
            for c in d.glob("*.csv"):
                head = c.read_text().splitlines()[0]
                if "p99" in head.lower():
                    src = c; break
            if src: break
    if not src:
        print("  ! no v3 summary csv found, skipping fig_23")
        return
    lines = src.read_text().splitlines()
    if len(lines) < 2: return
    cols = lines[0].split(",")
    # Use scenario col + p99 col
    try:
        sc_i = next(i for i,c in enumerate(cols) if 'scenario' in c.lower() or 'config' in c.lower())
    except StopIteration:
        return
    try:
        p99_i = next(i for i,c in enumerate(cols) if 'p99' in c.lower() and 'gap' in c.lower() or c.strip() == 'p99_gap_us')
    except StopIteration:
        p99_i = next((i for i, c in enumerate(cols) if 'p99' in c.lower()), 1)
    rows = []
    for line in lines[1:]:
        cells = line.split(",")
        if len(cells) <= max(sc_i, p99_i): continue
        try:
            rows.append((cells[sc_i], float(cells[p99_i])))
        except ValueError:
            pass
    if not rows: return
    rows.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(11, 9))
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    colors = ['#E74C3C' if v > 1200 else '#F39C12' if v > 1000 else '#27AE60' for v in vals]
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("inter-kernel gap p99 (us)")
    ax.set_title("Fig 23 — Inter-kernel gap p99 across scenarios (from analysis CSVs)")
    save(fig, 23, "nsys_gap_p99_ranked")


def fig_24_longtail_ratio():
    """24: Long-tail ratio (p999/p99) — outlier severity."""
    # Use Dv2 summary CSV
    csvp = ROOT / "nsys_deep_Dv2_analysis" / "Dv2_summary.csv"
    if not csvp.exists():
        print("  ! Dv2 summary missing"); return
    rows = []
    for line in csvp.read_text().splitlines()[1:]:
        c = line.split(",")
        if len(c) < 8: continue
        sc, n, p99, p99sd, lo, hi, p999, p999sd = c[0], int(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5]), float(c[6]), float(c[7])
        rows.append((sc, p99, p999, p999/p99 if p99 else 0))
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(rows))
    p99s = [r[1] for r in rows]
    p999s = [r[2] for r in rows]
    w = 0.35
    ax.bar(x - w/2, p99s, w, label='p99 gap (us)', color='#3498DB')
    ax.bar(x + w/2, p999s, w, label='p999 gap (us)', color='#E74C3C')
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=15)
    ax.set_ylabel("inter-kernel gap (us)")
    ax.set_title("Fig 24 — Dv2 (n=10) p99 vs p999: long-tail ratio")
    ax.legend()
    for i, (sc, p99, p999, ratio) in enumerate(rows):
        ax.text(i, max(p99, p999) + 50, f"x{ratio:.2f}", ha='center', fontsize=9)
    save(fig, 24, "longtail_ratio_dv2")


def fig_25_burst_share():
    """25: Burst share — top 1% gap as % of total idle."""
    csvp = ROOT / "nsys_deep_A_analysis" / "paper_table.csv"
    if not csvp.exists(): return
    lines = csvp.read_text().splitlines()
    if len(lines) < 2: return
    cols = lines[0].split(",")
    try:
        sc_i = cols.index("scenario")
        bs_i = next(i for i, c in enumerate(cols) if "top1" in c.lower() or "burst" in c.lower())
    except (ValueError, StopIteration):
        return
    rows = []
    for line in lines[1:]:
        c = line.split(",")
        if len(c) <= max(sc_i, bs_i): continue
        val = c[bs_i].rstrip('%')
        try:
            rows.append((c[sc_i], float(val)))
        except ValueError:
            pass
    if not rows: return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rows))
    vals = [r[1] for r in rows]
    ax.bar(x, vals, color='#9B59B6')
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=15)
    ax.set_ylabel("top 1% gap share of total idle (%)")
    ax.set_title("Fig 25 — Burst-mode contention: top 1% gap idle share")
    ax.axhline(np.mean(vals), ls='--', color='gray', label=f'mean {np.mean(vals):.1f}%')
    ax.legend()
    save(fig, 25, "burst_share_top1")


def fig_26_memory_ops():
    """26: Memcpy timing impact (use Dv2 max gap)."""
    csvp = ROOT / "nsys_deep_Dv2_analysis" / "Dv2_summary.csv"
    if not csvp.exists(): return
    rows = []
    for line in csvp.read_text().splitlines()[1:]:
        c = line.split(",")
        if len(c) < 10: continue
        rows.append((c[0], float(c[6]), float(c[8])))  # p999, max_mean
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(rows))
    p999s = [r[1] for r in rows]
    maxs = [r[2] for r in rows]
    w = 0.35
    ax.bar(x - w/2, p999s, w, label='p999 (us)', color='#3498DB')
    ax.bar(x + w/2, maxs, w, label='max (us)', color='#E74C3C')
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=15)
    ax.set_ylabel("gap (us)")
    ax.set_title("Fig 26 — Memory ops impact: p999 vs max gap (Dv2 n=10)")
    ax.legend()
    save(fig, 26, "memory_ops_p999_max")


# ===========================================================================
# Group 6: Deep-dive A/B/C/D/Dv2/E + saturation summary (figs 27-34)
# ===========================================================================
def fig_27_stage_A_dual_concurrent():
    """27: Stage A — p99 across A1/A2/A3/A4."""
    csvp = ROOT / "nsys_deep_A_analysis" / "paper_table.csv"
    if not csvp.exists(): return
    lines = csvp.read_text().splitlines()
    cols = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        c = line.split(",")
        if len(c) < 5: continue
        rows.append((c[0], float(c[3]), float(c[4])))  # p99, p999
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(rows))
    p99s = [r[1] for r in rows]
    p999s = [r[2] for r in rows]
    w = 0.35
    ax.bar(x - w/2, p99s, w, label='p99 gap', color='#3498DB')
    ax.bar(x + w/2, p999s, w, label='p999 gap', color='#E74C3C')
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=10)
    ax.set_ylabel("inter-kernel gap (us)")
    ax.set_title("Fig 27 — Stage A dual-concurrent: L1 gap p99/p999")
    ax.legend()
    save(fig, 27, "stageA_dual_concurrent")


def fig_28_stage_B_recovery():
    """28: Stage B — L1 p99 over time (phase markers from B rerun)."""
    bdir = ROOT / "nsys_deep_B_analysis"
    if not bdir.exists(): return
    fig, ax = plt.subplots(figsize=(11, 6))
    found = False
    for csv in bdir.glob("*_p99_timeline.csv"):
        try:
            lines = csv.read_text().splitlines()[1:]
            ts, p99s = [], []
            for line in lines:
                c = line.split(",")
                if len(c) < 2: continue
                ts.append(float(c[0])); p99s.append(float(c[1]))
            if ts:
                lbl = csv.stem.replace("_p99_timeline","")
                ax.plot(ts, p99s, label=lbl, lw=1.5)
                found = True
        except Exception:
            continue
    if not found: return
    # Add reference phase boundaries (rough)
    for t, lbl in [(0, "no AI"), (30, "AI ON"), (72, "AI OFF"), (93, "AI ON")]:
        ax.axvline(t, ls=':', color='gray', alpha=0.5)
        ax.text(t, ax.get_ylim()[1]*0.9, lbl, fontsize=8, rotation=90, va='top')
    ax.set_xlabel("time (s)")
    ax.set_ylabel("L1 gap p99 (us)")
    ax.set_title("Fig 28 — Stage B AI ON/OFF transition: p99 over time")
    ax.legend(fontsize=8)
    save(fig, 28, "stageB_transition")


def fig_29_stage_C_dose_response():
    """29: Stage C — chanpred BATCH sweep on 2g/3g/4g L1."""
    csvp = ROOT / "nsys_deep_C_analysis" / "dose_response.csv"
    if not csvp.exists(): return
    by_config = defaultdict(list)
    for line in csvp.read_text().splitlines()[1:]:
        c = line.split(",")
        if len(c) < 4: continue
        cfg, batch, p99 = c[0], int(c[1]), float(c[2])
        by_config[cfg].append((batch, p99))
    fig, ax = plt.subplots(figsize=(10, 6))
    for cfg, pts in by_config.items():
        pts.sort()
        bs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(bs, ys, 'o-', label=cfg)
    ax.set_xscale('log')
    ax.set_xlabel("BATCH size (chanpred) or DIM (Forecaster)")
    ax.set_ylabel("L1 gap p99 (us)")
    ax.set_title("Fig 29 — Stage C: dose-response (BATCH/DIM sweep)")
    ax.legend(fontsize=8)
    save(fig, 29, "stageC_dose_response")


def fig_30_stage_D_phase_decomposition():
    """30: Stage D — phase decomposition (memcpy/compute/launch)."""
    csvp = ROOT / "nsys_deep_D_analysis" / "phase_decomposition.csv"
    if not csvp.exists(): return
    rows = []
    for line in csvp.read_text().splitlines()[1:]:
        c = line.split(",")
        if len(c) < 6: continue
        sc, p99 = c[0], float(c[1])
        delta = c[2].rstrip('%')
        p999 = float(c[3])
        d999 = c[4].rstrip('%')
        try:
            rows.append((sc, p99, float(delta), p999, float(d999)))
        except ValueError:
            continue
    if not rows: return
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(rows))
    p99s = [r[1] for r in rows]
    p999s = [r[3] for r in rows]
    w = 0.35
    ax.bar(x - w/2, p99s, w, label='p99 (us)', color='#3498DB')
    ax.bar(x + w/2, p999s, w, label='p999 (us)', color='#E74C3C')
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=15, ha='right', fontsize=8)
    ax.set_ylabel("inter-kernel gap (us)")
    ax.set_title("Fig 30 — Stage D phase decomposition (n=3 baseline; subject to noise)")
    ax.legend()
    save(fig, 30, "stageD_phase_decomposition")


def fig_31_stage_E_smoothing():
    """31: Stage E — HOMO vs HET multi-AI frame-time p99."""
    csvp = ROOT / "nsys_deep_E_analysis" / "deep_E_stats.csv"
    if not csvp.exists(): return
    rows = []
    for line in csvp.read_text().splitlines()[1:]:
        c = line.split(",")
        if len(c) < 9: continue
        rows.append((c[0], float(c[2]), float(c[6]), float(c[7])))  # n_kern, p99_gap, p999_gap
    if not rows: return
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(rows))
    p99s = [r[2] for r in rows]
    p999s = [r[3] for r in rows]
    w = 0.35
    ax.bar(x - w/2, p99s, w, label='p99 gap', color='#3498DB')
    ax.bar(x + w/2, p999s, w, label='p999 gap', color='#E74C3C')
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=20, ha='right', fontsize=8)
    ax.set_ylabel("inter-kernel gap (us)")
    ax.set_title("Fig 31 — Stage E: HOMO vs HET multi-AI (gap p99/p999)")
    ax.legend()
    save(fig, 31, "stageE_homo_vs_het")


def fig_32_Dv2_n10_CI():
    """32: Dv2 n=10 p99 mean ± CI (shows all-overlap with baseline)."""
    csvp = ROOT / "nsys_deep_Dv2_analysis" / "Dv2_summary.csv"
    if not csvp.exists(): return
    rows = []
    for line in csvp.read_text().splitlines()[1:]:
        c = line.split(",")
        if len(c) < 6: continue
        rows.append((c[0], float(c[2]), float(c[3]), float(c[4]), float(c[5])))
    if not rows: return
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(rows))
    p99s = [r[1] for r in rows]
    ci_lo = [r[3] for r in rows]
    ci_hi = [r[4] for r in rows]
    err_lo = [p99s[i] - ci_lo[i] for i in range(len(rows))]
    err_hi = [ci_hi[i] - p99s[i] for i in range(len(rows))]
    colors = [color_for(r[0]) for r in rows]
    ax.bar(x, p99s, yerr=[err_lo, err_hi], color=colors, capsize=6)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=15)
    ax.set_ylabel("L1 gap p99 (us, mean ± 95% CI)")
    ax.set_title("Fig 32 — Dv2 (n=10): all phases overlap baseline → 'bandwidth' rejected (statistically)")
    base = p99s[0]
    ax.axhline(base, ls='--', color='black', alpha=0.5)
    ax.axhspan(ci_lo[0], ci_hi[0], alpha=0.15, color='gray', label='baseline 95% CI')
    ax.legend()
    save(fig, 32, "dv2_n10_CI")


def fig_33_longtail_all_stages():
    """33: Long-tail (p999/p99 ratio) across A/Dv2/E."""
    rows = []
    for src, csvp, p99col, p999col in [
        ("A", ROOT / "nsys_deep_A_analysis" / "paper_table.csv", 3, 4),
        ("Dv2", ROOT / "nsys_deep_Dv2_analysis" / "Dv2_summary.csv", 2, 6),
        ("E", ROOT / "nsys_deep_E_analysis" / "deep_E_stats.csv", 6, 7),
    ]:
        if not csvp.exists(): continue
        for line in csvp.read_text().splitlines()[1:]:
            c = line.split(",")
            if len(c) <= max(p99col, p999col): continue
            try:
                p99, p999 = float(c[p99col]), float(c[p999col])
                ratio = p999/p99 if p99 else 0
                rows.append((f"{src}:{c[0]}", ratio))
            except ValueError:
                pass
    if not rows: return
    rows.sort(key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(11, 9))
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    colors = ['#E74C3C' if v > 1.5 else '#F39C12' if v > 1.2 else '#27AE60' for v in vals]
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("p999 / p99 ratio")
    ax.set_title("Fig 33 — Long-tail severity across all stages")
    save(fig, 33, "longtail_all_stages")


def fig_34_stage_F_failed_summary():
    """34: Stage F (initial attempt, failed before 6/1 redo) — partial captures."""
    fdir = ROOT / "nsys_deep_F"
    if not fdir.exists(): return
    captures = list(fdir.glob("*.nsys-rep"))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.text(0.5, 0.5,
            f"Stage F (5/31 initial attempt)\n{len(captures)} partial captures collected\n"
            f"(continued and completed in 20260601/F_saturation with full n=5)",
            ha='center', va='center', fontsize=14)
    ax.set_axis_off()
    ax.set_title("Fig 34 — Stage F status note (5/31 partial → 6/1 expanded)")
    save(fig, 34, "stageF_partial_note")


# ===========================================================================
# Group 7: P3/P5 sweep summaries (figs 35-36)
# ===========================================================================
def fig_35_P5_sustained():
    """35: P5 sustained 9 workloads p99."""
    p5 = ROOT / "p5_sustained"
    if not p5.exists(): return
    data = []
    for d in sorted(p5.iterdir()):
        if not d.is_dir(): continue
        runs = []
        for log in sorted(d.glob("run_*_l1.log")):
            runs.extend(parse_log_stats(log))
        agg = aggregate(runs)
        if agg: data.append((d.name, agg))
    if not data: return
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(data))
    p99s = [d[1]["p99_mean"] for d in data]
    sds = [d[1]["p99_sd"] for d in data]
    colors = [color_for(d[0]) for d in data]
    ax.bar(x, p99s, yerr=sds, color=colors, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], rotation=20)
    ax.set_ylabel("L1 p99 (ms)")
    ax.set_title("Fig 35 — P5 sustained 5min: 9 workloads (mean ± SD)")
    base = next((d[1]["p99_mean"] for d in data if d[0] == "alone"), None)
    if base:
        ax.axhline(base, ls='--', color='gray', alpha=0.5, label=f'alone {base:.1f}ms')
        ax.legend()
    for i, (lbl, agg) in enumerate(data):
        ax.text(i, agg["p99_mean"] + 0.5, f"{agg['p99_mean']:.1f}", ha='center', fontsize=9)
    save(fig, 35, "p5_sustained")


def fig_36_P3_partition_sweep():
    """36: P3 partition sweep matrix (4 AI partitions × workloads)."""
    p3 = ROOT / "p3_partition_sweep"
    if not p3.exists(): return
    matrix = defaultdict(dict)
    for part_dir in sorted(p3.iterdir()):
        if not part_dir.is_dir(): continue
        part = part_dir.name.replace("AI=", "")
        for wl_dir in sorted(part_dir.iterdir()):
            if not wl_dir.is_dir(): continue
            runs = []
            for log in wl_dir.glob("run_*.log"):
                runs.extend(parse_log_stats(log))
            agg = aggregate(runs)
            if agg:
                matrix[wl_dir.name][part] = agg["p99_mean"]
    if not matrix: return
    workloads = sorted(matrix.keys())
    parts = sorted({p for d in matrix.values() for p in d.keys()})
    M = np.array([[matrix[w].get(p, np.nan) for p in parts] for w in workloads])
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(M, aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(np.arange(len(parts))); ax.set_xticklabels([f"AI={p}" for p in parts])
    ax.set_yticks(np.arange(len(workloads))); ax.set_yticklabels(workloads, fontsize=9)
    for i in range(len(workloads)):
        for j in range(len(parts)):
            if not np.isnan(M[i,j]):
                ax.text(j, i, f"{M[i,j]:.1f}", ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, label='L1 p99 (ms)')
    ax.set_title("Fig 36 — P3 partition sweep: L1 p99 (workload × AI partition)")
    save(fig, 36, "p3_partition_sweep")


# ===========================================================================
# main
# ===========================================================================
def main():
    print("Generating 36 figures →", OUT)
    fig_01_partition_baselines()
    fig_02_phase1_qwen()
    fig_03_phase2_multiAI()
    fig_04_phase3_dynamic()
    fig_05_phase4_PHY_AI()
    fig_06_tier1_ranked()
    fig_07_multi_ai_3wbal()
    fig_08_multi_ai_partition_size()
    fig_09_multi_ai_count_effect()
    fig_10_het_vs_homo()
    fig_11_partition_with_same_AI()
    fig_12_all_multi_AI_ranked()
    # AI per-op
    fig_ai_partition_scaling(13, "chanpred")
    fig_ai_partition_scaling(14, "neuralrx")
    fig_ai_partition_scaling(15, "resnet")
    fig_ai_partition_scaling(16, "qwen")
    fig_ai_partition_scaling(17, "forecaster")
    fig_18_all_ai_alone_vs_l1()
    # NCU
    fig_19_ncu_dram_throughput()
    fig_20_ncu_l2_hit()
    fig_21_ncu_sm_busy()
    fig_22_ncu_multimetric_heatmap()
    # NSYS aggregated
    fig_23_nsys_v3_gap_p99()
    fig_24_longtail_ratio()
    fig_25_burst_share()
    fig_26_memory_ops()
    # Deep-dive
    fig_27_stage_A_dual_concurrent()
    fig_28_stage_B_recovery()
    fig_29_stage_C_dose_response()
    fig_30_stage_D_phase_decomposition()
    fig_31_stage_E_smoothing()
    fig_32_Dv2_n10_CI()
    fig_33_longtail_all_stages()
    fig_34_stage_F_failed_summary()
    # Sweep
    fig_35_P5_sustained()
    fig_36_P3_partition_sweep()
    print(f"\nDone. {len(list(OUT.glob('fig_*.png')))} files in {OUT}")


if __name__ == "__main__":
    main()
