#!/usr/bin/env python3
"""
20-graph analysis for 5/24 sweep data.
Generates publication-quality figures organized by paper story arc.

Story arc:
  1. MIG mode itself is free (7g MIG ≈ no-MIG)
  2. Partition cap is the cost
  3. cuPHY saturates per partition (HBM bw theory)
  4. AI co-location: size matters more than count
  5. Real AI-RAN > LLM in disruption
  6. Bimodal intrinsic to cuPHY
  7. Tail latency catastrophic for URLLC

Usage:
    python3 generate_figures.py
Outputs:
    figures/fig_NN_description.png
"""
import json
import glob
import os
import sys
import statistics
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 11,
    'figure.figsize': (8, 5),
    'figure.dpi': 100,
    'savefig.dpi': 140,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.8,
    'lines.markersize': 7,
})

ROOT = Path(__file__).parent
EXTRA = ROOT / 'extra_2g_4g_cells'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)


def load_runs(d):
    """Load all run_*.json from directory, return list of dicts."""
    runs = []
    for f in sorted(glob.glob(str(d / 'run_*.json'))):
        try:
            runs.append(json.load(open(f)))
        except Exception:
            pass
    return runs


def stats(runs, key='mean_ms'):
    vals = [r[key] for r in runs if key in r]
    if not vals:
        return None
    return {
        'n': len(vals),
        'min': min(vals),
        'max': max(vals),
        'mean': statistics.mean(vals),
        'median': statistics.median(vals),
        'stdev': statistics.stdev(vals) if len(vals) > 1 else 0,
    }


def all_raw(runs):
    """Flatten all raw_ms across runs."""
    out = []
    for r in runs:
        out.extend(r.get('raw_ms', []))
    return np.array(out)


# Load datasets
DS = {
    # Baselines
    'full_GPU_noMIG': ROOT / 'n20_baseline_gpu1_fullGPU',
    '7g_MIG_alone': ROOT / 'n10_L1_alone_7g40gb_MIG',
    '4g_MIG_clean': ROOT / 'n10_L1_alone_4g20gb_clean',
    '4g_MIG_parallel': ROOT / 'n10_L1_alone_4g20gb',
    '3g_MIG_alone': ROOT / 'n20_L1_alone_3g20gb',
    '2g_MIG_alone': ROOT / 'n10_D1b_4060_alone',
    # Phase 1
    'A0_qwen_full': ROOT / 'n20_A0_qwen_baseline',
    'A1_prefill': ROOT / 'n20_A1_prefill',
    'A1_decode': ROOT / 'n20_A1_decode',
    'A2_hbm_static': ROOT / 'n20_A2_hbm',
    # Phase 2
    'M1_3way_balanced': ROOT / 'n10_M1_3way_balanced_AIRAN',
    'M2_3way_L1small': ROOT / 'n10_M2_3way_L1small_mixed',
    'M3_3way_asym': ROOT / 'n10_M3_3way_asym_AIRAN',
    'M4_4way_3xApp': ROOT / 'n10_M4_4way_3xApp',
    # Phase 3
    'D1a_split40_60_qwen': ROOT / 'n10_D1a_4060_qwen',
    'D1b_split40_60_alone': ROOT / 'n10_D1b_4060_alone',
    # Phase 4
    'AR1_neuralrx': ROOT / 'n10_AR1_6040_neuralrx',
    'AR2_chanpred': ROOT / 'n10_AR2_6040_chanpred',
    'AR3_xapp': ROOT / 'n10_AR3_6040_xapp',
    # Cell scaling 3g
    '3g_cells5': ROOT / 'n5_L1_alone_3g_cells5',
    '3g_cells10': ROOT / 'n5_L1_alone_3g_cells10',
    '3g_cells20': ROOT / 'n20_L1_alone_3g20gb',  # use N=20 for 20
    '3g_cells40': ROOT / 'n5_L1_alone_3g_cells40',
    # Cell scaling 2g
    '2g_cells5': EXTRA / 'n5_L1_alone_2g_cells5',
    '2g_cells10': EXTRA / 'n5_L1_alone_2g_cells10',
    '2g_cells20': EXTRA / 'n5_L1_alone_2g_cells20',
    '2g_cells40': EXTRA / 'n5_L1_alone_2g_cells40',
    # Cell scaling 4g
    '4g_cells5': EXTRA / 'n5_L1_alone_4g_cells5',
    '4g_cells10': EXTRA / 'n5_L1_alone_4g_cells10',
    '4g_cells40': EXTRA / 'n5_L1_alone_4g_cells40',
    # Multi-AI count
    '3g_3AI_1g': EXTRA / 'n5_L1_3g_3AI_1g',
    '3g_4AI_1g': EXTRA / 'n5_L1_3g_4AI_1g',
}

# Load all
D = {}
for name, p in DS.items():
    runs = load_runs(p) if p.exists() else []
    D[name] = runs
    s = stats(runs)
    if s:
        print(f"{name:30s}: N={s['n']:3d} mean={s['mean']:7.2f} med={s['median']:7.2f} stdev={s['stdev']:5.2f}")
    else:
        print(f"{name:30s}: NO DATA")


def filter_valid_4g_cells40(runs):
    """4g_cells40 has cross-contamination — filter by num_cells field."""
    return [r for r in runs if r.get('num_cells') == 40]


D['4g_cells40_clean'] = filter_valid_4g_cells40(D.get('4g_cells40', []))
print(f"4g_cells40_clean: N={len(D['4g_cells40_clean'])}")


# ============================================================
# Helper: bar plot
# ============================================================
def bar_plot(names, values, ax, ylabel, title, color=None, hline=None, hline_label=None, errors=None):
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=color or '#3a76d8', edgecolor='black', linewidth=0.8)
    if errors is not None:
        ax.errorbar(x, values, yerr=errors, fmt='none', ecolor='black', capsize=4, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if hline is not None:
        ax.axhline(hline, color='red', linestyle='--', linewidth=1.2, label=hline_label or f'{hline:.1f} ms')
        ax.legend(loc='upper left')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + max(values)*0.01, f'{val:.1f}',
                ha='center', va='bottom', fontsize=9)


# ============================================================
# FIG 1: MIG mode itself is free (7g MIG ≈ no-MIG)
# ============================================================
fig, ax = plt.subplots()
configs = ['Full GPU\n(no MIG)', '7g.40gb MIG\n(single instance)']
medians = [stats(D['full_GPU_noMIG'])['median'], stats(D['7g_MIG_alone'])['median']]
errors = [stats(D['full_GPU_noMIG'])['stdev'], stats(D['7g_MIG_alone'])['stdev']]
bar_plot(configs, medians, ax, 'L1 latency (ms, median)',
         'Fig 1: MIG mode itself has negligible overhead',
         color=['#666666', '#3a76d8'], errors=errors)
ax.set_ylim(0, 50)
plt.savefig(OUT / 'fig_01_mig_mode_overhead.png')
plt.close()


# ============================================================
# FIG 2: Partition cap is the dominant cost (L1 alone, no AI)
# ============================================================
fig, ax = plt.subplots()
configs_p = ['Full GPU\n(no MIG)', '7g.40gb\nMIG', '4g.20gb\nMIG', '3g.20gb\nMIG', '2g.10gb\nMIG']
medians_p = [
    stats(D['full_GPU_noMIG'])['median'],
    stats(D['7g_MIG_alone'])['median'],
    stats(D['4g_MIG_clean'])['median'],
    stats(D['3g_MIG_alone'])['median'],
    stats(D['2g_MIG_alone'])['median'],
]
errors_p = [
    stats(D['full_GPU_noMIG'])['stdev'],
    stats(D['7g_MIG_alone'])['stdev'],
    stats(D['4g_MIG_clean'])['stdev'],
    stats(D['3g_MIG_alone'])['stdev'],
    stats(D['2g_MIG_alone'])['stdev'],
]
baseline = medians_p[0]
bar_plot(configs_p, medians_p, ax, 'L1 latency (ms, median)',
         'Fig 2: L1 latency vs MIG partition size (no AI)',
         hline=baseline, hline_label=f'Full GPU baseline ({baseline:.1f} ms)',
         color=['#666', '#3a76d8', '#d8a83a', '#d83a3a', '#a83ad8'], errors=errors_p)
plt.savefig(OUT / 'fig_02_partition_cap.png')
plt.close()


# ============================================================
# FIG 3: Partition cap % overhead vs full GPU
# ============================================================
fig, ax = plt.subplots()
configs_o = configs_p[1:]
medians_o = medians_p[1:]
overhead = [(m - baseline) / baseline * 100 for m in medians_o]
bar_plot(configs_o, overhead, ax, '% slower than no-MIG baseline',
         'Fig 3: Partition cap penalty (vs Full GPU 39 ms)',
         color=['#3a76d8', '#d8a83a', '#d83a3a', '#a83ad8'])
ax.axhline(0, color='red', linestyle='-', linewidth=1)
plt.savefig(OUT / 'fig_03_partition_cap_overhead.png')
plt.close()


# ============================================================
# FIG 4: Cell scaling — main HBM bandwidth theory figure
# ============================================================
fig, ax = plt.subplots()
cells = [5, 10, 20, 40]
for label, key_prefix, color, marker in [
    ('2g.10gb', '2g_cells', '#a83ad8', 'o'),
    ('3g.20gb', '3g_cells', '#d83a3a', 's'),
    ('4g.20gb', '4g_cells', '#d8a83a', '^'),
]:
    vals = []
    for c in cells:
        key = f'{key_prefix}{c}'
        s = stats(D.get(key, []))
        if s is None and c == 40 and key_prefix == '4g_cells':
            s = stats(D['4g_cells40_clean'])
        vals.append(s['median'] if s else None)
    valid_cells = [c for c, v in zip(cells, vals) if v is not None]
    valid_vals = [v for v in vals if v is not None]
    ax.plot(valid_cells, valid_vals, marker=marker, label=label, color=color)
ax.set_xlabel('Number of cells (per L1 iteration)')
ax.set_ylabel('L1 latency (ms, median)')
ax.set_title('Fig 4: Cell scaling reveals HBM bandwidth saturation per partition')
ax.legend(title='L1 partition')
ax.set_xticks(cells)
plt.savefig(OUT / 'fig_04_cell_scaling.png')
plt.close()


# ============================================================
# FIG 5: Cell scaling — per-cell latency (efficiency)
# ============================================================
fig, ax = plt.subplots()
for label, key_prefix, color, marker in [
    ('2g.10gb', '2g_cells', '#a83ad8', 'o'),
    ('3g.20gb', '3g_cells', '#d83a3a', 's'),
    ('4g.20gb', '4g_cells', '#d8a83a', '^'),
]:
    per_cell = []
    valid_cells = []
    for c in cells:
        key = f'{key_prefix}{c}'
        s = stats(D.get(key, []))
        if s is None and c == 40 and key_prefix == '4g_cells':
            s = stats(D['4g_cells40_clean'])
        if s:
            per_cell.append(s['median'] / c)
            valid_cells.append(c)
    ax.plot(valid_cells, per_cell, marker=marker, label=label, color=color)
ax.set_xlabel('Number of cells')
ax.set_ylabel('Latency per cell (ms/cell)')
ax.set_title('Fig 5: Per-cell efficiency — saturation = per-cell latency increases')
ax.legend(title='L1 partition')
ax.set_xticks(cells)
plt.savefig(OUT / 'fig_05_per_cell_efficiency.png')
plt.close()


# ============================================================
# FIG 6: Bimodal — A0 histogram
# ============================================================
fig, ax = plt.subplots()
raw = all_raw(D['A0_qwen_full'])
ax.hist(raw, bins=50, edgecolor='black', alpha=0.7, color='#3a76d8')
ax.set_xlabel('Per-iteration L1 latency (ms)')
ax.set_ylabel('Count')
ax.set_title(f'Fig 6: Bimodal distribution — A0 (3g L1 + Qwen-7B)\nN={len(raw)} iterations across 20 runs')
ax.axvline(np.median(raw), color='red', linestyle='--', label=f'Median {np.median(raw):.2f}')
ax.legend()
plt.savefig(OUT / 'fig_06_bimodal_A0.png')
plt.close()


# ============================================================
# FIG 7: Bimodal — Baseline (no MIG, no AI) histogram
# ============================================================
fig, ax = plt.subplots()
raw = all_raw(D['full_GPU_noMIG'])
ax.hist(raw, bins=50, edgecolor='black', alpha=0.7, color='#666666')
ax.set_xlabel('Per-iteration L1 latency (ms)')
ax.set_ylabel('Count')
ax.set_title(f'Fig 7: Bimodal exists even in baseline (no MIG, no AI)\nN={len(raw)} iterations')
ax.axvline(np.median(raw), color='red', linestyle='--', label=f'Median {np.median(raw):.2f}')
ax.legend()
plt.savefig(OUT / 'fig_07_bimodal_baseline.png')
plt.close()


# ============================================================
# FIG 8: Bimodal comparison — overlay 3 configs
# ============================================================
fig, ax = plt.subplots()
for name, color, alpha in [
    ('full_GPU_noMIG', '#666', 0.5),
    ('3g_MIG_alone', '#3a76d8', 0.5),
    ('A0_qwen_full', '#d83a3a', 0.5),
]:
    raw = all_raw(D[name])
    label = name.replace('_', ' ') + f' (med {np.median(raw):.1f})'
    ax.hist(raw, bins=50, alpha=alpha, label=label, density=True)
ax.set_xlabel('L1 latency (ms)')
ax.set_ylabel('Density')
ax.set_title('Fig 8: All configurations show bimodal — intrinsic to cuPHY')
ax.legend()
plt.savefig(OUT / 'fig_08_bimodal_overlay.png')
plt.close()


# ============================================================
# FIG 9: Phase 1 — Qwen variants on 3g (H1 phase hypothesis test)
# ============================================================
fig, ax = plt.subplots()
configs = ['3g\nalone', 'A0\nQwen full', 'A1a\nprefill', 'A1b\ndecode', 'A2\nstatic HBM']
medians = [
    stats(D['3g_MIG_alone'])['median'],
    stats(D['A0_qwen_full'])['median'],
    stats(D['A1_prefill'])['median'],
    stats(D['A1_decode'])['median'],
    stats(D['A2_hbm_static'])['median'],
]
errors = [
    stats(D['3g_MIG_alone'])['stdev'],
    stats(D['A0_qwen_full'])['stdev'],
    stats(D['A1_prefill'])['stdev'],
    stats(D['A1_decode'])['stdev'],
    stats(D['A2_hbm_static'])['stdev'],
]
bar_plot(configs, medians, ax, 'L1 latency (ms, median)',
         'Fig 9: Phase 1 — All AI types converge to ~55 ms (H1 phase rejected)',
         hline=medians[0], hline_label=f'3g L1 alone ({medians[0]:.1f} ms)',
         color=['#3a76d8'] + ['#d8a83a']*4, errors=errors)
plt.savefig(OUT / 'fig_09_phase1_qwen_variants.png')
plt.close()


# ============================================================
# FIG 10: Phase 4 vs Phase 1 — real AI-RAN vs LLM
# ============================================================
fig, ax = plt.subplots()
configs = ['3g\nalone', 'A0\n(Qwen LLM)', 'AR1\n(NeuralRx)', 'AR2\n(ChanPred)', 'AR3\n(xApp)']
medians = [
    stats(D['3g_MIG_alone'])['median'],
    stats(D['A0_qwen_full'])['median'],
    stats(D['AR1_neuralrx'])['median'],
    stats(D['AR2_chanpred'])['median'],
    stats(D['AR3_xapp'])['median'],
]
errors = [
    stats(D['3g_MIG_alone'])['stdev'],
    stats(D['A0_qwen_full'])['stdev'],
    stats(D['AR1_neuralrx'])['stdev'],
    stats(D['AR2_chanpred'])['stdev'],
    stats(D['AR3_xapp'])['stdev'],
]
colors = ['#666', '#3a76d8', '#d83a3a', '#d83a3a', '#d83a3a']
bar_plot(configs, medians, ax, 'L1 latency (ms, median)',
         'Fig 10: Real AI-RAN workloads (TensorRT) >>> LLM (Qwen) impact',
         hline=medians[0], hline_label=f'3g L1 alone ({medians[0]:.1f} ms)',
         color=colors, errors=errors)
plt.savefig(OUT / 'fig_10_airan_vs_llm.png')
plt.close()


# ============================================================
# FIG 11: L1 latency decomposition (stacked bar)
# ============================================================
fig, ax = plt.subplots()
configs = ['L1 alone\n(7g/no MIG)', '+ Partition\ncap (3g)', '+ 1 AI\n(Qwen)', '+ 2 AI\n(M1)', '+ Real AI-RAN\n(NeuralRx)']
baseline_v = stats(D['full_GPU_noMIG'])['median']
cap_v = stats(D['3g_MIG_alone'])['median'] - baseline_v
a0_v = stats(D['A0_qwen_full'])['median'] - stats(D['3g_MIG_alone'])['median']
m1_v = stats(D['M1_3way_balanced'])['median'] - stats(D['A0_qwen_full'])['median']
ar1_v = stats(D['AR1_neuralrx'])['median'] - stats(D['A0_qwen_full'])['median']
x = np.arange(5)
bars1 = ax.bar(x, [baseline_v]*5, label='Baseline (cuPHY)', color='#888')
bars2 = ax.bar(x, [0, cap_v, cap_v, cap_v, cap_v], bottom=[baseline_v]*5, label='+ Partition cap', color='#d83a3a')
bars3 = ax.bar(x, [0, 0, a0_v, a0_v + m1_v, a0_v + ar1_v - a0_v], bottom=[baseline_v]*1 + [baseline_v+cap_v]*4, label='+ AI leakage', color='#d8a83a')
# Top of M1: cap+a0+m1; AR1: cap+ar1
heights = [baseline_v, baseline_v+cap_v, baseline_v+cap_v+a0_v, baseline_v+cap_v+a0_v+m1_v, baseline_v+cap_v+ar1_v]
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=20, ha='right')
ax.set_ylabel('L1 latency (ms)')
ax.set_title('Fig 11: L1 latency decomposition — baseline + cap + AI leakage')
ax.legend(loc='upper left')
for xi, h in zip(x, heights):
    ax.text(xi, h + 1, f'{h:.1f}', ha='center', va='bottom', fontsize=9)
plt.savefig(OUT / 'fig_11_decomposition_stacked.png')
plt.close()


# ============================================================
# FIG 12: Multi-AI count on 3g — AI partition size matters
# ============================================================
fig, ax = plt.subplots()
configs = ['3g\nalone\n(0 AI)', '+1 Qwen\non 2g (A0)', '+2 AI\non 2g+2g (M1)', '+3 AI\non 3× 1g', '+4 AI\non 4× 1g']
medians = [
    stats(D['3g_MIG_alone'])['median'],
    stats(D['A0_qwen_full'])['median'],
    stats(D['M1_3way_balanced'])['median'],
    stats(D['3g_3AI_1g'])['median'],
    stats(D['3g_4AI_1g'])['median'],
]
colors = ['#666', '#3a76d8', '#d83a3a', '#3aa83a', '#3aa83a']
bar_plot(configs, medians, ax, 'L1 latency (ms, median)',
         'Fig 12: Multi-AI on 3g L1 — partition SIZE matters more than COUNT',
         hline=medians[0], hline_label=f'3g L1 alone ({medians[0]:.1f} ms)',
         color=colors)
plt.savefig(OUT / 'fig_12_multi_ai_count.png')
plt.close()


# ============================================================
# FIG 13: Phase 2 multi-partition comparison
# ============================================================
fig, ax = plt.subplots()
configs = ['M1\n3g L1 + 2×2g AI', 'M2\n2g L1 + 3g+2g AI', 'M3\n4g L1 + 1g+2g AI', 'M4\n4g L1 + 3×1g AI']
medians = [
    stats(D['M1_3way_balanced'])['median'],
    stats(D['M2_3way_L1small'])['median'],
    stats(D['M3_3way_asym'])['median'],
    stats(D['M4_4way_3xApp'])['median'],
]
errors = [
    stats(D['M1_3way_balanced'])['stdev'],
    stats(D['M2_3way_L1small'])['stdev'],
    stats(D['M3_3way_asym'])['stdev'],
    stats(D['M4_4way_3xApp'])['stdev'],
]
bar_plot(configs, medians, ax, 'L1 latency (ms, median)',
         'Fig 13: Phase 2 — Multi-partition AI-RAN configurations',
         hline=stats(D['full_GPU_noMIG'])['median'], hline_label='Full GPU baseline',
         color=['#d83a3a', '#a83ad8', '#d8a83a', '#3aa83a'], errors=errors)
plt.savefig(OUT / 'fig_13_phase2_multipartition.png')
plt.close()


# ============================================================
# FIG 14: Phase 3 D1 — partition cap vs AI leakage separation
# ============================================================
fig, ax = plt.subplots()
configs = ['Full GPU\nbaseline', '3g L1\nalone', '3g L1\n+ Qwen', '2g L1\nalone (D1b)', '2g L1\n+ Qwen (D1a)']
medians = [
    stats(D['full_GPU_noMIG'])['median'],
    stats(D['3g_MIG_alone'])['median'],
    stats(D['A0_qwen_full'])['median'],
    stats(D['D1b_split40_60_alone'])['median'],
    stats(D['D1a_split40_60_qwen'])['median'],
]
bar_plot(configs, medians, ax, 'L1 latency (ms, median)',
         'Fig 14: D1 decomposition — partition cap dominates, AI leakage small',
         color=['#666', '#3a76d8', '#d8a83a', '#a83ad8', '#d83a3a'])
plt.savefig(OUT / 'fig_14_D1_decomposition.png')
plt.close()


# ============================================================
# FIG 15: Tail latency (p99) — URLLC infeasibility
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
configs = ['Full GPU', '7g MIG', '3g alone', 'A0\n+Qwen', 'M1\n+2 AI', 'M4\n+3 light', 'AR1\nNeuralRx', 'M2\n2g L1']
p99s = [
    np.median([r['p99_ms'] for r in D['full_GPU_noMIG']]),
    np.median([r['p99_ms'] for r in D['7g_MIG_alone']]),
    np.median([r['p99_ms'] for r in D['3g_MIG_alone']]),
    np.median([r['p99_ms'] for r in D['A0_qwen_full']]),
    np.median([r['p99_ms'] for r in D['M1_3way_balanced']]),
    np.median([r['p99_ms'] for r in D['M4_4way_3xApp']]),
    np.median([r['p99_ms'] for r in D['AR1_neuralrx']]),
    np.median([r['p99_ms'] for r in D['M2_3way_L1small']]),
]
bar_plot(configs, p99s, ax, 'p99 L1 latency (ms)',
         'Fig 15: p99 tail latency — URLLC 1ms requirement infeasible in all configs',
         hline=1.0, hline_label='URLLC target (1 ms)',
         color=['#666', '#3a76d8', '#d8a83a', '#d8a83a', '#d83a3a', '#3aa83a', '#d83a3a', '#a83ad8'])
plt.savefig(OUT / 'fig_15_p99_urllc.png')
plt.close()


# ============================================================
# FIG 16: CDF comparison — baseline vs MIG vs MIG+AI
# ============================================================
fig, ax = plt.subplots()
for name, color, label in [
    ('full_GPU_noMIG', '#666', 'Full GPU baseline'),
    ('3g_MIG_alone', '#3a76d8', '3g MIG alone'),
    ('A0_qwen_full', '#d8a83a', '3g + Qwen (A0)'),
    ('M1_3way_balanced', '#d83a3a', '3g + 2 AI (M1)'),
    ('AR1_neuralrx', '#a83ad8', '3g + NeuralRx (AR1)'),
]:
    raw = np.sort(all_raw(D[name]))
    cdf = np.arange(1, len(raw)+1) / len(raw)
    ax.plot(raw, cdf, label=label, color=color, linewidth=1.5)
ax.set_xlabel('L1 latency (ms)')
ax.set_ylabel('CDF')
ax.set_title('Fig 16: CDF — tail distributions across configurations')
ax.legend(loc='lower right')
ax.set_xlim(0, 200)
plt.savefig(OUT / 'fig_16_cdf_comparison.png')
plt.close()


# ============================================================
# FIG 17: Q-Q plot for bimodal evidence
# ============================================================
fig, ax = plt.subplots()
data = all_raw(D['A0_qwen_full'])
data_sorted = np.sort(data)
# normal quantiles
p = (np.arange(1, len(data_sorted)+1) - 0.5) / len(data_sorted)
# Inverse CDF via numpy approximation
from numpy import sqrt, log, pi
def normq(p):
    """Approx normal quantile."""
    return np.sqrt(2) * np.array([np.sign(2*pi - 1) * np.sqrt(-np.log(4*pi*(1-pi))) for pi in p])
# Simpler: use scipy if available, else skip
try:
    from scipy import stats as sps
    q = sps.norm.ppf(p, loc=np.mean(data), scale=np.std(data))
    ax.scatter(q, data_sorted, alpha=0.3, s=10, color='#3a76d8')
    ax.plot([q.min(), q.max()], [q.min(), q.max()], 'r--', linewidth=1)
    ax.set_xlabel('Theoretical normal quantile (ms)')
    ax.set_ylabel('Observed L1 latency (ms)')
    ax.set_title('Fig 17: Q-Q plot — A0 deviates from normal (S-curve = bimodal)')
except ImportError:
    ax.text(0.5, 0.5, 'scipy not available', ha='center', va='center', transform=ax.transAxes)
plt.savefig(OUT / 'fig_17_qq_plot.png')
plt.close()


# ============================================================
# FIG 18: Per-config p99 / median ratio (tail amplification)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
all_keys = ['full_GPU_noMIG', '7g_MIG_alone', '3g_MIG_alone', '2g_MIG_alone',
            'A0_qwen_full', 'M1_3way_balanced', 'M4_4way_3xApp',
            'AR1_neuralrx', 'AR2_chanpred', 'AR3_xapp']
labels_p99 = ['Full GPU', '7g MIG', '3g alone', '2g alone', 'A0', 'M1', 'M4',
              'AR1', 'AR2', 'AR3']
ratios = []
for k in all_keys:
    runs = D[k]
    p99s = [r['p99_ms'] for r in runs]
    means = [r['mean_ms'] for r in runs]
    ratios.append(np.median(p99s) / np.median(means))
bar_plot(labels_p99, ratios, ax, 'p99 / mean ratio',
         'Fig 18: Tail amplification factor across configurations',
         hline=1.0, hline_label='No tail (=1)',
         color='#d83a3a')
plt.savefig(OUT / 'fig_18_tail_ratio.png')
plt.close()


# ============================================================
# FIG 19: AI-RAN best vs worst case + URLLC line
# ============================================================
fig, ax = plt.subplots()
configs = ['Full GPU\n(impossible\nwith AI)', 'Best AI-RAN\n(M4: 4g+3 light)', 'Typical AI-RAN\n(A0: 3g+Qwen)', 'Heavy AI-RAN\n(M1: 3g+2 AI)', 'Real AI-RAN\n(AR1: NeuralRx)', 'Catastrophic\n(M2: 2g L1)']
medians = [
    stats(D['full_GPU_noMIG'])['median'],
    stats(D['M4_4way_3xApp'])['median'],
    stats(D['A0_qwen_full'])['median'],
    stats(D['M1_3way_balanced'])['median'],
    stats(D['AR1_neuralrx'])['median'],
    stats(D['M2_3way_L1small'])['median'],
]
p99s = [
    np.median([r['p99_ms'] for r in D['full_GPU_noMIG']]),
    np.median([r['p99_ms'] for r in D['M4_4way_3xApp']]),
    np.median([r['p99_ms'] for r in D['A0_qwen_full']]),
    np.median([r['p99_ms'] for r in D['M1_3way_balanced']]),
    np.median([r['p99_ms'] for r in D['AR1_neuralrx']]),
    np.median([r['p99_ms'] for r in D['M2_3way_L1small']]),
]
x = np.arange(len(configs))
w = 0.35
ax.bar(x - w/2, medians, w, label='median', color='#3a76d8')
ax.bar(x + w/2, p99s, w, label='p99', color='#d83a3a')
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('L1 latency (ms)')
ax.set_title('Fig 19: AI-RAN configuration spectrum — median + p99')
ax.legend()
plt.savefig(OUT / 'fig_19_airan_spectrum.png')
plt.close()


# ============================================================
# FIG 20: Summary panel — overhead breakdown vs configurations
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
baseline = stats(D['full_GPU_noMIG'])['median']

scenarios = [
    ('Full GPU\n(baseline)', 0, 0),
    ('7g MIG\nsingle', stats(D['7g_MIG_alone'])['median'] - baseline, 0),
    ('3g MIG\nalone', stats(D['3g_MIG_alone'])['median'] - baseline, 0),
    ('3g + Qwen\n(A0)', stats(D['3g_MIG_alone'])['median'] - baseline,
     stats(D['A0_qwen_full'])['median'] - stats(D['3g_MIG_alone'])['median']),
    ('3g + 2 AI\n(M1)', stats(D['3g_MIG_alone'])['median'] - baseline,
     stats(D['M1_3way_balanced'])['median'] - stats(D['3g_MIG_alone'])['median']),
    ('3g + Real\nAI-RAN (AR1)', stats(D['3g_MIG_alone'])['median'] - baseline,
     stats(D['AR1_neuralrx'])['median'] - stats(D['3g_MIG_alone'])['median']),
    ('Best AI-RAN\n4g + 3 light (M4)', stats(D['4g_MIG_clean'])['median'] - baseline,
     stats(D['M4_4way_3xApp'])['median'] - stats(D['4g_MIG_clean'])['median']),
    ('Worst case\n2g L1 (M2)', stats(D['2g_MIG_alone'])['median'] - baseline,
     stats(D['M2_3way_L1small'])['median'] - stats(D['2g_MIG_alone'])['median']),
]
labels = [s[0] for s in scenarios]
caps = [s[1] for s in scenarios]
leakages = [s[2] for s in scenarios]
x = np.arange(len(labels))
ax.bar(x, [baseline]*len(labels), label='cuPHY baseline (no MIG)', color='#888')
ax.bar(x, caps, bottom=[baseline]*len(labels), label='Partition cap', color='#d83a3a')
ax.bar(x, leakages, bottom=[baseline+c for c in caps], label='AI leakage', color='#d8a83a')
ax.axhline(baseline, color='red', linestyle='--', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('L1 latency (ms)')
ax.set_title('Fig 20: L1 latency breakdown across AI-RAN configurations\n(cuPHY baseline = 39 ms, dashed line)')
ax.legend(loc='upper left')
totals = [baseline + c + l for c, l in zip(caps, leakages)]
for xi, t in zip(x, totals):
    ax.text(xi, t + 2, f'{t:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.savefig(OUT / 'fig_20_overall_breakdown.png')
plt.close()


print(f"\nDONE: {len(list(OUT.glob('fig_*.png')))} figures generated in {OUT}")
