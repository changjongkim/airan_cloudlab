#!/usr/bin/env python3
"""
Additional figures from raw_ms data (since dmon.csv not available).
Generates fig_24 - fig_35 covering:
- Run-over-run trends
- Iteration-level distributions
- Bimodal classification
- Box/violin plots
- Per-config density curves
"""
import json
import glob
import statistics
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

mpl.rcParams.update({
    'font.size': 10, 'figure.dpi': 100, 'savefig.dpi': 140,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3,
})

ROOT = Path('.')
EXTRA = ROOT / 'extra_2g_4g_cells'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)


def load(d):
    return [json.load(open(f)) for f in sorted(glob.glob(str(d / 'run_*.json')))]


DS = {
    'Full GPU': ROOT / 'n20_baseline_gpu1_fullGPU',
    '7g MIG': ROOT / 'n10_L1_alone_7g40gb_MIG',
    '4g MIG': ROOT / 'n10_L1_alone_4g20gb_clean',
    '3g MIG': ROOT / 'n20_L1_alone_3g20gb',
    '2g MIG': ROOT / 'n10_D1b_4060_alone',
    'A0 +Qwen': ROOT / 'n20_A0_qwen_baseline',
    'A1a +prefill': ROOT / 'n20_A1_prefill',
    'A1b +decode': ROOT / 'n20_A1_decode',
    'A2 +HBM': ROOT / 'n20_A2_hbm',
    'M1 +2 AI': ROOT / 'n10_M1_3way_balanced_AIRAN',
    'M2 (2g L1)': ROOT / 'n10_M2_3way_L1small_mixed',
    'M3 (4g+2 AI)': ROOT / 'n10_M3_3way_asym_AIRAN',
    'M4 (4g+3 light)': ROOT / 'n10_M4_4way_3xApp',
    'AR1 NeuralRx': ROOT / 'n10_AR1_6040_neuralrx',
    'AR2 ChanPred': ROOT / 'n10_AR2_6040_chanpred',
    'AR3 xApp': ROOT / 'n10_AR3_6040_xapp',
    '3g +3 AI (1g)': EXTRA / 'n5_L1_3g_3AI_1g',
    '3g +4 AI (1g)': EXTRA / 'n5_L1_3g_4AI_1g',
}

D = {k: load(p) for k, p in DS.items() if p.exists()}


# ============================================================
# FIG 24: Run-over-run progression (mean of each run, time order)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
groups = [
    ('Baselines', ['Full GPU', '7g MIG', '4g MIG', '3g MIG', '2g MIG']),
    ('Phase 1 (3g + Qwen variants)', ['3g MIG', 'A0 +Qwen', 'A1a +prefill', 'A1b +decode', 'A2 +HBM']),
    ('Phase 4 (Real AI-RAN)', ['3g MIG', 'A0 +Qwen', 'AR1 NeuralRx', 'AR2 ChanPred', 'AR3 xApp']),
    ('Multi-AI count', ['3g MIG', 'A0 +Qwen', 'M1 +2 AI', '3g +3 AI (1g)', '3g +4 AI (1g)']),
]
for ax, (title, keys) in zip(axes.flat, groups):
    for k in keys:
        if k not in D:
            continue
        means = [r['mean_ms'] for r in D[k] if 'mean_ms' in r]
        ax.plot(range(1, len(means)+1), means, marker='o', label=k, linewidth=1.5)
    ax.set_xlabel('Run index')
    ax.set_ylabel('L1 mean (ms)')
    ax.set_title(title)
    ax.legend(fontsize=8)
plt.suptitle('Fig 24: Run-over-run mean L1 latency — stability check', fontsize=12)
plt.tight_layout()
plt.savefig(OUT / 'fig_24_run_progression.png')
plt.close()


# ============================================================
# FIG 25: Box plot — distribution per config
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6))
keys = ['Full GPU', '7g MIG', '4g MIG', '3g MIG', '2g MIG',
        'A0 +Qwen', 'M1 +2 AI', 'M4 (4g+3 light)',
        'AR1 NeuralRx', 'AR2 ChanPred', 'AR3 xApp']
data = []
labels = []
for k in keys:
    if k not in D:
        continue
    all_iter = []
    for r in D[k]:
        all_iter.extend(r.get('raw_ms', []))
    data.append(all_iter)
    labels.append(f"{k}\nN={len(all_iter)}")
bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#9bc4f2')
plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=8)
ax.set_ylabel('L1 latency per iteration (ms)')
ax.set_title('Fig 25: Per-iteration L1 latency distribution (box plot, fliers hidden)')
ax.axhline(stats := np.median([r['mean_ms'] for r in D['Full GPU']]), color='red', linestyle='--',
           linewidth=1, label=f'Full GPU median ({stats:.1f})')
ax.legend()
plt.savefig(OUT / 'fig_25_boxplot.png')
plt.close()


# ============================================================
# FIG 26: Violin plot — distribution shape per config
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6))
parts = ax.violinplot(data, showmeans=True, showmedians=True, widths=0.7)
for body in parts['bodies']:
    body.set_facecolor('#9bc4f2')
    body.set_alpha(0.7)
ax.set_xticks(range(1, len(labels)+1))
ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
ax.set_ylabel('L1 latency per iteration (ms)')
ax.set_title('Fig 26: Per-iteration L1 latency density (violin — bimodal shape visible)')
plt.savefig(OUT / 'fig_26_violin.png')
plt.close()


# ============================================================
# FIG 27: Iteration index vs latency (intra-run pattern)
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))
for k, color in [('Full GPU', '#666'), ('3g MIG', '#3a76d8'),
                  ('A0 +Qwen', '#d8a83a'), ('AR1 NeuralRx', '#d83a3a')]:
    if k not in D:
        continue
    # Stack raw_ms from all runs aligned by iteration index
    all_runs = [r['raw_ms'] for r in D[k] if 'raw_ms' in r]
    max_len = max(len(r) for r in all_runs)
    arr = np.full((len(all_runs), max_len), np.nan)
    for i, r in enumerate(all_runs):
        arr[i, :len(r)] = r
    medians = np.nanmedian(arr, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    iter_idx = np.arange(max_len)
    ax.plot(iter_idx, medians, label=k, color=color, linewidth=1.5)
    ax.fill_between(iter_idx, q25, q75, alpha=0.2, color=color)
ax.set_xlabel('Iteration index (within run)')
ax.set_ylabel('L1 latency (ms, median across runs ± IQR shaded)')
ax.set_title('Fig 27: Latency vs iteration index — drift/warmup check')
ax.legend()
plt.savefig(OUT / 'fig_27_iter_index.png')
plt.close()


# ============================================================
# FIG 28: Bimodal classification — % iterations in HIGH mode
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6))
keys = list(D.keys())
high_pcts = []
labels_b = []
for k in keys:
    all_iter = []
    for r in D[k]:
        all_iter.extend(r.get('raw_ms', []))
    if not all_iter:
        continue
    arr = np.array(all_iter)
    # Two-cluster split via 1D k-means
    sorted_arr = np.sort(arr)
    # initial centroids
    c_low, c_high = sorted_arr[len(arr)//4], sorted_arr[3*len(arr)//4]
    for _ in range(30):
        d_low = np.abs(arr - c_low)
        d_high = np.abs(arr - c_high)
        low_mask = d_low <= d_high
        if low_mask.sum() == 0 or low_mask.sum() == len(arr):
            break
        c_low = arr[low_mask].mean()
        c_high = arr[~low_mask].mean()
    high_pct = (~low_mask).sum() / len(arr) * 100
    high_pcts.append(high_pct)
    labels_b.append(f"{k}\n(L={c_low:.1f}/H={c_high:.1f})")
x = np.arange(len(labels_b))
ax.bar(x, high_pcts, color='#d83a3a', edgecolor='black')
ax.axhline(50, color='gray', linestyle='--', label='50% balanced')
ax.set_xticks(x)
ax.set_xticklabels(labels_b, rotation=25, ha='right', fontsize=7)
ax.set_ylabel('% iterations in HIGH cluster')
ax.set_title('Fig 28: Bimodal balance — fraction of iterations in HIGH mode')
ax.legend()
ax.set_ylim(0, 100)
plt.savefig(OUT / 'fig_28_bimodal_balance.png')
plt.close()


# ============================================================
# FIG 29: Bimodal gap (HIGH - LOW centroid) per config
# ============================================================
fig, ax = plt.subplots(figsize=(13, 5))
keys = list(D.keys())
gaps = []
labels_g = []
for k in keys:
    all_iter = []
    for r in D[k]:
        all_iter.extend(r.get('raw_ms', []))
    if not all_iter:
        continue
    arr = np.array(all_iter)
    sorted_arr = np.sort(arr)
    c_low, c_high = sorted_arr[len(arr)//4], sorted_arr[3*len(arr)//4]
    for _ in range(30):
        d_low = np.abs(arr - c_low)
        d_high = np.abs(arr - c_high)
        low_mask = d_low <= d_high
        if low_mask.sum() == 0 or low_mask.sum() == len(arr):
            break
        c_low = arr[low_mask].mean()
        c_high = arr[~low_mask].mean()
    gaps.append(c_high - c_low)
    labels_g.append(k)
x = np.arange(len(labels_g))
ax.bar(x, gaps, color='#3a76d8', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(labels_g, rotation=25, ha='right', fontsize=8)
ax.set_ylabel('HIGH - LOW cluster gap (ms)')
ax.set_title('Fig 29: Bimodal gap magnitude per configuration')
plt.savefig(OUT / 'fig_29_bimodal_gap.png')
plt.close()


# ============================================================
# FIG 30: Run-to-run variance vs iteration variance (where noise lives)
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6))
keys_v = list(D.keys())
across_run = []  # stdev of run means
within_run = []  # mean of within-run stdev
labels_v = []
for k in keys_v:
    means = [r['mean_ms'] for r in D[k] if 'mean_ms' in r]
    if len(means) < 2:
        continue
    across_run.append(statistics.stdev(means))
    within_stds = []
    for r in D[k]:
        if 'raw_ms' in r and len(r['raw_ms']) > 1:
            within_stds.append(statistics.stdev(r['raw_ms']))
    within_run.append(statistics.mean(within_stds) if within_stds else 0)
    labels_v.append(k)
x = np.arange(len(labels_v))
w = 0.4
ax.bar(x - w/2, across_run, w, label='across-run σ (of run means)', color='#3a76d8')
ax.bar(x + w/2, within_run, w, label='within-run σ (of iterations)', color='#d83a3a')
ax.set_xticks(x)
ax.set_xticklabels(labels_v, rotation=25, ha='right', fontsize=8)
ax.set_ylabel('Standard deviation (ms)')
ax.set_title('Fig 30: Variance source — across-run vs within-run')
ax.legend()
plt.savefig(OUT / 'fig_30_variance_source.png')
plt.close()


# ============================================================
# FIG 31: Per-cell latency stacked (cell scaling sanity)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
cells_data = {
    '2g.10gb': {5: 18.24, 10: 41.60, 20: 76.81, 40: 141.42},
    '3g.20gb': {5: 18.29, 10: 36.09, 20: 52.54, 40: 147.98},
    '4g.20gb': {5: 17.19, 10: 29.84, 20: 56.50},
}
for label, vals in cells_data.items():
    cells = sorted(vals.keys())
    per_cell = [vals[c] / c for c in cells]
    ax.plot(cells, per_cell, marker='o', label=f'{label}', linewidth=2)
ax.set_xlabel('Number of cells')
ax.set_ylabel('Latency per cell (ms/cell)')
ax.set_title('Fig 31: Per-cell efficiency — saturation point (rise) per partition')
ax.legend()
ax.set_xticks([5, 10, 20, 40])
ax.axhline(0.125, color='black', linestyle='-.', linewidth=1, label='cuPHY production target')
ax.legend()
plt.savefig(OUT / 'fig_31_per_cell_scaling.png')
plt.close()


# ============================================================
# FIG 32: Heatmap — partition × cells (gap to baseline)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
partitions = ['2g.10gb', '3g.20gb', '4g.20gb']
cells_labels = ['cells=5', 'cells=10', 'cells=20', 'cells=40']
heat = [
    [18.24, 41.60, 76.81, 141.42],
    [18.29, 36.09, 52.54, 147.98],
    [17.19, 29.84, 56.50, np.nan],
]
heat = np.array(heat)
im = ax.imshow(heat, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(cells_labels)))
ax.set_xticklabels(cells_labels)
ax.set_yticks(range(len(partitions)))
ax.set_yticklabels(partitions)
for i in range(len(partitions)):
    for j in range(len(cells_labels)):
        v = heat[i, j]
        txt = f'{v:.0f}' if not np.isnan(v) else 'N/A'
        ax.text(j, i, txt, ha='center', va='center', color='black', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax, label='L1 latency (ms)')
ax.set_title('Fig 32: L1 latency heatmap — partition × cells')
plt.savefig(OUT / 'fig_32_heatmap.png')
plt.close()


# ============================================================
# FIG 33: Density (KDE-like) overlay for top 5 configs
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))
keys_d = ['Full GPU', '7g MIG', '3g MIG', 'A0 +Qwen', 'M1 +2 AI', 'AR1 NeuralRx', 'M2 (2g L1)']
colors_d = ['#666', '#3a76d8', '#d8a83a', '#3aa83a', '#d83a3a', '#a83ad8', '#000']
for k, c in zip(keys_d, colors_d):
    if k not in D:
        continue
    all_iter = []
    for r in D[k]:
        all_iter.extend(r.get('raw_ms', []))
    if not all_iter:
        continue
    ax.hist(all_iter, bins=80, density=True, alpha=0.3, color=c, label=f'{k} (med {np.median(all_iter):.1f})')
ax.set_xlabel('L1 latency (ms)')
ax.set_ylabel('Density')
ax.set_title('Fig 33: Latency density across configurations (full data)')
ax.set_xlim(0, 200)
ax.legend(loc='upper right', fontsize=9)
plt.savefig(OUT / 'fig_33_density_overlay.png')
plt.close()


# ============================================================
# FIG 34: p50/p95/p99 percentile spread per config
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6))
keys_p = ['Full GPU', '7g MIG', '4g MIG', '3g MIG', '2g MIG',
          'A0 +Qwen', 'M1 +2 AI', 'M4 (4g+3 light)',
          'AR1 NeuralRx', 'AR2 ChanPred', 'AR3 xApp']
p50 = []; p95 = []; p99 = []; labels_p = []
for k in keys_p:
    if k not in D:
        continue
    all_iter = []
    for r in D[k]:
        all_iter.extend(r.get('raw_ms', []))
    p50.append(np.percentile(all_iter, 50))
    p95.append(np.percentile(all_iter, 95))
    p99.append(np.percentile(all_iter, 99))
    labels_p.append(k)
x = np.arange(len(labels_p))
w = 0.27
ax.bar(x - w, p50, w, label='p50', color='#3a76d8')
ax.bar(x, p95, w, label='p95', color='#d8a83a')
ax.bar(x + w, p99, w, label='p99', color='#d83a3a')
ax.set_xticks(x)
ax.set_xticklabels(labels_p, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('L1 latency (ms)')
ax.set_title('Fig 34: Percentile spread (p50/p95/p99) per configuration')
ax.legend()
plt.savefig(OUT / 'fig_34_percentile_spread.png')
plt.close()


# ============================================================
# FIG 35: AI workload disruption — Δ vs alone (sorted)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
alone_3g = statistics.median([r['mean_ms'] for r in D['3g MIG']])
ai_configs = [
    ('Qwen full (A0)', 'A0 +Qwen'),
    ('Qwen prefill (A1a)', 'A1a +prefill'),
    ('Qwen decode (A1b)', 'A1b +decode'),
    ('HBM static (A2)', 'A2 +HBM'),
    ('2 AI on 2g (M1)', 'M1 +2 AI'),
    ('NeuralRx (AR1)', 'AR1 NeuralRx'),
    ('ChanPred (AR2)', 'AR2 ChanPred'),
    ('xApp (AR3)', 'AR3 xApp'),
    ('3 AI on 1g', '3g +3 AI (1g)'),
    ('4 AI on 1g', '3g +4 AI (1g)'),
]
deltas = []
labels_dl = []
for label, k in ai_configs:
    if k not in D:
        continue
    m = statistics.median([r['mean_ms'] for r in D[k]])
    deltas.append(m - alone_3g)
    labels_dl.append(label)
order = np.argsort(deltas)
deltas_s = [deltas[i] for i in order]
labels_s = [labels_dl[i] for i in order]
colors_s = ['#3aa83a' if d < 5 else '#d8a83a' if d < 15 else '#d83a3a' for d in deltas_s]
x = np.arange(len(labels_s))
ax.barh(x, deltas_s, color=colors_s, edgecolor='black')
ax.set_yticks(x)
ax.set_yticklabels(labels_s, fontsize=9)
ax.set_xlabel('Δ L1 latency (ms) vs 3g L1 alone (52.5 ms)')
ax.set_title('Fig 35: AI co-location L1 disruption ranking')
ax.axvline(0, color='red', linewidth=1)
for xi, v in zip(x, deltas_s):
    ax.text(v + 0.5, xi, f'{v:+.1f}', va='center', fontsize=9)
plt.savefig(OUT / 'fig_35_ai_disruption_rank.png')
plt.close()


print(f"DONE: {len(list(OUT.glob('fig_*.png')))} total figures in {OUT}")
print(f"New figures: 24-35")
