#!/usr/bin/env python3
"""Comprehensive Stage 3 nsight analysis — 16 scenarios × 28 metrics."""
import csv
import statistics
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent / "nsight_csv"

# Logical grouping for output
METRIC_GROUPS = {
    "Compute": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "smsp__warps_active.avg.pct_of_peak_sustained_elapsed",
    ],
    "L1 cache": [
        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    ],
    "L2 cache": [
        "lts__t_request_hit_rate.pct",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__t_bytes.sum",
        "lts__t_bytes_op_read.sum",
        "lts__t_bytes_op_write.sum",
    ],
    "DRAM": [
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.max.pct_of_peak_sustained_elapsed",
        "dram__cycles_active.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__sectors_read.sum",
        "dram__sectors_write.sum",
        "dram__sectors_op_atom.sum",
    ],
    "Warp stalls (per-warp-active %)": [
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
        "smsp__warp_issue_stalled_membar_per_warp_active.pct",
        "smsp__warp_issue_stalled_pipe_busy_per_warp_active.pct",
        "smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct",
        "smsp__warp_issue_stalled_drain_per_warp_active.pct",
    ],
    "Scheduling/Timing": [
        "smsp__inst_executed_pipe_lsu.sum",
        "gpc__cycles_elapsed.avg",
        "gpu__time_duration.sum",
    ],
}

# Scenario groupings
SCENARIO_DESC = {
    "S2_7g_mig": "7g MIG single (L1 alone, 98 SMs)",
    "S5_3g_alone": "3g L1 alone (split-60-40)",
    "S6_3g_qwen": "3g L1 + Qwen-7B on 2g",
    "S7_3g_neuralrx": "3g L1 + NeuralRx on 2g",
    "S9_3g_3AI_1g": "3g L1 + 3 small AI on 1g×3",
    "S10_2g_alone": "2g L1 alone (split-40-60)",
    "S12_2g_2AI": "2g L1 + 2 AI on 3g+2g",
    "S13_3g_sat_compute": "3g L1 + sat_compute on 2g (saturating)",
    "S14_3g_sat_hbm": "3g L1 + sat_hbm on 2g (HBM saturated)",
    "S15_4g_sat_compute": "4g L1 + sat_compute on 2g",
    "S17_2g_sat_compute": "2g L1 + sat_compute on 3g",
    "S18_4g_neuralrx": "4g L1 + NeuralRx on 2g",
    "S21_4g_2sat": "4g L1 + 2 sat (2g+1g)",
    "S22_2g_neuralrx": "2g L1 + NeuralRx on 3g",
    "S24_3g_2sat": "3g L1 + 2 sat on 2g+2g (M5a)",
    "S26_4g_3sat": "4g L1 + 3 sat on 1g×3 (M7a worst)",
}


def parse(path):
    metrics = {}
    if not os.path.exists(path):
        return metrics
    with open(path) as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"'):
            header_idx = i
            break
    if header_idx is None:
        return metrics
    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        mn = row.get("Metric Name", "")
        mv = row.get("Metric Value", "")
        if not mv or mv == "n/a":
            continue
        try:
            v = float(mv)
            metrics.setdefault(mn, []).append(v)
        except Exception:
            pass
    return metrics


# Parse all
data = {}
for fname in sorted(ROOT.glob("*.csv")):
    name = fname.stem
    data[name] = parse(str(fname))
    n = max((len(v) for v in data[name].values()), default=0)
    print(f"  loaded {name}: {n} kernel entries", file=sys.stderr)


def med(s, metric):
    if metric in data.get(s, {}) and data[s][metric]:
        return statistics.median(data[s][metric])
    return None


def fmt(v):
    if v is None:
        return "  N/A"
    if abs(v) >= 1e6:
        return f"{v/1e6:>7.2f}M"
    if abs(v) >= 1e3:
        return f"{v/1e3:>7.2f}K"
    return f"{v:>7.2f}"


# Comparison sets
COMPARISONS = [
    {
        "title": "[1] 3g L1: alone vs various AI co-tenants",
        "baseline": "S5_3g_alone",
        "scenarios": ["S5_3g_alone", "S6_3g_qwen", "S7_3g_neuralrx", "S9_3g_3AI_1g",
                      "S13_3g_sat_compute", "S14_3g_sat_hbm", "S24_3g_2sat"],
    },
    {
        "title": "[2] 4g L1: alone proxy + AI co-tenants",
        "baseline": "S5_3g_alone",
        "scenarios": ["S5_3g_alone", "S15_4g_sat_compute", "S18_4g_neuralrx",
                      "S21_4g_2sat", "S26_4g_3sat"],
    },
    {
        "title": "[3] 2g L1: alone vs AI co-tenants",
        "baseline": "S10_2g_alone",
        "scenarios": ["S10_2g_alone", "S12_2g_2AI", "S17_2g_sat_compute", "S22_2g_neuralrx"],
    },
    {
        "title": "[4] Partition size effect (L1 alone)",
        "baseline": "S2_7g_mig",
        "scenarios": ["S2_7g_mig", "S5_3g_alone", "S10_2g_alone"],
    },
    {
        "title": "[5] NeuralRx — the outlier — across L1 partitions",
        "baseline": "S5_3g_alone",
        "scenarios": ["S5_3g_alone", "S7_3g_neuralrx", "S18_4g_neuralrx", "S22_2g_neuralrx"],
    },
]


def print_comparison(cmp):
    print(f"\n{'='*100}")
    print(f"{cmp['title']}")
    print(f"{'='*100}")
    scenarios = [s for s in cmp["scenarios"] if s in data and data[s]]
    base = cmp["baseline"]
    if base not in data or not data[base]:
        print("  Baseline missing.")
        return

    # Print scenarios + descriptions
    for s in scenarios:
        print(f"  {s:<22} = {SCENARIO_DESC.get(s, '?')}")
    print()

    # Header
    col_w = 12
    print(f"  {'Metric':<60}{'':<5}" + "".join(f"{s.split('_')[0]:>{col_w}}" for s in scenarios))
    print("  " + "-" * (60 + 5 + col_w * len(scenarios)))

    for group_name, metrics in METRIC_GROUPS.items():
        print(f"\n  -- {group_name} --")
        for m in metrics:
            row_vals = [med(s, m) for s in scenarios]
            label = m.replace("__", ".")
            if len(label) > 58:
                label = label[:58]
            cells = "".join(f"{fmt(v):>{col_w}}" for v in row_vals)
            print(f"  {label:<60}{'':<5}{cells}")

    # % change vs baseline (most interesting metrics only)
    print(f"\n  -- % change vs {base} (median per kernel) --")
    key_metrics = [
        "lts__t_request_hit_rate.pct",
        "lts__t_bytes.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__cycles_active.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes_read.sum",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
        "gpc__cycles_elapsed.avg",
        "gpu__time_duration.sum",
    ]
    print(f"  {'Metric':<60}{'':<5}" + "".join(f"{s.split('_')[0]:>{col_w}}" for s in scenarios))
    print("  " + "-" * (60 + 5 + col_w * len(scenarios)))
    for m in key_metrics:
        base_v = med(base, m)
        if base_v is None or base_v == 0:
            continue
        row = []
        for s in scenarios:
            v = med(s, m)
            if v is None:
                row.append("  N/A")
            elif s == base:
                row.append("    --")
            else:
                pct = (v - base_v) / base_v * 100
                row.append(f"{pct:+.1f}%")
        label = m.replace("__", ".")
        if len(label) > 58:
            label = label[:58]
        cells = "".join(f"{r:>{col_w}}" for r in row)
        print(f"  {label:<60}{'':<5}{cells}")


for cmp in COMPARISONS:
    print_comparison(cmp)

print(f"\n{'='*100}\nDONE\n{'='*100}")
