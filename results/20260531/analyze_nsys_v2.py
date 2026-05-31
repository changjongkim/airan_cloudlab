#!/usr/bin/env python3
"""
Analyze nsys v2 (Tier1-matched: CELLS=20 ITERS=30, N=3) kernel timing.

For each scenario × run:
  - Number of cuPHY L1 kernels (excluding memcpy/memset)
  - Median kernel duration
  - Kernel gap distribution (median, p95, p99, max)
  - Total wall-clock time
  - Cumulative kernel time
  - Idle time and fraction

Aggregate over 3 runs per scenario, compare alone vs with-AI.
"""
import csv
import statistics
import sys
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent / "nsys_csv_v2"

SCENARIO_DESC = {
    "S2_7g_mig": "7g MIG single (98 SMs, L1 alone)",
    "S5_3g_alone": "3g L1 alone (split-60-40)",
    "S6_3g_qwen": "3g L1 + Qwen-7B (on 2g)",
    "S7_3g_neuralrx": "3g L1 + NeuralRx (on 2g)",
    "S9_3g_3AI_1g": "3g L1 + 3 small AI (on 1g×3)",
    "S10_2g_alone": "2g L1 alone",
    "S12_2g_2AI": "2g L1 + 2 AI (3g+2g)",
    "S13_3g_sat_compute": "3g L1 + sat_compute (on 2g)",
    "S14_3g_sat_hbm": "3g L1 + sat_hbm (on 2g)",
    "S15_4g_sat_compute": "4g L1 + sat_compute (on 2g)",
    "S17_2g_sat_compute": "2g L1 + sat_compute (on 3g)",
    "S18_4g_neuralrx": "4g L1 + NeuralRx (on 2g)",
    "S21_4g_2sat": "4g L1 + 2 sat (2g+1g)",
    "S22_2g_neuralrx": "2g L1 + NeuralRx (on 3g)",
    "S24_3g_2sat": "3g L1 + 2 sat (2g+2g)",
    "S26_4g_3sat": "4g L1 + 3 sat (1g×3)",
}


def analyze_trace(path):
    """Parse one cuda_gpu_trace CSV; return kernel timing stats."""
    kernels = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "").strip()
            if not name or "memcpy" in name.lower() or "memset" in name.lower():
                continue
            try:
                start = int(row["Start (ns)"])
                dur = int(row["Duration (ns)"])
            except (KeyError, ValueError):
                continue
            kernels.append((start, start + dur, name))
    if not kernels:
        return None
    kernels.sort(key=lambda x: x[0])
    gaps = []
    for i in range(1, len(kernels)):
        gap = kernels[i][0] - kernels[i - 1][1]
        if gap >= 0:
            gaps.append(gap)
    durations = [k[1] - k[0] for k in kernels]
    total_time = kernels[-1][1] - kernels[0][0]
    cumulative_kernel = sum(durations)
    idle = max(0, total_time - cumulative_kernel)
    gaps.sort()
    durations.sort()

    def pct(arr, p):
        return arr[int(len(arr) * p / 100)] if arr else 0

    return {
        "n_kernels": len(kernels),
        "median_dur_us": pct(durations, 50) / 1000.0,
        "p99_dur_us": pct(durations, 99) / 1000.0,
        "mean_gap_us": (statistics.mean(gaps) if gaps else 0) / 1000.0,
        "median_gap_us": pct(gaps, 50) / 1000.0,
        "p95_gap_us": pct(gaps, 95) / 1000.0,
        "p99_gap_us": pct(gaps, 99) / 1000.0,
        "max_gap_us": (gaps[-1] if gaps else 0) / 1000.0,
        "total_time_ms": total_time / 1e6,
        "cumulative_kernel_ms": cumulative_kernel / 1e6,
        "idle_ms": idle / 1e6,
        "idle_fraction": idle / total_time if total_time else 0,
    }


# Group runs by scenario
by_scenario = defaultdict(list)
for f in sorted(ROOT.glob("*_cuda_gpu_trace.csv")):
    m = re.match(r"(.+?)_run(\d+)_cuda_gpu_trace\.csv", f.name)
    if not m:
        continue
    scenario, run = m.group(1), int(m.group(2))
    by_scenario[scenario].append((run, str(f)))

# Aggregate
results = {}
for scenario, runs in sorted(by_scenario.items()):
    run_stats = []
    for run, path in sorted(runs):
        r = analyze_trace(path)
        if r:
            run_stats.append(r)
    if not run_stats:
        continue
    # Average across runs
    avg = {}
    for key in run_stats[0]:
        vals = [r[key] for r in run_stats]
        avg[key] = statistics.mean(vals)
        avg[key + "_stdev"] = statistics.stdev(vals) if len(vals) >= 2 else 0
    avg["n_runs"] = len(run_stats)
    results[scenario] = avg

# Print main table
print(f"\n{'='*120}")
print("NSYS v2 (Tier1-matched CELLS=20 ITERS=30, N=3 runs each) — Kernel Gap Analysis")
print(f"{'='*120}\n")

print(f"{'Scenario':<22}{'N':>4}{'Kernels':>10}{'MedDur(us)':>12}{'MedGap(us)':>12}{'p99Gap(us)':>12}{'MaxGap(us)':>12}{'Total(ms)':>11}{'Idle(ms)':>11}{'Idle%':>8}")
print("-" * 120)
for s, r in results.items():
    desc = s[:22]
    print(f"{desc:<22}{r['n_runs']:>4}{r['n_kernels']:>10.0f}{r['median_dur_us']:>12.2f}{r['median_gap_us']:>12.2f}{r['p99_gap_us']:>12.2f}{r['max_gap_us']:>12.1f}{r['total_time_ms']:>11.1f}{r['idle_ms']:>11.1f}{r['idle_fraction']*100:>7.1f}%")


def pct_change(v, base):
    return (v - base) / base * 100 if base else 0


# Comparisons
print(f"\n{'='*120}")
print("COMPARISONS vs respective alone baseline (% change in median over 3 runs)")
print(f"{'='*120}")

groups = [
    ("3g L1 alone vs +AI", "S5_3g_alone",
     ["S5_3g_alone", "S6_3g_qwen", "S7_3g_neuralrx", "S9_3g_3AI_1g",
      "S13_3g_sat_compute", "S14_3g_sat_hbm", "S24_3g_2sat"]),
    ("4g L1 alone vs +AI (using S5 baseline)", "S5_3g_alone",
     ["S5_3g_alone", "S15_4g_sat_compute", "S18_4g_neuralrx", "S21_4g_2sat", "S26_4g_3sat"]),
    ("2g L1 alone vs +AI", "S10_2g_alone",
     ["S10_2g_alone", "S12_2g_2AI", "S17_2g_sat_compute", "S22_2g_neuralrx"]),
    ("Partition size effect (alone)", "S2_7g_mig",
     ["S2_7g_mig", "S5_3g_alone", "S10_2g_alone"]),
]
metrics_to_show = [
    ("MedDur", "median_dur_us"),
    ("MedGap", "median_gap_us"),
    ("p95Gap", "p95_gap_us"),
    ("p99Gap", "p99_gap_us"),
    ("MaxGap", "max_gap_us"),
    ("Total", "total_time_ms"),
    ("Idle", "idle_ms"),
    ("Idle%", "idle_fraction"),
]
for title, base, scenarios in groups:
    print(f"\n--- {title} ---")
    if base not in results:
        continue
    b = results[base]
    print(f"  {'Scenario':<22}" + "".join(f"{label:>10}" for label, _ in metrics_to_show))
    print("  " + "-" * (22 + len(metrics_to_show) * 10))
    for s in scenarios:
        if s not in results:
            continue
        r = results[s]
        cells = []
        for label, key in metrics_to_show:
            if s == base:
                cells.append(f"{'--':>10}")
            else:
                if key == "idle_fraction":
                    cells.append(f"{pct_change(r[key]*100, b[key]*100):>+9.1f}%")
                else:
                    cells.append(f"{pct_change(r[key], b[key]):>+9.1f}%")
        print(f"  {s:<22}" + "".join(cells))

print(f"\n{'='*120}\nDONE\n{'='*120}")
