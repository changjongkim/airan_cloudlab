#!/usr/bin/env python3
"""
Analyze nsys kernel timeline traces for inter-kernel gap distribution.

Hypothesis: AI co-tenant causes longer gaps BETWEEN L1 kernels (not within kernel).
If gaps are larger with AI, confirms the 'inter-kernel scheduling contention' theory.

For each scenario, computes:
  - Number of L1 cuPHY kernels (excluding memcpy / setup)
  - Median, p95, p99, max gap (ns) between consecutive kernel completions
  - Total L1 measurement time (last_end - first_start)
  - Cumulative kernel time (sum of durations)
  - Idle time = total_time - cumulative_kernel_time
  - Idle fraction = idle_time / total_time
"""
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).parent / "nsys_csv"

SCENARIO_DESC = {
    "S2_7g_mig": "7g MIG single (L1 alone, 98 SMs)",
    "S5_3g_alone": "3g L1 alone (split-60-40)",
    "S6_3g_qwen": "3g L1 + Qwen on 2g",
    "S7_3g_neuralrx": "3g L1 + NeuralRx on 2g",
    "S9_3g_3AI_1g": "3g L1 + 3 small AI on 1g×3",
    "S10_2g_alone": "2g L1 alone (split-40-60)",
    "S12_2g_2AI": "2g L1 + 2 AI on 3g+2g",
    "S13_3g_sat_compute": "3g L1 + sat_compute on 2g",
    "S14_3g_sat_hbm": "3g L1 + sat_hbm on 2g",
    "S15_4g_sat_compute": "4g L1 + sat_compute on 2g",
    "S17_2g_sat_compute": "2g L1 + sat_compute on 3g",
    "S18_4g_neuralrx": "4g L1 + NeuralRx on 2g",
    "S21_4g_2sat": "4g L1 + 2 sat (2g+1g)",
    "S22_2g_neuralrx": "2g L1 + NeuralRx on 3g",
    "S24_3g_2sat": "3g L1 + 2 sat on 2g+2g",
    "S26_4g_3sat": "4g L1 + 3 sat on 1g×3 (worst)",
}


def analyze_trace(path):
    """Return dict of stats for one nsys trace CSV."""
    kernels = []  # (start_ns, end_ns, name)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "").strip()
            if not name:
                continue
            # Skip memory operations — focus on actual kernels
            if "memcpy" in name.lower() or "memset" in name.lower():
                continue
            try:
                start = int(row["Start (ns)"])
                dur = int(row["Duration (ns)"])
            except (KeyError, ValueError):
                continue
            kernels.append((start, start + dur, name))
    if not kernels:
        return None
    # Sort by start time
    kernels.sort(key=lambda x: x[0])
    # Compute gaps
    gaps = []
    for i in range(1, len(kernels)):
        gap = kernels[i][0] - kernels[i - 1][1]
        if gap >= 0:  # non-overlapping
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
        "median_gap_us": pct(gaps, 50) / 1000.0,
        "p95_gap_us": pct(gaps, 95) / 1000.0,
        "p99_gap_us": pct(gaps, 99) / 1000.0,
        "max_gap_us": (gaps[-1] if gaps else 0) / 1000.0,
        "mean_gap_us": (statistics.mean(gaps) if gaps else 0) / 1000.0,
        "total_time_ms": total_time / 1e6,
        "cumulative_kernel_ms": cumulative_kernel / 1e6,
        "idle_ms": idle / 1e6,
        "idle_fraction": idle / total_time if total_time else 0,
    }


# Discover scenarios
results = {}
for f in sorted(ROOT.glob("*_cuda_gpu_trace.csv")):
    name = f.name.replace("_cuda_gpu_trace.csv", "")
    print(f"  parsing {name}...")
    r = analyze_trace(str(f))
    if r:
        results[name] = r

print(f"\n{'='*100}")
print(f"NSYS Kernel Timeline Analysis — Gap distribution per scenario")
print(f"{'='*100}\n")

# Print table
col_names = [
    ("Scenario", "name"),
    ("Kernels", "n_kernels"),
    ("Med dur (us)", "median_dur_us"),
    ("Med gap (us)", "median_gap_us"),
    ("p95 gap (us)", "p95_gap_us"),
    ("p99 gap (us)", "p99_gap_us"),
    ("Max gap (us)", "max_gap_us"),
    ("Total (ms)", "total_time_ms"),
    ("Idle (ms)", "idle_ms"),
    ("Idle %", "idle_fraction"),
]
header = " | ".join(f"{cn:>14}" for cn, _ in col_names)
print(f"{'':<22}{header}")
print("-" * (22 + len(header)))
for name, r in results.items():
    desc = SCENARIO_DESC.get(name, "?")[:22]
    row_cells = []
    for cn, key in col_names:
        if key == "name":
            continue
        v = r[key]
        if key in ("median_dur_us", "median_gap_us", "p95_gap_us", "p99_gap_us", "max_gap_us"):
            row_cells.append(f"{v:>14.2f}")
        elif key == "n_kernels":
            row_cells.append(f"{v:>14d}")
        elif key in ("total_time_ms", "idle_ms"):
            row_cells.append(f"{v:>14.1f}")
        elif key == "idle_fraction":
            row_cells.append(f"{v*100:>13.1f}%")
    print(f"{desc:<22}" + " | ".join(row_cells))

# Comparisons vs baseline
print(f"\n{'='*100}")
print("COMPARISONS vs respective alone baseline (% change)")
print(f"{'='*100}\n")


def pct_change(v, base):
    if base == 0:
        return float('inf')
    return (v - base) / base * 100


for base_label, base_name, related in [
    ("3g L1 alone (S5)", "S5_3g_alone", ["S5_3g_alone", "S6_3g_qwen", "S7_3g_neuralrx",
                                          "S9_3g_3AI_1g", "S13_3g_sat_compute",
                                          "S14_3g_sat_hbm", "S24_3g_2sat"]),
    ("2g L1 alone (S10)", "S10_2g_alone", ["S10_2g_alone", "S17_2g_sat_compute",
                                            "S22_2g_neuralrx"]),
    ("4g + AI vs 3g alone", "S5_3g_alone", ["S5_3g_alone", "S15_4g_sat_compute",
                                             "S18_4g_neuralrx", "S21_4g_2sat",
                                             "S26_4g_3sat"]),
]:
    print(f"\n--- {base_label} ---")
    if base_name not in results:
        continue
    b = results[base_name]
    keys = ["median_dur_us", "median_gap_us", "p99_gap_us", "max_gap_us",
            "total_time_ms", "idle_ms", "idle_fraction"]
    print(f"  {'Scenario':<22} | " + " | ".join(f"{k:>15}" for k in keys))
    print("  " + "-" * (22 + 3 + len(keys) * 18))
    for s in related:
        if s not in results:
            continue
        r = results[s]
        cells = []
        for k in keys:
            if s == base_name:
                cells.append(f"{'-':>15}")
            else:
                if k == "idle_fraction":
                    cells.append(f"{pct_change(r[k]*100, b[k]*100):>+14.1f}%")
                else:
                    cells.append(f"{pct_change(r[k], b[k]):>+14.1f}%")
        print(f"  {s:<22} | " + " | ".join(cells))

print(f"\n{'='*100}\nDONE\n{'='*100}")
