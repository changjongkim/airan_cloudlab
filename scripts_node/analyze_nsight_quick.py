"""Quick nsight scenario comparison — extract median per metric, compare vs baseline."""
import csv
import statistics
import sys
import os

KEY_METRICS = [
    ("L2 hit rate (%)", "lts__t_request_hit_rate.pct"),
    ("L2 throughput (%)", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("L2 traffic (KB)", "lts__t_bytes.sum"),
    ("DRAM avg throughput (%)", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("DRAM max throughput (%)", "dram__throughput.max.pct_of_peak_sustained_elapsed"),
    ("DRAM cycles active (%)", "dram__cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("DRAM bytes read (KB)", "dram__bytes_read.sum"),
    ("DRAM bytes write (KB)", "dram__bytes_write.sum"),
    ("SM throughput (%)", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("Warps active (%)", "smsp__warps_active.avg.pct_of_peak_sustained_elapsed"),
    ("GPU time/kernel (us)", "gpu__time_duration.sum"),
]

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

scenarios = sys.argv[1:] if len(sys.argv) > 1 else ["S5_3g_alone", "S6_3g_qwen", "S7_3g_neuralrx", "S9_3g_3AI_1g"]
data = {s: parse(f"/tmp/{s}.csv") for s in scenarios}

print()
print(f"{'Metric':<28} | " + " | ".join(f"{s[:14]:>14}" for s in scenarios))
print("-" * 28 + "-+-" + " + ".join("-" * 14 for _ in scenarios))
for label, metric in KEY_METRICS:
    row_vals = []
    for s in scenarios:
        if metric in data[s] and data[s][metric]:
            row_vals.append(f"{statistics.median(data[s][metric]):>14.2f}")
        else:
            row_vals.append(f"{'N/A':>14}")
    print(f"{label:<28} | " + " | ".join(row_vals))

if len(scenarios) >= 2:
    base_scenario = scenarios[0]
    print()
    print(f"\n=== % Change vs {base_scenario} baseline ===")
    others = scenarios[1:]
    print(f"{'Metric':<28} | " + " | ".join(f"{s[:14]:>14}" for s in others))
    print("-" * 28 + "-+-" + " + ".join("-" * 14 for _ in others))
    for label, metric in KEY_METRICS:
        if metric not in data[base_scenario] or not data[base_scenario][metric]:
            continue
        base = statistics.median(data[base_scenario][metric])
        if base == 0:
            continue
        row_vals = []
        for s in others:
            if metric in data[s] and data[s][metric]:
                v = statistics.median(data[s][metric])
                delta = (v - base) / base * 100
                row_vals.append(f"{delta:>+13.1f}%")
            else:
                row_vals.append(f"{'N/A':>14}")
        print(f"{label:<28} | " + " | ".join(row_vals))
