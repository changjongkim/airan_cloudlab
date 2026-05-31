#!/usr/bin/env python3
"""
Comprehensive nsys SQLite analysis — every angle.

Outputs:
  - all_kernel_summary.csv — per-kernel-type stats per scenario
  - kernel_pair_transitions.csv — A→B transition gap analysis
  - timeseries_gaps.csv — gap distribution over time windows
  - memory_ops_analysis.csv — memcpy/memset analysis
"""
import sqlite3
import statistics
import csv
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent / "nsys_sqlite_v2"
OUT = Path(__file__).parent / "nsys_sqlite_v2_analysis"
OUT.mkdir(exist_ok=True)

CUPHY_KEYWORDS = [
    "channel_eq", "ch_est", "noise_intf", "ldpc", "pusch", "crc",
    "windowed", "eqMmse", "noiseIntf", "ldpcDeRate", "ldpcDecode", "crcCheck",
    "scrambling", "descrambling", "demap", "softDemap", "chEstFilter",
]

SCENARIO_DESC = {
    "S2_7g_mig": "7g MIG alone",
    "S5_3g_alone": "3g alone (split-60-40)",
    "S6_3g_qwen": "3g + Qwen on 2g",
    "S7_3g_neuralrx": "3g + NeuralRx on 2g",
    "S9_3g_3AI_1g": "3g + 3 AI on 1g×3",
    "S10_2g_alone": "2g alone",
    "S12_2g_2AI": "2g + 2 AI",
    "S13_3g_sat_compute": "3g + sat_compute",
    "S14_3g_sat_hbm": "3g + sat_hbm",
    "S15_4g_sat_compute": "4g + sat_compute",
    "S17_2g_sat_compute": "2g + sat_compute",
    "S18_4g_neuralrx": "4g + NeuralRx",
    "S21_4g_2sat": "4g + 2 sat",
    "S22_2g_neuralrx": "2g + NeuralRx",
    "S24_3g_2sat": "3g + 2 sat",
    "S26_4g_3sat": "4g + 3 sat",
}


def is_cuphy(name):
    if not name:
        return False
    low = name.lower()
    return any(k.lower() in low for k in CUPHY_KEYWORDS)


def short_name(name):
    """Extract kernel identifier — namespace::kernelName, stop before template/args."""
    if not name:
        return "unknown"
    # cupy_copy patterns
    if "cupy_copy" in name:
        # extract just the type signature
        m = re.search(r"cupy_copy__(\w+)", name)
        return f"cupy_copy__{m.group(1)}" if m else "cupy_copy"
    # "void <ns::name><...>" or "void <name>(...)"
    m = re.match(r"void (\w+(?:::\w+)+)", name)
    if m:
        return m.group(1)
    m = re.match(r"void (\w+)", name)
    if m:
        return m.group(1)
    # Already short
    return name[:60]


def load_kernels(db_path):
    """Load all kernels from sqlite."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT k.start, k.end, COALESCE(s_d.value, s_s.value, 'unknown')
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds s_d ON k.demangledName = s_d.id
        LEFT JOIN StringIds s_s ON k.shortName = s_s.id
        ORDER BY k.start
    """).fetchall()
    conn.close()
    return [(start, end, short_name(name)) for start, end, name in rows]


def load_memops(db_path):
    """Load memcpy/memset operations."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        memcpy = cur.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"
        ).fetchall()
    except sqlite3.OperationalError:
        memcpy = []
    try:
        memset = cur.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET ORDER BY start"
        ).fetchall()
    except sqlite3.OperationalError:
        memset = []
    conn.close()
    return memcpy, memset


def percentile(arr, p):
    if not arr:
        return 0
    return sorted(arr)[int(len(arr) * p / 100)]


# Group sqlite files by scenario
by_scenario = defaultdict(list)
for f in sorted(ROOT.glob("*.sqlite")):
    m = f.stem.rsplit("_run", 1)
    if len(m) == 2:
        by_scenario[m[0]].append(str(f))

print(f"Found {sum(len(v) for v in by_scenario.values())} sqlite files in {len(by_scenario)} scenarios")


# ============================================================
# 1. Per-kernel-type summary across all scenarios
# ============================================================
print("\n[1/4] Per-kernel-type analysis...")
kernel_data = {}  # scenario → {kernel_name: {count, durations, post_gaps}}
for scenario, paths in sorted(by_scenario.items()):
    merged = defaultdict(lambda: {"durations": [], "post_gaps": []})
    for path in paths:
        rows = load_kernels(path)
        for i, (s, e, name) in enumerate(rows):
            merged[name]["durations"].append(e - s)
            if i + 1 < len(rows):
                gap = rows[i + 1][0] - e
                if gap >= 0:
                    merged[name]["post_gaps"].append(gap)
    summary = {}
    for name, data in merged.items():
        if not data["post_gaps"]:
            continue
        summary[name] = {
            "count": len(data["durations"]),
            "total_dur_ns": sum(data["durations"]),
            "median_dur_ns": percentile(data["durations"], 50),
            "median_gap_ns": percentile(data["post_gaps"], 50),
            "p95_gap_ns": percentile(data["post_gaps"], 95),
            "p99_gap_ns": percentile(data["post_gaps"], 99),
            "max_gap_ns": max(data["post_gaps"]),
        }
    kernel_data[scenario] = summary

# Write per-kernel summary CSV
with open(OUT / "all_kernel_summary.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["scenario", "kernel_name", "count", "total_dur_us", "median_dur_us",
                "median_post_gap_us", "p95_post_gap_us", "p99_post_gap_us", "max_post_gap_us"])
    for scenario, kernels in kernel_data.items():
        for name, s in kernels.items():
            w.writerow([scenario, name, s["count"],
                        s["total_dur_ns"] / 1000.0,
                        s["median_dur_ns"] / 1000.0,
                        s["median_gap_ns"] / 1000.0,
                        s["p95_gap_ns"] / 1000.0,
                        s["p99_gap_ns"] / 1000.0,
                        s["max_gap_ns"] / 1000.0])
print(f"  → {OUT / 'all_kernel_summary.csv'}")


# ============================================================
# 2. Per-kernel inflation matrix (% change vs baseline)
# ============================================================
print("\n[2/4] Inflation matrix (vs S5 baseline)...")
baseline = "S5_3g_alone"
if baseline in kernel_data:
    base = kernel_data[baseline]
    # Top kernels in baseline by total time
    top_kernels = sorted(base.items(), key=lambda kv: kv[1]["total_dur_ns"], reverse=True)[:20]
    top_names = [name for name, _ in top_kernels]

    rows_out = []
    for name in top_names:
        if name not in base:
            continue
        b = base[name]
        row = {
            "kernel": name,
            "baseline_p99_gap_us": b["p99_gap_ns"] / 1000.0,
            "baseline_total_dur_us": b["total_dur_ns"] / 1000.0,
            "count": b["count"],
        }
        for scenario in sorted(kernel_data.keys()):
            if scenario == baseline:
                continue
            if name not in kernel_data[scenario]:
                row[scenario] = None
                continue
            scen_p99 = kernel_data[scenario][name]["p99_gap_ns"]
            base_p99 = b["p99_gap_ns"]
            row[scenario] = (scen_p99 - base_p99) / base_p99 * 100 if base_p99 else 0
        rows_out.append(row)

    with open(OUT / "kernel_inflation_vs_S5.csv", "w") as f:
        if rows_out:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
    print(f"  → {OUT / 'kernel_inflation_vs_S5.csv'}")


# ============================================================
# 3. Kernel-pair transition analysis (which A→B has biggest gap inflation)
# ============================================================
print("\n[3/4] Kernel-pair transition analysis...")
pair_data = {}  # scenario → {(A,B): post_gaps}
for scenario, paths in sorted(by_scenario.items()):
    pairs = defaultdict(list)
    for path in paths:
        rows = load_kernels(path)
        for i in range(len(rows) - 1):
            a = rows[i]
            b = rows[i + 1]
            gap = b[0] - a[1]
            if gap >= 0:
                pairs[(a[2], b[2])].append(gap)
    pair_data[scenario] = pairs

# Top pairs by frequency
if baseline in pair_data:
    pair_counts = sorted(pair_data[baseline].items(), key=lambda kv: -len(kv[1]))[:20]
    with open(OUT / "kernel_pair_transitions.csv", "w") as f:
        w = csv.writer(f)
        header = ["from_kernel", "to_kernel", "count_in_baseline"]
        scenarios_sorted = sorted(kernel_data.keys())
        for s in scenarios_sorted:
            header.append(f"{s}_p99_us")
        w.writerow(header)
        for (a, b), gaps in pair_counts:
            row = [a, b, len(gaps)]
            for s in scenarios_sorted:
                if s in pair_data and (a, b) in pair_data[s]:
                    row.append(percentile(pair_data[s][(a, b)], 99) / 1000.0)
                else:
                    row.append("")
            w.writerow(row)
    print(f"  → {OUT / 'kernel_pair_transitions.csv'}")


# ============================================================
# 4. Memory operations analysis
# ============================================================
print("\n[4/4] Memory operations (memcpy/memset)...")
mem_data = []
for scenario, paths in sorted(by_scenario.items()):
    memcpy_durs, memset_durs = [], []
    memcpy_total, memset_total = 0, 0
    for path in paths:
        memcpy, memset = load_memops(path)
        for s, e in memcpy:
            d = e - s
            memcpy_durs.append(d)
            memcpy_total += d
        for s, e in memset:
            d = e - s
            memset_durs.append(d)
            memset_total += d
    if memcpy_durs or memset_durs:
        mem_data.append({
            "scenario": scenario,
            "memcpy_count": len(memcpy_durs),
            "memcpy_total_us": memcpy_total / 1000.0,
            "memcpy_median_us": percentile(memcpy_durs, 50) / 1000.0 if memcpy_durs else 0,
            "memcpy_p99_us": percentile(memcpy_durs, 99) / 1000.0 if memcpy_durs else 0,
            "memset_count": len(memset_durs),
            "memset_total_us": memset_total / 1000.0,
            "memset_median_us": percentile(memset_durs, 50) / 1000.0 if memset_durs else 0,
            "memset_p99_us": percentile(memset_durs, 99) / 1000.0 if memset_durs else 0,
        })

with open(OUT / "memory_ops_analysis.csv", "w") as f:
    if mem_data:
        w = csv.DictWriter(f, fieldnames=list(mem_data[0].keys()))
        w.writeheader()
        w.writerows(mem_data)
print(f"  → {OUT / 'memory_ops_analysis.csv'}")


# ============================================================
# 5. Print headline summary
# ============================================================
print(f"\n{'=' * 100}\nHEADLINE SUMMARY\n{'=' * 100}")

base = kernel_data.get(baseline, {})
top = sorted(base.items(), key=lambda kv: kv[1]["total_dur_ns"], reverse=True)[:5]
print(f"\nTop cuPHY kernels in {baseline}:")
for name, s in top:
    print(f"  {name:<55} count={s['count']:>5}  median_gap={s['median_gap_ns']/1000:>8.2f}us  p99_gap={s['p99_gap_ns']/1000:>8.2f}us")

print(f"\nKernel post-gap p99 inflation vs {baseline}:")
print(f"  {'Kernel':<45}" + "".join(f"{s.split('_')[0]:>10}" for s in sorted(kernel_data.keys()) if s != baseline))
for name, _ in top[:5]:
    cells = []
    for s in sorted(kernel_data.keys()):
        if s == baseline:
            continue
        if name in kernel_data[s] and name in base:
            scen = kernel_data[s][name]["p99_gap_ns"]
            b = base[name]["p99_gap_ns"]
            d = (scen - b) / b * 100 if b else 0
            cells.append(f"{d:+9.1f}%")
        else:
            cells.append(f"{'N/A':>10}")
    print(f"  {name[:45]:<45}" + "".join(cells))

print(f"\nCSV outputs: {OUT}/")
