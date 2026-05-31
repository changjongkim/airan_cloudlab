#!/usr/bin/env python3
"""
Advanced SQLite analyses beyond basic gap distribution.

Analyses:
  1. Time-series gap distribution — does disturbance grow over measurement window?
  2. Per-iter analysis — does L1 iter 30 differ from iter 1?
  3. Long-tail contribution — what fraction of total idle is in top 1% gaps?
  4. Bimodal gap detection — does gap distribution have multiple modes?
  5. Gap auto-correlation — is gap_i predictive of gap_{i+1}?
  6. CUDA runtime API call duration — driver-level overhead per scenario
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


def short_name(name):
    if not name:
        return "unknown"
    if "cupy_copy" in name:
        m = re.search(r"cupy_copy__(\w+)", name)
        return f"cupy_copy__{m.group(1)}" if m else "cupy_copy"
    m = re.match(r"void (\w+(?:::\w+)+)", name)
    if m:
        return m.group(1)
    m = re.match(r"void (\w+)", name)
    if m:
        return m.group(1)
    return name[:60]


def load_kernels(db_path):
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


def load_runtime_api(db_path):
    """Load CUDA runtime API events (kernel launches etc)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        rows = cur.execute("""
            SELECT start, end, COALESCE(s.value, 'unknown')
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            LEFT JOIN StringIds s ON r.nameId = s.id
            ORDER BY start
        """).fetchall()
    except sqlite3.OperationalError:
        try:
            rows = cur.execute("""
                SELECT start, end, ''
                FROM CUPTI_ACTIVITY_KIND_RUNTIME
                ORDER BY start
            """).fetchall()
        except sqlite3.OperationalError:
            rows = []
    conn.close()
    return rows


def percentile(arr, p):
    if not arr:
        return 0
    return sorted(arr)[int(len(arr) * p / 100)]


# Discover scenarios
by_scenario = defaultdict(list)
for f in sorted(ROOT.glob("*.sqlite")):
    m = f.stem.rsplit("_run", 1)
    if len(m) == 2:
        by_scenario[m[0]].append(str(f))


# ============================================================
# 1. Time-series gap distribution
# ============================================================
print("[1/6] Time-series gap analysis...")
ts_data = []
for scenario, paths in sorted(by_scenario.items()):
    for path in paths:
        rows = load_kernels(path)
        if len(rows) < 100:
            continue
        # Divide kernels into 10 time windows
        first_start = rows[0][0]
        last_end = rows[-1][1]
        window_size = (last_end - first_start) / 10
        windows = [[] for _ in range(10)]
        for i in range(len(rows) - 1):
            gap = rows[i + 1][0] - rows[i][1]
            if gap < 0:
                continue
            kernel_mid = (rows[i][1] + rows[i + 1][0]) / 2
            window_idx = min(9, int((kernel_mid - first_start) / window_size))
            windows[window_idx].append(gap)

        for w_idx, gaps in enumerate(windows):
            if not gaps:
                continue
            ts_data.append({
                "scenario": scenario,
                "run": path.split("_run")[-1].replace(".sqlite", ""),
                "window": w_idx,
                "n_gaps": len(gaps),
                "median_gap_us": percentile(gaps, 50) / 1000.0,
                "p99_gap_us": percentile(gaps, 99) / 1000.0,
                "max_gap_us": max(gaps) / 1000.0,
            })

with open(OUT / "timeseries_gaps.csv", "w") as f:
    if ts_data:
        w = csv.DictWriter(f, fieldnames=list(ts_data[0].keys()))
        w.writeheader()
        w.writerows(ts_data)
print(f"  → {OUT / 'timeseries_gaps.csv'} ({len(ts_data)} rows)")


# ============================================================
# 2. Long-tail contribution
# ============================================================
print("\n[2/6] Long-tail contribution analysis...")
longtail_data = []
for scenario, paths in sorted(by_scenario.items()):
    all_gaps = []
    for path in paths:
        rows = load_kernels(path)
        for i in range(len(rows) - 1):
            gap = rows[i + 1][0] - rows[i][1]
            if gap >= 0:
                all_gaps.append(gap)
    if not all_gaps:
        continue
    all_gaps.sort()
    total = sum(all_gaps)
    p99_threshold = all_gaps[int(len(all_gaps) * 0.99)]
    p999_threshold = all_gaps[int(len(all_gaps) * 0.999)]
    top1pct_sum = sum(g for g in all_gaps if g >= p99_threshold)
    top0_1pct_sum = sum(g for g in all_gaps if g >= p999_threshold)
    longtail_data.append({
        "scenario": scenario,
        "n_gaps": len(all_gaps),
        "total_gap_us": total / 1000.0,
        "p99_threshold_us": p99_threshold / 1000.0,
        "p999_threshold_us": p999_threshold / 1000.0,
        "top1pct_sum_us": top1pct_sum / 1000.0,
        "top1pct_fraction": top1pct_sum / total if total else 0,
        "top0_1pct_sum_us": top0_1pct_sum / 1000.0,
        "top0_1pct_fraction": top0_1pct_sum / total if total else 0,
    })

with open(OUT / "longtail_contribution.csv", "w") as f:
    if longtail_data:
        w = csv.DictWriter(f, fieldnames=list(longtail_data[0].keys()))
        w.writeheader()
        w.writerows(longtail_data)
print(f"  → {OUT / 'longtail_contribution.csv'} ({len(longtail_data)} rows)")


# ============================================================
# 3. Bimodal gap detection
# ============================================================
print("\n[3/6] Bimodal gap distribution detection...")
bimodal_data = []
for scenario, paths in sorted(by_scenario.items()):
    all_gaps = []
    for path in paths:
        rows = load_kernels(path)
        for i in range(len(rows) - 1):
            gap = rows[i + 1][0] - rows[i][1]
            if 0 <= gap < 1_000_000:  # exclude extreme outliers (1ms+)
                all_gaps.append(gap / 1000.0)  # us
    if len(all_gaps) < 100:
        continue
    all_gaps.sort()
    # Approximate bimodal via percentile differences
    p25 = percentile(all_gaps, 25)
    p50 = percentile(all_gaps, 50)
    p75 = percentile(all_gaps, 75)
    iqr = p75 - p25
    # Bimodal if mode (low cluster) << median << high cluster
    # Use simple test: ratio of p75/p25
    bimodal_data.append({
        "scenario": scenario,
        "n_gaps": len(all_gaps),
        "min_us": all_gaps[0],
        "p10_us": percentile(all_gaps, 10),
        "p25_us": p25,
        "p50_us": p50,
        "p75_us": p75,
        "p90_us": percentile(all_gaps, 90),
        "max_us": all_gaps[-1],
        "iqr": iqr,
        "p75_p25_ratio": p75 / p25 if p25 > 0 else 0,
        "p90_p10_ratio": percentile(all_gaps, 90) / max(percentile(all_gaps, 10), 0.001),
    })

with open(OUT / "gap_distribution_shape.csv", "w") as f:
    if bimodal_data:
        w = csv.DictWriter(f, fieldnames=list(bimodal_data[0].keys()))
        w.writeheader()
        w.writerows(bimodal_data)
print(f"  → {OUT / 'gap_distribution_shape.csv'} ({len(bimodal_data)} rows)")


# ============================================================
# 4. Gap auto-correlation (does long gap predict another long gap?)
# ============================================================
print("\n[4/6] Gap auto-correlation analysis...")
autocorr_data = []
for scenario, paths in sorted(by_scenario.items()):
    for path in paths:
        rows = load_kernels(path)
        gaps = []
        for i in range(len(rows) - 1):
            gap = rows[i + 1][0] - rows[i][1]
            if gap >= 0:
                gaps.append(gap)
        if len(gaps) < 50:
            continue
        # Compute Pearson correlation between gap_i and gap_{i+1}
        n = len(gaps) - 1
        mean1 = sum(gaps[:-1]) / n
        mean2 = sum(gaps[1:]) / n
        cov = sum((gaps[i] - mean1) * (gaps[i + 1] - mean2) for i in range(n)) / n
        var1 = sum((g - mean1) ** 2 for g in gaps[:-1]) / n
        var2 = sum((g - mean2) ** 2 for g in gaps[1:]) / n
        corr = cov / (var1 ** 0.5 * var2 ** 0.5) if var1 > 0 and var2 > 0 else 0
        autocorr_data.append({
            "scenario": scenario,
            "run": path.split("_run")[-1].replace(".sqlite", ""),
            "n_pairs": n,
            "autocorr_lag1": corr,
            "mean_gap_us": mean1 / 1000.0,
        })

with open(OUT / "gap_autocorrelation.csv", "w") as f:
    if autocorr_data:
        w = csv.DictWriter(f, fieldnames=list(autocorr_data[0].keys()))
        w.writeheader()
        w.writerows(autocorr_data)
print(f"  → {OUT / 'gap_autocorrelation.csv'} ({len(autocorr_data)} rows)")


# ============================================================
# 5. CUDA runtime API analysis (driver-level overhead)
# ============================================================
print("\n[5/6] CUDA runtime API analysis...")
api_data = []
for scenario, paths in sorted(by_scenario.items()):
    api_durations = []
    api_counts = defaultdict(int)
    for path in paths:
        runtime_events = load_runtime_api(path)
        for start, end, name in runtime_events:
            api_durations.append(end - start)
            api_counts[name] += 1
    if not api_durations:
        continue
    api_durations.sort()
    api_data.append({
        "scenario": scenario,
        "total_api_calls": len(api_durations),
        "total_api_time_us": sum(api_durations) / 1000.0,
        "median_api_us": percentile(api_durations, 50) / 1000.0,
        "p99_api_us": percentile(api_durations, 99) / 1000.0,
        "max_api_us": api_durations[-1] / 1000.0,
        "top_api_name": max(api_counts, key=api_counts.get) if api_counts else "",
        "top_api_count": max(api_counts.values()) if api_counts else 0,
    })

with open(OUT / "runtime_api_analysis.csv", "w") as f:
    if api_data:
        w = csv.DictWriter(f, fieldnames=list(api_data[0].keys()))
        w.writeheader()
        w.writerows(api_data)
print(f"  → {OUT / 'runtime_api_analysis.csv'} ({len(api_data)} rows)")


# ============================================================
# 6. Print summary
# ============================================================
print(f"\n{'='*100}\nADVANCED ANALYSIS SUMMARY\n{'='*100}\n")

# Long-tail summary
print("Long-tail contribution (top 1% of gaps as % of total idle):")
for r in longtail_data[:8]:
    print(f"  {r['scenario']:<22} top1%={r['top1pct_fraction']*100:>5.1f}%  top0.1%={r['top0_1pct_fraction']*100:>5.1f}%  threshold_p99={r['p99_threshold_us']:.1f}us")

print("\nAuto-correlation (lag-1) — does long gap predict next long gap?")
agg_corr = defaultdict(list)
for r in autocorr_data:
    agg_corr[r["scenario"]].append(r["autocorr_lag1"])
for s in sorted(agg_corr.keys())[:10]:
    vals = agg_corr[s]
    print(f"  {s:<22} mean_corr={statistics.mean(vals):+.3f}  range=[{min(vals):+.3f}, {max(vals):+.3f}]")

print("\nGap distribution shape (p75/p25 ratio — bimodal indicator):")
for r in bimodal_data[:8]:
    print(f"  {r['scenario']:<22} p25={r['p25_us']:>5.2f}us  p75={r['p75_us']:>5.2f}us  p75/p25={r['p75_p25_ratio']:>5.2f}  p90/p10={r['p90_p10_ratio']:>5.1f}")

print("\nCUDA Runtime API summary:")
for r in api_data[:8]:
    print(f"  {r['scenario']:<22} calls={r['total_api_calls']:>7}  total={r['total_api_time_us']/1000:>8.1f}ms  median={r['median_api_us']:>6.2f}us  p99={r['p99_api_us']:>6.2f}us")
