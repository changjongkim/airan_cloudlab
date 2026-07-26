#!/usr/bin/env python3
"""Extract kernel timing from ALL 360 Chain 17 sqlite files directly.

nsys stores CUPTI_ACTIVITY_KIND_KERNEL with (start, end, streamId, demangledName).
This lets us extract per-kernel data without needing nsys binary on Mac.

Coverage: Config A/B/C × N ∈ {1,2,3,4,6,8} × MPS off/on × 3 trials = 360 files.
Previously only 13 (Config A t1 only) were extracted.
"""
import sqlite3, os, json, glob
from collections import defaultdict

CH17 = "/Users/changjongkim/New_research/cloudlab_results/results/20260724/chain17"
OUT_JSON = "/Users/changjongkim/New_research/cloudlab_results/results/20260725/chain17_all_stats.json"

def extract_stats(sqlite_path):
    """Return {duty, dur_med, gap_med, gap_p95, gap_p99, dur_p95, kernel_count, wall_ns}
       Uses raw sqlite kernel activity."""
    try:
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        # Fetch (start, end, streamId, name) for kernels only
        # Join on string ids for name
        cur.execute("""
            SELECT k.start, k.end, k.streamId, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds s ON k.demangledName = s.id
            ORDER BY k.streamId, k.start
        """)
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return None
    if not rows: return None
    # Filter out memcpy/memset (they shouldn't be in KERNEL table but safety)
    kernels = [(s, e, sid, n or "") for s, e, sid, n in rows
               if n and 'memcpy' not in n.lower() and 'memset' not in n.lower()]
    if not kernels: return None
    # Group by stream
    by_stream = defaultdict(list)
    for s, e, sid, n in kernels:
        by_stream[sid].append((s, e))
    # Compute stats
    all_dur = []; all_gap = []
    for sid, kl in by_stream.items():
        kl.sort()
        for i, (s, e) in enumerate(kl):
            all_dur.append(e - s)
            if i > 0:
                g = s - kl[i-1][1]
                if g > 0: all_gap.append(g)
    import numpy as np
    d = np.array(all_dur)
    g = np.array(all_gap) if all_gap else np.array([0])
    t0 = min(k[0] for k in kernels); tf = max(k[1] for k in kernels)
    return dict(
        kernel_count = len(d),
        wall_ns = tf - t0,
        launch_rate = float(len(d) / ((tf-t0)/1e9)),
        dur_med = float(np.median(d)),
        dur_p95 = float(np.percentile(d, 95)),
        gap_med = float(np.median(g)),
        gap_p95 = float(np.percentile(g, 95)),
        gap_p99 = float(np.percentile(g, 99)),
        duty = float(d.sum() / (d.sum() + g.sum()) * 100),
        stream_count = len(by_stream),
    )

# Enumerate all sqlite files
files = sorted(glob.glob(f"{CH17}/cfg*.sqlite"))
print(f"Found {len(files)} sqlite files")

stats = {}
for i, f in enumerate(files):
    label = os.path.basename(f).replace(".sqlite", "")
    s = extract_stats(f)
    if s: stats[label] = s
    if (i+1) % 30 == 0: print(f"  processed {i+1}/{len(files)}")

with open(OUT_JSON, "w") as fp:
    json.dump(stats, fp, indent=2)
print(f"\nSaved {len(stats)} conditions to {OUT_JSON}")

# Quick summary — how many per config/N/mps
by_group = defaultdict(int)
for label in stats:
    parts = label.split("_")
    if len(parts) >= 4:
        cfg = parts[0]; nrx = parts[2]; mps = parts[3]
        by_group[(cfg, nrx, mps)] += 1
print(f"\nCoverage: {len(by_group)} unique (config, N, MPS) combos")
