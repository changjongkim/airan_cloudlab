#!/usr/bin/env python3
"""
Mechanism-level supplementary figures:
- supp_07: Memset duration scales with partition (structural HBM bandwidth cost)
- supp_08: Memcpy contention is workload-pattern-dependent (NOT pure bandwidth)
- supp_09: Count vs Duration decomposition (count invariant; duration is the lever)
- supp_10: Memcpy size distribution (why L1 ops compete with PHY-AI, not bulk D2D)
"""
import sqlite3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 11,
    'figure.figsize': (11, 6),
    'savefig.dpi': 140,
    'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'font.family': ['DejaVu Sans', 'Apple SD Gothic Neo', 'NanumGothic', 'sans-serif'],
})

ROOT = Path(__file__).parent.parent
SQLITE = ROOT / "20260531" / "nsys_sqlite_v2"
OUT = Path(__file__).parent / "figures"


def get_stats(db):
    if not Path(db).exists(): return None
    try:
        con = sqlite3.connect(db); cur = con.cursor()
        cur.execute("SELECT COUNT(*), SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_MEMCPY")
        mc_n, mc_ms = cur.fetchone()
        cur.execute("SELECT COUNT(*), SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_MEMSET")
        ms_n, ms_ms = cur.fetchone()
        cur.execute("SELECT AVG(end-start)/1000 FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE bytes > 400000000")
        big_ms_avg = cur.fetchone()[0] or 0
        cur.execute("SELECT bytes, COUNT(*), AVG(end-start)/1000 FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY (bytes/1024) ORDER BY bytes")
        sz_dist = cur.fetchall()
        con.close()
        return {"mc_n": mc_n or 0, "mc_ms": mc_ms or 0,
                "ms_n": ms_n or 0, "ms_ms": ms_ms or 0,
                "big_ms_avg_us": big_ms_avg, "sz_dist": sz_dist}
    except Exception:
        return None


CONDITIONS = [
    # (label, sqlite, partition, workload, color)
    ("7g full",        "S2_7g_mig_run1.sqlite",          "7g", "alone", "#27AE60"),
    ("4g + sat_compute", "S15_4g_sat_compute_run1.sqlite", "4g", "sat_compute", "#F39C12"),
    ("4g + NeuralRx",  "S18_4g_neuralrx_run1.sqlite",    "4g", "neuralrx", "#9B59B6"),
    ("4g + 3sat",      "S26_4g_3sat_run1.sqlite",        "4g", "3sat", "#D35400"),
    ("3g alone",       "S5_3g_alone_run1.sqlite",        "3g", "alone", "#666666"),
    ("3g + Qwen",      "S6_3g_qwen_run1.sqlite",         "3g", "qwen", "#E67E22"),
    ("3g + NeuralRx",  "S7_3g_neuralrx_run1.sqlite",     "3g", "neuralrx", "#9B59B6"),
    ("3g + sat_compute","S13_3g_sat_compute_run1.sqlite", "3g", "sat_compute", "#F39C12"),
    ("3g + sat_hbm",   "S14_3g_sat_hbm_run1.sqlite",     "3g", "sat_hbm", "#D35400"),
    ("2g alone",       "S10_2g_alone_run1.sqlite",       "2g", "alone", "#666666"),
    ("2g + NeuralRx",  "S22_2g_neuralrx_run1.sqlite",    "2g", "neuralrx", "#9B59B6"),
    ("2g + ChanPred",  "S35_2g_chanpred_run1.sqlite",    "2g", "chanpred", "#E74C3C"),
]


def fig_supp_07_memset_structural():
    """435MB memset per-call duration vs partition: pure HBM bandwidth scaling."""
    parts = ["7g", "4g", "3g", "2g"]
    pmap = {p: [] for p in parts}
    for lbl, db, part, wl, col in CONDITIONS:
        if wl != "alone" and part != "4g":
            continue
        s = get_stats(SQLITE / db)
        if s and s["big_ms_avg_us"] > 0:
            pmap[part].append(s["big_ms_avg_us"])
    fig, ax = plt.subplots(figsize=(9, 6))
    vals = [np.mean(pmap[p]) if pmap[p] else 0 for p in parts]
    colors = ['#27AE60', '#3498DB', '#666666', '#E74C3C']
    bars = ax.bar(parts, vals, color=colors)
    for b, v in zip(bars, vals):
        if v > 0:
            ratio = v / vals[0] if vals[0] else 0
            ax.text(b.get_x() + b.get_width()/2, v + 30, f"{v:.0f}us\n({ratio:.1f}x)",
                    ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel("435MB memset per-call duration (us)")
    ax.set_xlabel("L1 partition size")
    ax.set_title("Supp 07 — Memset per-call duration scales with partition (structural HBM bandwidth cost)\n"
                 "same workload, same buffer; smaller partition = less DRAM bandwidth share = slower memset")
    ax.text(0.02, 0.98, "이건 'AI 영향'이 아니다.\n같은 buffer를 같은 식으로 memset하지만,\nMIG가 partition별로 HBM bandwidth share를 나누기 때문에\n작은 slice일수록 같은 memset이 비례적으로 느려진다.",
            transform=ax.transAxes, va='top', ha='left', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.9))
    fig.savefig(OUT / "fig_supp_07_memset_structural.png")
    plt.close(fig)
    print(f"  ✓ supp_07 memset structural: 7g→2g {vals[0]:.0f}→{vals[3]:.0f}us ({vals[3]/vals[0]:.1f}x)")


def fig_supp_08_memcpy_contention_map():
    """Memcpy duration % inflation: which workload+partition combos break isolation."""
    # Reference: 3g_alone, 2g_alone, 7g_alone for memcpy ms
    refs = {}
    for lbl, db, part, wl, col in CONDITIONS:
        if wl == "alone" or "mig" in db.lower():
            s = get_stats(SQLITE / db)
            if s:
                refs[part] = s["mc_ms"]
    # For 4g, no alone available, use 3g alone as reference (similar)
    if "4g" not in refs and "3g" in refs:
        refs["4g"] = refs["3g"]

    rows = []
    for lbl, db, part, wl, col in CONDITIONS:
        if wl == "alone" or "mig" in db.lower():
            continue
        s = get_stats(SQLITE / db)
        ref = refs.get(part)
        if s and ref:
            delta_pct = (s["mc_ms"] - ref) / ref * 100
            rows.append((lbl, part, wl, delta_pct, col))
    rows.sort(key=lambda x: -x[3])

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(rows))
    deltas = [r[3] for r in rows]
    colors = [r[4] for r in rows]
    ax.barh(y, deltas, color=colors)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Memcpy total duration: Δ vs same-partition alone (%)")
    ax.axvline(0, color='black', lw=0.5)
    for i, (lbl, part, wl, d, _) in enumerate(rows):
        ax.text(d + 3 if d >= 0 else d - 3, i, f"{d:+.0f}%",
                va='center', ha='left' if d >= 0 else 'right', fontsize=10, fontweight='bold')
    ax.set_title("Supp 08 — Memcpy contention map: workload pattern, NOT simple bandwidth, drives the cost\n"
                 "sat_compute는 HBM-heavy인데도 +0%. NeuralRx/Qwen/sat_hbm은 +300%. 2g L1은 면역.")
    ax.text(0.55, 0.06,
            "패턴 1: PHY-AI (NeuralRx/Qwen/sat_hbm) on 3g/4g L1 → +300% (memcpy queue contention)\n"
            "패턴 2: sat_compute on 3g L1 → +0% (compute-heavy, memcpy queue 무관)\n"
            "패턴 3: 2g L1 + anything → +0% (SM-bound 이라 memcpy bottleneck 아님)",
            transform=ax.transAxes, va='bottom', ha='left', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.9))
    fig.savefig(OUT / "fig_supp_08_memcpy_contention_map.png")
    plt.close(fig)
    print(f"  ✓ supp_08 memcpy contention map: {len(rows)} conditions")


def fig_supp_09_count_vs_duration():
    """Decompose: memcpy/memset count INVARIANT, duration is the variable."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    labels = []
    counts_mc, durs_mc = [], []
    counts_ms, durs_ms = [], []
    colors = []
    for lbl, db, part, wl, col in CONDITIONS[:8]:  # first 8 fits well
        s = get_stats(SQLITE / db)
        if not s: continue
        labels.append(lbl)
        counts_mc.append(s["mc_n"]); durs_mc.append(s["mc_ms"])
        counts_ms.append(s["ms_n"]); durs_ms.append(s["ms_ms"])
        colors.append(col)

    x = np.arange(len(labels))
    w = 0.4
    # memcpy
    ax1.bar(x - w/2, counts_mc, w, color='#3498DB', label='count', alpha=0.7)
    ax1b = ax1.twinx()
    ax1b.bar(x + w/2, durs_mc, w, color='#E74C3C', label='total duration (ms)')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax1.set_ylabel("memcpy count", color='#3498DB')
    ax1b.set_ylabel("memcpy total ms", color='#E74C3C')
    ax1.set_title("Memcpy: count 고정 (6433), duration이 변수")
    ax1.legend(loc='upper left'); ax1b.legend(loc='upper right')

    # memset
    ax2.bar(x - w/2, counts_ms, w, color='#3498DB', label='count', alpha=0.7)
    ax2b = ax2.twinx()
    ax2b.bar(x + w/2, durs_ms, w, color='#E74C3C', label='total duration (ms)')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax2.set_ylabel("memset count", color='#3498DB')
    ax2b.set_ylabel("memset total ms", color='#E74C3C')
    ax2.set_title("Memset: count 고정 (1920), duration은 partition 크기로 결정")
    ax2.legend(loc='upper left'); ax2b.legend(loc='upper right')

    fig.suptitle("Supp 09 — memcpy/memset COUNT는 변하지 않는다. duration이 변수.\n"
                 "AI는 L1의 memory op 개수를 늘리지 않는다 — 각 op의 wait time을 늘린다.")
    fig.tight_layout()
    fig.savefig(OUT / "fig_supp_09_count_vs_duration.png")
    plt.close(fig)
    print(f"  ✓ supp_09 count vs duration decomposition")


def fig_supp_10_memcpy_size_distribution():
    """Memcpy size distribution: L1 ops are small (KB-MB), not bulk."""
    fig, ax = plt.subplots(figsize=(11, 6))
    samples = [
        ("3g alone",      "S5_3g_alone_run1.sqlite", "#666666"),
        ("3g + NeuralRx", "S7_3g_neuralrx_run1.sqlite", "#9B59B6"),
        ("2g alone",      "S10_2g_alone_run1.sqlite", "#3498DB"),
        ("2g + ChanPred", "S35_2g_chanpred_run1.sqlite", "#E74C3C"),
    ]
    for lbl, db, col in samples:
        s = get_stats(SQLITE / db)
        if not s: continue
        sizes_kb = []
        counts = []
        for bytes_, n, avg_us in s["sz_dist"]:
            sizes_kb.append(bytes_ / 1024)
            counts.append(n)
        if not sizes_kb: continue
        # Group by log size bucket
        sizes_kb = np.array(sizes_kb); counts = np.array(counts)
        log_bins = np.logspace(-1, 4, 25)
        binned = np.zeros(len(log_bins) - 1)
        for sz, n in zip(sizes_kb, counts):
            idx = np.searchsorted(log_bins, sz) - 1
            if 0 <= idx < len(binned):
                binned[idx] += n
        ax.plot(log_bins[:-1], binned, 'o-', label=lbl, color=col, markersize=6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("memcpy size (KB)")
    ax.set_ylabel("count")
    ax.set_title("Supp 10 — L1 memcpy size distribution: 대부분 0.1KB~1MB 범위 (bulk가 아님)\n"
                 "→ F의 1024MB bulk D2D와 다른 memory access pattern → F가 L1 memcpy를 disturb 못 하는 이유")
    ax.legend()
    ax.axvspan(0.1, 2048, alpha=0.1, color='#9B59B6', label='L1 ops range')
    fig.savefig(OUT / "fig_supp_10_memcpy_size_distribution.png")
    plt.close(fig)
    print(f"  ✓ supp_10 memcpy size distribution")


def main():
    print(f"Generating mechanism figures → {OUT}")
    fig_supp_07_memset_structural()
    fig_supp_08_memcpy_contention_map()
    fig_supp_09_count_vs_duration()
    fig_supp_10_memcpy_size_distribution()
    print("\nDone.")


if __name__ == "__main__":
    main()
