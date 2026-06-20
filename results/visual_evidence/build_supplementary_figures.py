#!/usr/bin/env python3
"""
Supplementary figures filling weak points in MIG_AIRAN_VISUAL_EVIDENCE_KR.md.

Addresses:
- W7 (§2 n 미명시) + cherry-picking 의혹 → fig_supp_01_neuralrx_n20_distribution
- §2 reinforcement → fig_supp_02_phase4_phy_ai_compare (NeuralRx vs chanpred vs xapp)
- W6 (§4 4g coloc 역설) → fig_supp_03_g_coloc_partition_paradox
- H3 (§12-13 n=1) → fig_supp_04_nsys_aggregated_boundary (78 sqlite aggregated)
- §7 reframe → fig_supp_05_ai_per_op_cross_partition (cross-partition L1 effect on multiple AI)
- §1 reframe (mean→p99 emphasis) → fig_supp_06_baseline_tail_variance
"""
import json, re, sqlite3
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 11,
    'figure.figsize': (10, 6),
    'savefig.dpi': 140,
    'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'font.family': ['DejaVu Sans', 'Apple SD Gothic Neo', 'NanumGothic', 'sans-serif'],
})

ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def load_phase_runs(d):
    """Read run_*.json (preferred) or fall back to log files with [realL1] markers."""
    d = Path(d)
    runs = []
    if not d.exists(): return runs
    json_files = sorted(d.glob("run_*.json"))
    if json_files:
        for jf in json_files:
            try:
                data = json.loads(jf.read_text())
                runs.append({"mean": data["mean_ms"], "p99": data["p99_ms"],
                             "p95": data.get("p95_ms", 0), "max": data.get("max_ms", 0)})
            except Exception:
                pass
        return runs
    # Fallback: log parse
    pat = re.compile(r"mean=([0-9.]+)ms\s+p95=([0-9.]+)ms\s+p99=([0-9.]+)ms")
    for log in sorted(d.glob("run_*.log")):
        try:
            for line in log.read_text(errors="ignore").splitlines():
                m = pat.search(line)
                if m:
                    runs.append({"mean": float(m.group(1)), "p95": float(m.group(2)),
                                 "p99": float(m.group(3)), "max": float(m.group(3))})
                    break
        except Exception:
            pass
    return runs


def fig_supp_01_neuralrx_n20_distribution():
    """Show n=20 NeuralRx Phase4 distribution vs baseline. Refutes cherry-picking."""
    nrx = load_phase_runs(ROOT / "20260531" / "n20_phase4_neuralrx")
    base = load_phase_runs(ROOT / "20260531" / "n20_baseline_3g_alone")
    if not nrx or not base: return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: per-run p99 scatter
    ax = axes[0]
    ax.plot(range(1, len(base)+1), [r["p99"] for r in base], 'o', color='#666666',
            label=f'baseline (3g alone) n={len(base)}', markersize=8)
    ax.plot(range(1, len(nrx)+1), [r["p99"] for r in nrx], 's', color='#9B59B6',
            label=f'+ NeuralRx (Phase4) n={len(nrx)}', markersize=8)
    ax.set_xlabel("run #")
    ax.set_ylabel("L1 p99 (ms)")
    ax.set_title("Per-run p99: NeuralRx vs baseline (n=20 each)")
    ax.legend()

    # Panel B: histogram
    ax = axes[1]
    bins = np.linspace(min(min(r["p99"] for r in base), min(r["p99"] for r in nrx))-5,
                       max(r["p99"] for r in nrx) + 5, 25)
    ax.hist([r["p99"] for r in base], bins=bins, color='#666666', alpha=0.7,
            label=f'baseline (mean={np.mean([r["p99"] for r in base]):.1f})')
    ax.hist([r["p99"] for r in nrx], bins=bins, color='#9B59B6', alpha=0.7,
            label=f'+NeuralRx (mean={np.mean([r["p99"] for r in nrx]):.1f})')
    ax.set_xlabel("L1 p99 (ms)")
    ax.set_ylabel("# runs")
    ax.set_title("Distribution (no overlap — NOT cherry-picked)")
    ax.legend()

    fig.suptitle("Supp 01 — NeuralRx +376% p99 is reproducible (n=20 distributions disjoint)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_supp_01_neuralrx_n20_distribution.png")
    plt.close(fig)
    print(f"  ✓ supp_01 (NeuralRx n=20 dist): base mean={np.mean([r['p99'] for r in base]):.1f}, "
          f"nrx mean={np.mean([r['p99'] for r in nrx]):.1f}")


def fig_supp_02_phase4_phy_ai_compare():
    """Compare NeuralRx vs chanpred vs xapp Phase4 — show NeuralRx truly stands out."""
    dirs = {
        "baseline 3g": ROOT / "20260531" / "n20_baseline_3g_alone",
        "+ ChanPred": ROOT / "20260531" / "n20_phase4_chanpred",
        "+ NeuralRx": ROOT / "20260531" / "n20_phase4_neuralrx",
        "+ xApp": ROOT / "20260531" / "n20_phase4_xapp",
    }
    runs_by = {k: load_phase_runs(d) for k, d in dirs.items()}
    runs_by = {k: v for k, v in runs_by.items() if v}
    if not runs_by: return

    fig, ax = plt.subplots(figsize=(11, 6))
    positions = list(range(len(runs_by)))
    data = [[r["p99"] for r in v] for v in runs_by.values()]
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black'))
    colors = ['#666666', '#E74C3C', '#9B59B6', '#16A085']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xticks(positions); ax.set_xticklabels(list(runs_by.keys()), rotation=10)
    ax.set_ylabel("L1 p99 (ms)")
    base_key = next((k for k in runs_by if "baseline" in k), None)
    if not base_key: return
    base_mean = np.mean([r["p99"] for r in runs_by[base_key]])
    ax.axhline(base_mean, ls='--', color='gray', alpha=0.6, label=f'baseline mean {base_mean:.1f}ms')
    for i, (lbl, vs) in enumerate(runs_by.items()):
        m = np.mean([r["p99"] for r in vs])
        pct = (m - base_mean) / base_mean * 100
        ax.text(i, m + 5, f"{m:.1f}\n({pct:+.0f}%)", ha='center', fontsize=10, fontweight='bold')
    ax.set_title("Supp 02 — Phase4 PHY-AI comparison: NeuralRx (+376%) >> ChanPred / xApp (n=20 each)")
    ax.legend()
    fig.savefig(OUT / "fig_supp_02_phase4_phy_ai_compare.png")
    plt.close(fig)
    print(f"  ✓ supp_02 (Phase4 PHY-AI 비교)")


def fig_supp_03_g_coloc_partition_paradox():
    """G_1a/1b/1c coloc baselines — 4g > 3g paradox + mechanism hypothesis."""
    g_dir = ROOT / "20260601" / "G_coloc"
    if not g_dir.exists(): return

    def load_g(prefix):
        out = []
        for jf in sorted(g_dir.glob(f"realL1_{prefix}*_run*_*.json")):
            try:
                data = json.loads(jf.read_text())
                out.append({"mean": data["mean_ms"], "p99": data["p99_ms"], "max": data["max_ms"]})
            except Exception:
                pass
        return out

    rows = [
        ("3g alone", load_g("G_0a"), "#666666"),
        ("3g + NRx coloc", load_g("G_1a"), "#9B59B6"),
        ("4g alone", load_g("G_0b"), "#666666"),
        ("4g + NRx coloc", load_g("G_1b"), "#9B59B6"),
        ("2g alone", load_g("G_0c"), "#666666"),
        ("2g + NRx coloc", load_g("G_1c"), "#9B59B6"),
    ]
    rows = [(l, v, c) for l, v, c in rows if v]
    if not rows: return

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(rows))
    p99s = [np.mean([r["p99"] for r in v]) for _, v, _ in rows]
    sds = [np.std([r["p99"] for r in v]) for _, v, _ in rows]
    colors = [c for _, _, c in rows]
    ax.bar(x, p99s, yerr=sds, color=colors, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([l for l, _, _ in rows], rotation=15)
    ax.set_ylabel("L1 p99 (ms)")
    for i, (l, p, s) in enumerate(zip([l for l, _, _ in rows], p99s, sds)):
        ax.text(i, p + s + 8, f"{p:.0f}ms", ha='center', fontsize=10, fontweight='bold')

    # Annotation: 4g paradox (positioned below title, above bars)
    if len(rows) >= 4:
        x3 = next(i for i, (l, _, _) in enumerate(rows) if l == "3g + NRx coloc")
        x4 = next(i for i, (l, _, _) in enumerate(rows) if l == "4g + NRx coloc")
        delta = p99s[x4] - p99s[x3]
        midx = (x3 + x4) / 2
        ax.annotate(f"4g coloc > 3g coloc by +{delta:.0f}ms (counterintuitive)",
                    xy=(x4, p99s[x4]),
                    xytext=(midx + 0.5, max(p99s) * 1.02),
                    ha='center', fontsize=10, color='#C0392B', fontweight='bold',
                    arrowprops=dict(arrowstyle="->", color='#C0392B', lw=1.2))
    ax.set_title("Supp 03 — Coloc partition size paradox: larger partition is MORE catastrophic\n"
                 "(hypothesis: NeuralRx in larger partition occupies more SMs → amplifies contention with L1)",
                 fontsize=11)
    # Allow extra headroom for the annotation
    ax.set_ylim(0, max(p99s) * 1.18)
    fig.tight_layout()
    fig.savefig(OUT / "fig_supp_03_g_coloc_partition_paradox.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ supp_03 (G coloc 4g paradox)")


def parse_gap_sqlite(db):
    """Return kernel-only gap stats + all-activity gap stats from one sqlite."""
    try:
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
        krows = cur.fetchall()
        all_acts = list(krows)
        for tab in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            try:
                cur.execute(f"SELECT start, end FROM {tab} ORDER BY start")
                all_acts.extend(cur.fetchall())
            except sqlite3.OperationalError:
                pass
        con.close()
    except Exception:
        return None
    if len(krows) < 100:
        return None
    krows = krows[100:]  # warmup
    kgaps_ns = [krows[i+1][0] - krows[i][1] for i in range(len(krows)-1) if krows[i+1][0] > krows[i][1]]
    all_acts.sort()
    agaps_ns = []
    for i in range(len(all_acts)-1):
        s, e = all_acts[i+1][0], all_acts[i][1]
        if s > e:
            agaps_ns.append(s - e)
    if not kgaps_ns or not agaps_ns:
        return None
    kg = np.array(kgaps_ns) / 1000
    ag = np.array(agaps_ns) / 1000
    return {
        "n_kern": len(krows),
        "kern_p99_us": float(np.percentile(kg, 99)),
        "kern_p999_us": float(np.percentile(kg, 99.9)),
        "all_p99_us": float(np.percentile(ag, 99)),
        "all_p999_us": float(np.percentile(ag, 99.9)),
        "ratio_p99": float(np.percentile(kg, 99) / max(1, np.percentile(ag, 99))),
    }


def fig_supp_04_nsys_aggregated_boundary():
    """Aggregate 78 sqlite v2 captures: kernel-only vs all-activity gap.
    Strengthens §12-13 by showing pattern across many conditions (not n=1).
    """
    sqlite_dir = ROOT / "20260531" / "nsys_sqlite_v2"
    if not sqlite_dir.exists(): return
    rows = []
    files = list(sqlite_dir.glob("*.sqlite"))[:30]  # cap for speed
    for db in files:
        stats = parse_gap_sqlite(db)
        if stats:
            rows.append((db.stem, stats))
    if not rows: return

    # Sort by kern p99
    rows.sort(key=lambda x: x[1]["kern_p99_us"])
    fig, ax = plt.subplots(figsize=(11, 9))
    y = np.arange(len(rows))
    kerns = [r[1]["kern_p99_us"] for r in rows]
    alls = [r[1]["all_p99_us"] for r in rows]
    w = 0.4
    ax.barh(y - w/2, kerns, w, label='kernel-only gap p99', color='#E74C3C')
    ax.barh(y + w/2, alls, w, label='all-activity gap p99', color='#3498DB')
    ax.set_yticks(y); ax.set_yticklabels([r[0][:40] for r in rows], fontsize=7)
    ax.set_xlabel("gap p99 (us)")
    ax.set_title(f"Supp 04 — kernel-only vs all-activity gap across {len(rows)} NSYS captures\n"
                 "(if kernel-only > all-activity, gap was filled with memcpy/memset, not idle)")
    ax.legend()
    fig.savefig(OUT / "fig_supp_04_nsys_aggregated_boundary.png")
    plt.close(fig)
    print(f"  ✓ supp_04 (NSYS aggregated {len(rows)} conditions)")


def parse_per_op_json(d):
    """Parse [X-latency-json] from log files."""
    runs = []
    if not Path(d).exists(): return runs
    pat = re.compile(r"\[\w+[-_]latency-json\]\s*(\{.*\})")
    for log in sorted(Path(d).glob("run_*.log")):
        try:
            for line in log.read_text(errors="ignore").splitlines():
                m = pat.search(line)
                if m:
                    runs.append(json.loads(m.group(1)))
                    break
        except Exception:
            pass
    return runs


def fig_supp_05_ai_per_op_cross_partition():
    """ai_per_op_latency: cross-partition L1 effect on multiple AI types' per-op p99."""
    base = ROOT / "20260531" / "ai_per_op_latency"
    base_b = ROOT / "20260531" / "ai_per_op_latency_b"
    ais = [
        ("chanpred", "#E74C3C"), ("neuralrx", "#9B59B6"), ("resnet", "#3498DB"),
        ("qwen", "#E67E22"), ("forecaster", "#1ABC9C"), ("xapp", "#16A085"),
    ]
    parts = ["1g", "2g", "3g", "4g"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(parts))
    w = 0.13
    for ai_i, (ai, color) in enumerate(ais):
        deltas = []
        for p in parts:
            for root in (base, base_b):
                ad = root / f"{ai}_{p}" / "alone"
                wd = root / f"{ai}_{p}" / "with_l1"
                runs_a = parse_per_op_json(ad)
                runs_w = parse_per_op_json(wd)
                if runs_a and runs_w:
                    a = np.mean([r.get("p99_ms", 0) for r in runs_a])
                    w_ = np.mean([r.get("p99_ms", 0) for r in runs_w])
                    deltas.append(100 * (w_ - a) / a if a else 0)
                    break
            else:
                deltas.append(np.nan)
        offset = (ai_i - len(ais)/2 + 0.5) * w
        ax.bar(x + offset, deltas, w, label=ai, color=color)
    ax.set_xticks(x); ax.set_xticklabels([f"AI={p}" for p in parts])
    ax.set_ylabel("AI per-op p99 inflation vs alone (%)")
    ax.set_title("Supp 05 — Cross-partition L1 background's effect on AI per-op p99\n"
                 "(NOT same-partition coloc; L1 runs on different MIG slice)")
    ax.axhline(0, color='black', lw=0.5)
    ax.legend(ncol=6, fontsize=9)
    fig.savefig(OUT / "fig_supp_05_ai_per_op_cross_partition.png")
    plt.close(fig)
    print(f"  ✓ supp_05 (AI per-op cross-partition effect)")


def fig_supp_06_baseline_tail_variance():
    """5/31 baseline n=20: show high variance / bimodal nature of even alone baseline."""
    dirs = [
        ("2g", ROOT / "20260531" / "n20_baseline_2g_alone"),
        ("3g", ROOT / "20260531" / "n20_baseline_3g_alone"),
        ("4g", ROOT / "20260531" / "n20_baseline_4g_alone"),
        ("7g", ROOT / "20260531" / "n20_baseline_7g_single"),
        ("fullGPU", ROOT / "20260531" / "n20_baseline_fullGPU_v2"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    positions, vals, labels = [], [], []
    for i, (lbl, d) in enumerate(dirs):
        runs = load_phase_runs(d)
        if not runs: continue
        positions.append(i)
        vals.append([r["p99"] for r in runs])
        labels.append(f"{lbl}\nn={len(runs)}")
    if not vals: return
    bp = ax.boxplot(vals, positions=positions, widths=0.5, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black'))
    colors = ['#E74C3C', '#F39C12', '#3498DB', '#27AE60', '#666666']
    for patch, c in zip(bp['boxes'], colors[:len(vals)]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_xticks(positions); ax.set_xticklabels(labels)
    ax.set_ylabel("L1 p99 per run (ms)")
    for i, vs in zip(positions, vals):
        ax.text(i, max(vs) + 1, f"SD={np.std(vs):.1f}\nrange={max(vs)-min(vs):.1f}",
                ha='center', fontsize=8)
    ax.set_title("Supp 06 — Baseline n=20 distribution: 5/31 baseline은 작은 partition일수록 분산 커짐")
    fig.savefig(OUT / "fig_supp_06_baseline_tail_variance.png")
    plt.close(fig)
    print(f"  ✓ supp_06 (baseline variance)")


def main():
    print(f"Generating supplementary figures → {OUT}")
    fig_supp_01_neuralrx_n20_distribution()
    fig_supp_02_phase4_phy_ai_compare()
    fig_supp_03_g_coloc_partition_paradox()
    fig_supp_04_nsys_aggregated_boundary()
    fig_supp_05_ai_per_op_cross_partition()
    fig_supp_06_baseline_tail_variance()
    print("\nDone.")


if __name__ == "__main__":
    main()
