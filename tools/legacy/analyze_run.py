"""
Comprehensive run analyzer — combines bimodal detection + SLA miss rate +
CDF/Q-Q + HBM BW correlation in one script.

Produces multiple analysis PNGs + text summary from a results directory.

Replaces and extends bimodal_detect.py.

Usage:
  python3 analyze_run.py /path/to/results/n20_split-60-40_qwen7b

Inputs in <run_dir>:
  run_*.json         — per-iter latency arrays (real_l1.py output)
  dmon.csv           — nvidia-smi dmon (optional)
  markers.txt        — run window timestamps (optional)

Outputs in <run_dir>:
  bimodal_analysis.png        — histogram + strip plot (cluster)
  cdf_qq.png                  — CDF + Q-Q plot
  sla_miss_rates.png          — SLA deadline miss rate per run
  dmon_correlation.png        — HBM BW vs L1 latency (if dmon.csv present)
  full_summary.txt            — all metrics combined
"""
import os
import sys
import json
import glob
import datetime
import re

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_runs(run_dir):
    """Return list of dicts: {idx, mean_ms, p99_ms, raw_ms, miss_1ms, json_mtime}"""
    files = sorted(glob.glob(os.path.join(run_dir, "run_*.json")))
    runs = []
    for f in files:
        m = re.search(r"run_(\d+)\.json", f)
        idx = int(m.group(1)) if m else len(runs) + 1
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  WARN: cannot parse {f}: {e}")
            continue
        runs.append({
            "idx": idx,
            "mean_ms": d.get("mean_ms"),
            "p50_ms": d.get("p50_ms"),
            "p95_ms": d.get("p95_ms"),
            "p99_ms": d.get("p99_ms"),
            "raw_ms": d.get("raw_ms", []),
            "miss_1ms": d.get("miss_1ms", 0),
            "iterations": d.get("iterations", 0),
            "mtime": os.path.getmtime(f),
        })
    return runs


def load_dmon(run_dir):
    """Parse dmon.csv: lines like 'YYYY MM DD HH:MM:SS  0  pwr gtemp mtemp sm mem ...'.
    Returns list of (epoch, mem_pct, sm_pct, pwr_w)."""
    fp = os.path.join(run_dir, "dmon.csv")
    if not os.path.exists(fp):
        return []
    rows = []
    with open(fp) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                # Format with -o DT: "2026 05 24 14:32:11  0  XXX  XX  XX  XX  XX  ..."
                yr, mo, dy = int(parts[0]), int(parts[1]), int(parts[2])
                hh, mm, ss = parts[3].split(":")
                dt = datetime.datetime(yr, mo, dy, int(hh), int(mm), int(ss))
                ep = dt.timestamp()
                # gpu_idx = parts[4]
                pwr = float(parts[5])
                # gtemp = parts[6], mtemp = parts[7]
                sm = float(parts[8])
                mem = float(parts[9])
                rows.append((ep, mem, sm, pwr))
            except (ValueError, IndexError):
                continue
    return rows


def load_markers(run_dir):
    """Parse markers.txt: 'YYYY-MM-DDTHH:MM:SS LABEL'. Returns list of (epoch, label)."""
    fp = os.path.join(run_dir, "markers.txt")
    if not os.path.exists(fp):
        return []
    rows = []
    with open(fp) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            try:
                dt = datetime.datetime.fromisoformat(parts[0])
                rows.append((dt.timestamp(), parts[1]))
            except ValueError:
                continue
    return rows


# ----------------------------------------------------------------------------
# Analyses
# ----------------------------------------------------------------------------
def two_means_1d(xs, n_iter=50):
    """1D 2-cluster k-means."""
    xs = np.asarray(xs, dtype=float)
    c1, c2 = xs.min(), xs.max()
    for _ in range(n_iter):
        d1 = np.abs(xs - c1); d2 = np.abs(xs - c2)
        labels = (d2 < d1).astype(int)
        if (labels == 0).sum() == 0 or (labels == 1).sum() == 0:
            break
        new_c1 = xs[labels == 0].mean(); new_c2 = xs[labels == 1].mean()
        if abs(new_c1 - c1) < 1e-6 and abs(new_c2 - c2) < 1e-6:
            break
        c1, c2 = new_c1, new_c2
    if c1 > c2:
        c1, c2 = c2, c1; labels = 1 - labels
    return (c1, c2), labels


def bimodality_score(xs, c1, c2):
    xs = np.asarray(xs, dtype=float)
    gap = abs(c2 - c1)
    s1 = xs[xs < (c1 + c2) / 2].std() if (xs < (c1 + c2) / 2).sum() > 1 else 0
    s2 = xs[xs >= (c1 + c2) / 2].std() if (xs >= (c1 + c2) / 2).sum() > 1 else 0
    avg_s = (s1 + s2) / 2
    return float("inf") if avg_s < 1e-6 else gap / avg_s


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_bimodal(runs, run_dir):
    means = np.array([r["mean_ms"] for r in runs if r["mean_ms"] is not None])
    if len(means) < 2:
        return None
    (c1, c2), labels = two_means_1d(means)
    score = bimodality_score(means, c1, c2)
    verdict = "BIMODAL" if score > 2 else ("AMBIGUOUS" if score > 1 else "UNIMODAL")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(means, bins=max(8, len(means) // 2), color="#2ca02c", alpha=0.7, edgecolor="black")
    ax.axvline(c1, color="green", ls="--", label=f"LOW {c1:.2f}")
    ax.axvline(c2, color="red", ls="--", label=f"HIGH {c2:.2f}")
    ax.set_xlabel("L1 mean (ms)"); ax.set_ylabel("Run count")
    ax.set_title(f"Distribution N={len(means)}, gap={c2-c1:.2f}ms, score={score:.2f} → {verdict}")
    ax.legend()

    ax = axes[1]
    colors = ["#2ca02c" if l == 0 else "#d62728" for l in labels]
    ax.scatter(range(1, len(means) + 1), means, c=colors, s=80, edgecolor="black", zorder=5)
    ax.axhline(c1, color="green", ls="--", alpha=0.5)
    ax.axhline(c2, color="red", ls="--", alpha=0.5)
    ax.set_xlabel("Run index"); ax.set_ylabel("L1 mean (ms)")
    ax.set_title("Sequential — LOW (green) vs HIGH (red)")

    fig.suptitle(f"Bimodal: {os.path.basename(run_dir)}", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(run_dir, "bimodal_analysis.png")
    fig.savefig(out, bbox_inches="tight", dpi=130); plt.close(fig)
    return {"verdict": verdict, "c1": c1, "c2": c2, "gap": c2 - c1, "score": score,
            "low_n": int((labels == 0).sum()), "high_n": int((labels == 1).sum())}


def plot_cdf_qq(runs, run_dir):
    """CDF + Q-Q plot of ALL per-iter latencies combined."""
    all_iters = []
    for r in runs:
        all_iters.extend(r["raw_ms"])
    if len(all_iters) < 10:
        return None
    iters = np.array(all_iters)
    iters_sorted = np.sort(iters)
    n = len(iters)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # CDF
    ax = axes[0]
    cdf = np.arange(1, n + 1) / n
    ax.plot(iters_sorted, cdf, lw=2, color="#1f77b4")
    ax.axvline(1.0, color="red", ls="--", alpha=0.5, label="1ms TTI deadline")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.4)
    ax.axhline(0.99, color="gray", ls=":", alpha=0.4)
    ax.set_xlabel("L1 latency per iter (ms)"); ax.set_ylabel("CDF")
    ax.set_title(f"CDF of all {n} per-iter latencies\n"
                 f"median={np.median(iters):.2f}, p99={np.percentile(iters,99):.2f}, max={iters.max():.2f}")
    ax.legend()

    # Q-Q vs normal — use numpy/erfinv (no scipy dep)
    ax = axes[1]
    theoretical_q = np.linspace(0.01, 0.99, n)
    z = np.array([_z_from_q(q) for q in theoretical_q])
    ax.scatter(z, iters_sorted, s=15, alpha=0.5, color="#2ca02c", edgecolor="black", lw=0.3)
    # Reference line through (q25, q75) — fit linear
    q25, q75 = np.percentile(iters, [25, 75])
    z25, z75 = _z_from_q(0.25), _z_from_q(0.75)
    if z75 != z25:
        slope = (q75 - q25) / (z75 - z25); intercept = q25 - slope * z25
        ax.plot(z, slope * z + intercept, "r--", alpha=0.7, label="Normal fit")
    ax.set_xlabel("Theoretical normal quantile (z)"); ax.set_ylabel("Observed latency (ms)")
    ax.set_title("Q-Q plot vs Normal\n(curves/steps = non-Gaussian, indicates multi-modal)")
    ax.legend()

    fig.suptitle(f"Distribution shape: {os.path.basename(run_dir)}", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(run_dir, "cdf_qq.png")
    fig.savefig(out, bbox_inches="tight", dpi=130); plt.close(fig)

    return {
        "n_iters": int(n),
        "median": float(np.median(iters)),
        "p95": float(np.percentile(iters, 95)),
        "p99": float(np.percentile(iters, 99)),
        "p999": float(np.percentile(iters, 99.9)) if n >= 1000 else None,
        "max": float(iters.max()),
        "tti_miss_pct": float(100 * np.sum(iters > 1.0) / n),
    }


def _erfinv(x):
    """Approximate erfinv (good enough for plotting)."""
    a = 0.147
    sign = 1 if x >= 0 else -1
    x_abs = abs(x)
    if x_abs >= 1: return sign * 5.0
    ln = np.log(1 - x_abs ** 2)
    t = 2 / (np.pi * a) + ln / 2
    return sign * np.sqrt(np.sqrt(t ** 2 - ln / a) - t)


def _z_from_q(q):
    return np.sqrt(2) * _erfinv(2 * q - 1)


def plot_sla_miss(runs, run_dir):
    """Per-run SLA miss rates at multiple thresholds."""
    valid = [r for r in runs if r["raw_ms"]]
    if not valid:
        return None
    thresholds_ms = [1.0, 1.5, 2.0, 5.0]   # absolute TTI bounds
    relative_thr = 1.1   # also compute % above mean
    fig, ax = plt.subplots(figsize=(11, 5.5))

    indices = [r["idx"] for r in valid]
    means = [r["mean_ms"] for r in valid]
    rates = {t: [] for t in thresholds_ms}
    rel_rates = []
    for r in valid:
        a = np.array(r["raw_ms"])
        for t in thresholds_ms:
            rates[t].append(100 * np.sum(a > t) / len(a))
        baseline = np.median(a)   # use per-run median as baseline
        rel_rates.append(100 * np.sum(a > baseline * relative_thr) / len(a))

    x = np.arange(len(valid))
    width = 0.18
    colors = ["#fdae61", "#f46d43", "#d73027", "#a50026"]
    for i, t in enumerate(thresholds_ms):
        ax.bar(x + (i - 1.5) * width, rates[t], width,
               label=f">{t}ms", color=colors[i], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"run{i}" for i in indices], fontsize=8)
    ax.set_ylabel("Fraction of iterations (%)")
    ax.set_title(f"SLA miss rate per run — fraction of TTIs exceeding deadline\n"
                 f"({os.path.basename(run_dir)})")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    out = os.path.join(run_dir, "sla_miss_rates.png")
    fig.savefig(out, bbox_inches="tight", dpi=130); plt.close(fig)

    # Aggregate
    agg = {f"{t}ms": float(np.mean(rates[t])) for t in thresholds_ms}
    agg["relative_10pct_over_median"] = float(np.mean(rel_rates))
    return agg


def plot_dmon_correlation(runs, dmon, markers, run_dir):
    """Correlate HBM BW utilization (from dmon) with L1 latency per run window."""
    if not dmon or not markers:
        return None

    # Match markers to runs: pairs of (run_X_start, run_X_end)
    starts = {m[1].replace("_start", ""): m[0] for m in markers if "_start" in m[1]}
    ends = {m[1].replace("_end", ""): m[0] for m in markers if "_end" in m[1]}
    matched = []
    for r in runs:
        key = f"run_{r['idx']}"
        if key in starts and key in ends:
            ep_s, ep_e = starts[key], ends[key]
            # find dmon entries in this window
            window = [(ep, mem, sm, pwr) for (ep, mem, sm, pwr) in dmon if ep_s <= ep <= ep_e]
            if window:
                avg_mem = np.mean([w[1] for w in window])
                avg_sm = np.mean([w[2] for w in window])
                avg_pwr = np.mean([w[3] for w in window])
                matched.append({
                    "idx": r["idx"], "mean_ms": r["mean_ms"],
                    "avg_mem_pct": avg_mem, "avg_sm_pct": avg_sm, "avg_pwr_w": avg_pwr,
                })
    if len(matched) < 3:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    means = [m["mean_ms"] for m in matched]
    mem_pcts = [m["avg_mem_pct"] for m in matched]
    sm_pcts = [m["avg_sm_pct"] for m in matched]
    pwrs = [m["avg_pwr_w"] for m in matched]

    # Scatter: HBM BW vs L1 latency
    ax = axes[0]
    ax.scatter(mem_pcts, means, s=80, alpha=0.7, color="#d62728", edgecolor="black")
    for m in matched:
        ax.annotate(f"r{m['idx']}", (m["avg_mem_pct"], m["mean_ms"]),
                    fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("HBM BW utilization avg (%) during run")
    ax.set_ylabel("L1 mean (ms)")
    ax.set_title("HBM BW ↔ L1 latency correlation")
    # correlation
    if len(matched) > 2:
        corr = np.corrcoef(mem_pcts, means)[0, 1]
        ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}",
                transform=ax.transAxes, fontsize=10, color="red", fontweight="bold",
                verticalalignment="top")

    # Scatter: SM util vs L1 latency
    ax = axes[1]
    ax.scatter(sm_pcts, means, s=80, alpha=0.7, color="#2ca02c", edgecolor="black")
    ax.set_xlabel("SM utilization avg (%) during run")
    ax.set_ylabel("L1 mean (ms)")
    ax.set_title("SM util ↔ L1 latency")
    if len(matched) > 2:
        corr = np.corrcoef(sm_pcts, means)[0, 1]
        ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}",
                transform=ax.transAxes, fontsize=10, color="red", fontweight="bold",
                verticalalignment="top")

    # Scatter: Power vs L1 latency
    ax = axes[2]
    ax.scatter(pwrs, means, s=80, alpha=0.7, color="#1f77b4", edgecolor="black")
    ax.set_xlabel("Power draw avg (W) during run")
    ax.set_ylabel("L1 mean (ms)")
    ax.set_title("Power ↔ L1 latency")
    if len(matched) > 2:
        corr = np.corrcoef(pwrs, means)[0, 1]
        ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}",
                transform=ax.transAxes, fontsize=10, color="red", fontweight="bold",
                verticalalignment="top")

    fig.suptitle(f"dmon correlation: {os.path.basename(run_dir)}", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(run_dir, "dmon_correlation.png")
    fig.savefig(out, bbox_inches="tight", dpi=130); plt.close(fig)

    return {
        "n_matched": len(matched),
        "mem_corr": float(np.corrcoef(mem_pcts, means)[0, 1]) if len(matched) > 2 else None,
        "sm_corr": float(np.corrcoef(sm_pcts, means)[0, 1]) if len(matched) > 2 else None,
        "pwr_corr": float(np.corrcoef(pwrs, means)[0, 1]) if len(matched) > 2 else None,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(run_dir):
    print(f"Analyzing {run_dir}")
    runs = load_runs(run_dir)
    if not runs:
        print("  ERROR: no run_*.json found"); sys.exit(1)
    dmon = load_dmon(run_dir)
    markers = load_markers(run_dir)
    print(f"  loaded {len(runs)} runs, {len(dmon)} dmon samples, {len(markers)} markers")

    summary = {
        "run_dir": run_dir,
        "n_runs": len(runs),
        "bimodal": plot_bimodal(runs, run_dir),
        "distribution": plot_cdf_qq(runs, run_dir),
        "sla_miss": plot_sla_miss(runs, run_dir),
        "dmon_corr": plot_dmon_correlation(runs, dmon, markers, run_dir),
    }

    # Write text summary
    out_txt = os.path.join(run_dir, "full_summary.txt")
    with open(out_txt, "w") as f:
        f.write(f"# Analysis — {run_dir}\n\n")
        f.write(f"N runs: {summary['n_runs']}\n\n")
        if summary["bimodal"]:
            b = summary["bimodal"]
            f.write(f"## Bimodal cluster\n")
            f.write(f"  verdict     : {b['verdict']}\n")
            f.write(f"  LOW cluster : {b['c1']:.3f} ms (N={b['low_n']})\n")
            f.write(f"  HIGH cluster: {b['c2']:.3f} ms (N={b['high_n']})\n")
            f.write(f"  gap         : {b['gap']:.3f} ms\n")
            f.write(f"  bimodality  : {b['score']:.2f}\n\n")
        if summary["distribution"]:
            d = summary["distribution"]
            f.write(f"## Distribution (per-iter)\n")
            f.write(f"  iters    : {d['n_iters']}\n")
            f.write(f"  median   : {d['median']:.3f} ms\n")
            f.write(f"  p95      : {d['p95']:.3f} ms\n")
            f.write(f"  p99      : {d['p99']:.3f} ms\n")
            if d.get("p999"): f.write(f"  p99.9    : {d['p999']:.3f} ms\n")
            f.write(f"  max      : {d['max']:.3f} ms\n")
            f.write(f"  >1ms TTI : {d['tti_miss_pct']:.2f}%\n\n")
        if summary["sla_miss"]:
            s = summary["sla_miss"]
            f.write(f"## SLA miss rates (per-run avg)\n")
            for k, v in s.items():
                f.write(f"  {k:<30}: {v:.2f}%\n")
            f.write("\n")
        if summary["dmon_corr"]:
            c = summary["dmon_corr"]
            f.write(f"## dmon correlation (N={c['n_matched']} matched runs)\n")
            if c["mem_corr"] is not None: f.write(f"  HBM BW ↔ L1 mean (Pearson r): {c['mem_corr']:+.3f}\n")
            if c["sm_corr"] is not None:  f.write(f"  SM util ↔ L1 mean (Pearson r): {c['sm_corr']:+.3f}\n")
            if c["pwr_corr"] is not None: f.write(f"  Power ↔ L1 mean (Pearson r): {c['pwr_corr']:+.3f}\n")
            f.write("\n")

    # Console summary
    print(f"\n=== Summary: {os.path.basename(run_dir)} ===")
    if summary["bimodal"]:
        b = summary["bimodal"]
        print(f"  VERDICT: {b['verdict']}  (LOW={b['c1']:.2f}, HIGH={b['c2']:.2f}, score={b['score']:.2f})")
    if summary["distribution"]:
        d = summary["distribution"]
        print(f"  TTI miss rate (>1ms): {d['tti_miss_pct']:.2f}%")
    if summary["dmon_corr"]:
        c = summary["dmon_corr"]
        if c["mem_corr"] is not None:
            print(f"  HBM BW correlation r = {c['mem_corr']:+.3f}  "
                  f"({'STRONG' if abs(c['mem_corr']) > 0.5 else 'weak'})")
    print(f"  Saved: {out_txt}")
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_run.py <results_dir>"); sys.exit(1)
    main(sys.argv[1])
