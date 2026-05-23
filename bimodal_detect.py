"""
Bimodal cluster detection for N=20 split-60-40 + Qwen runs.

Reads a directory of run_<i>.json files produced by run_n20.sh, extracts
mean_ms from each, then:
  1. Plots a histogram + strip plot
  2. Tests for bimodality (Hartigan's dip test approximation via simple gap
     detection — we don't ship scipy.cluster, so use 1D 2-means)
  3. Reports: mean, std, p99, mode count, gap, cluster fractions
  4. Saves PNG to <out_dir>/bimodal_analysis.png

Usage:
  python3 bimodal_detect.py /path/to/results/n20_split-60-40_qwen7b
"""
import os
import sys
import json
import glob

import numpy as np
import matplotlib.pyplot as plt


def two_means_1d(xs, n_iter=50):
    """1D 2-cluster k-means. Returns (cluster_means, labels)."""
    xs = np.asarray(xs, dtype=float)
    c1, c2 = xs.min(), xs.max()
    for _ in range(n_iter):
        d1 = np.abs(xs - c1)
        d2 = np.abs(xs - c2)
        labels = (d2 < d1).astype(int)   # 0 → c1, 1 → c2
        if (labels == 0).sum() == 0 or (labels == 1).sum() == 0:
            break
        new_c1 = xs[labels == 0].mean()
        new_c2 = xs[labels == 1].mean()
        if abs(new_c1 - c1) < 1e-6 and abs(new_c2 - c2) < 1e-6:
            break
        c1, c2 = new_c1, new_c2
    if c1 > c2:
        c1, c2 = c2, c1
        labels = 1 - labels
    return (c1, c2), labels


def bimodality_score(xs, c1, c2):
    """Ratio of inter-cluster gap to intra-cluster spread.
    >1 = clearly bimodal, ~1 = ambiguous, <1 = unimodal."""
    xs = np.asarray(xs, dtype=float)
    gap = abs(c2 - c1)
    spread1 = xs[xs < (c1 + c2) / 2].std() if (xs < (c1 + c2) / 2).sum() > 1 else 0
    spread2 = xs[xs >= (c1 + c2) / 2].std() if (xs >= (c1 + c2) / 2).sum() > 1 else 0
    avg_spread = (spread1 + spread2) / 2
    if avg_spread < 1e-6:
        return float("inf")
    return gap / avg_spread


def load_run(json_path):
    try:
        with open(json_path) as f:
            d = json.load(f)
        return d.get("mean_ms"), d.get("p99_ms")
    except Exception as e:
        print(f"  WARN: failed to parse {json_path}: {e}")
        return None, None


def main(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "run_*.json")))
    if not files:
        print(f"No run_*.json found in {out_dir}")
        sys.exit(1)
    print(f"Loading {len(files)} run files from {out_dir}")

    means, p99s = [], []
    for f in files:
        m, p = load_run(f)
        if m is not None:
            means.append(m)
            p99s.append(p)

    means = np.array(means)
    p99s = np.array(p99s)
    n = len(means)

    if n < 2:
        print(f"Only {n} valid runs — need at least 2"); sys.exit(1)

    print(f"\n=== Stats (N={n}) ===")
    print(f"mean_ms : avg={means.mean():.3f}, std={means.std():.3f}, "
          f"min={means.min():.3f}, max={means.max():.3f}")
    print(f"p99_ms  : avg={p99s.mean():.3f}, std={p99s.std():.3f}")

    # Cluster analysis
    (c1, c2), labels = two_means_1d(means)
    gap = c2 - c1
    score = bimodality_score(means, c1, c2)
    low_frac = (labels == 0).sum() / n
    high_frac = (labels == 1).sum() / n

    print(f"\n=== Cluster analysis ===")
    print(f"LOW  cluster: {c1:.3f} ms (N={int(low_frac*n)}, {low_frac*100:.0f}%)")
    print(f"HIGH cluster: {c2:.3f} ms (N={int(high_frac*n)}, {high_frac*100:.0f}%)")
    print(f"gap           : {gap:.3f} ms")
    print(f"bimodality score: {score:.2f}  "
          f"(>2 = clearly bimodal, ~1 = ambiguous, <0.5 = unimodal)")

    verdict = "BIMODAL" if score > 2 else ("AMBIGUOUS" if score > 1 else "UNIMODAL")
    print(f"VERDICT: {verdict}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # histogram
    ax = axes[0]
    ax.hist(means, bins=max(8, n // 2), color="#2ca02c", alpha=0.7, edgecolor="black")
    ax.axvline(c1, color="green", ls="--", label=f"LOW {c1:.2f}")
    ax.axvline(c2, color="red", ls="--", label=f"HIGH {c2:.2f}")
    ax.set_xlabel("L1 mean (ms)")
    ax.set_ylabel("Run count")
    ax.set_title(f"Distribution of N={n} runs\n"
                 f"gap={gap:.2f} ms, bimodality score={score:.2f} → {verdict}")
    ax.legend()

    # strip
    ax = axes[1]
    colors = ["#2ca02c" if l == 0 else "#d62728" for l in labels]
    ax.scatter(range(1, n + 1), means, c=colors, s=80, edgecolor="black", zorder=5)
    ax.axhline(c1, color="green", ls="--", alpha=0.5)
    ax.axhline(c2, color="red", ls="--", alpha=0.5)
    ax.set_xlabel("Run index")
    ax.set_ylabel("L1 mean (ms)")
    ax.set_title(f"Sequential view — LOW (green) vs HIGH (red)")

    fig.suptitle(f"Bimodal detection: {os.path.basename(out_dir)}", fontweight="bold")
    fig.tight_layout()
    out_png = os.path.join(out_dir, "bimodal_analysis.png")
    fig.savefig(out_png, bbox_inches="tight", dpi=130)
    print(f"\nSaved: {out_png}")

    # Save numeric summary
    out_txt = os.path.join(out_dir, "bimodal_summary.txt")
    with open(out_txt, "w") as f:
        f.write(f"# Bimodal analysis — {out_dir}\n")
        f.write(f"N = {n}\n")
        f.write(f"mean_ms  avg={means.mean():.3f}  std={means.std():.3f}  "
                f"min={means.min():.3f}  max={means.max():.3f}\n")
        f.write(f"p99_ms   avg={p99s.mean():.3f}  std={p99s.std():.3f}\n")
        f.write(f"LOW  cluster:  {c1:.3f} ms  ({low_frac*100:.0f}%)\n")
        f.write(f"HIGH cluster:  {c2:.3f} ms  ({high_frac*100:.0f}%)\n")
        f.write(f"gap = {gap:.3f} ms\n")
        f.write(f"bimodality_score = {score:.2f}\n")
        f.write(f"VERDICT: {verdict}\n")
        f.write("\nRaw means:\n")
        for i, m in enumerate(means, 1):
            f.write(f"  run_{i:02d}  {m:.3f}  ({'HIGH' if labels[i-1] else 'LOW'})\n")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 bimodal_detect.py <results_dir>")
        sys.exit(1)
    main(sys.argv[1])
