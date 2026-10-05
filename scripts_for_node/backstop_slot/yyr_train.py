#!/usr/bin/env python3
"""Reliability estimator of the YinYangRAN-style baseline (reconstruction of the paper's design).

YinYangRAN (INFOCOM'24) picks, every decision period, the smallest GPU share for the radio
workload whose predicted reliability meets a target.  The prediction is a neural network that
maps the traffic observed in the previous period (a histogram) and a candidate share to
quantiles of the reliability, trained offline with a quantile loss.

Reconstruction for this testbed:
  state        histogram (``BINS`` bins) of the active share of a GPU's cells, over (GPU, slot)
               pairs of one decision period
  action       AI share (MPS), from the shares of the calibration runs
  reliability  two values per decision period, both against the run without AI of the same seed:
               ``kept``  recovered TBs / recovered TBs without AI
               ``l1``    share of TBs whose conventional result met the L1 deadline
  estimator    MLP (inputs: histogram + share; outputs: quantiles of kept and of l1), trained
               with the quantile (pinball) loss on the decision periods of the calibration runs

usage: yyr_train.py OUT.json JOB:TAG:CELLS[:KIND] [JOB:TAG:CELLS[:KIND] ...]
Calibration runs: steady loads, one tag per load, policies sP (KIND s: fixed share) or pP
(KIND p: fixed share with low-priority AI) and the no-AI run.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from activity import activity_mask
from loss_trace import RAW

BINS = 5
QUANTILES = (0.1, 0.25, 0.5)
DECISION = 400               # uplink periods per decision period (1 s)


def per_period(path: Path):
    result = json.loads(path.read_text())
    config = result["config"]
    work = Path(str(path)[:-5] + "_work")
    periods = int(config["periods"])
    rescue_ns = float(config["rescue_deadline_ms"]) * 1e6
    deadline_ns = float(config["deadline_ms"]) * 1e6
    lanes = {}
    for lane in work.glob("lane*.json"):
        for r in json.loads(lane.read_text())["records"]:
            lanes[(r[0], r[1])] = r
    recovered, late, tbs = np.zeros(periods), np.zeros(periods), np.zeros(periods)
    for cell in config["cells"]:
        index, ues = int(cell["cell"]), int(cell.get("num_ue", 1))
        for r in json.loads((work / f"conv{index}.json").read_text())["records"]:
            period, release, done = r[0], r[2], r[4]
            tbs[period] += ues
            late[period] += ues * int(done - release > deadline_ns)
            if cell.get("nrx_gpu") is None:
                continue
            conv_good = r[9] if ues > 1 else r[6]
            ran = lanes.get((index, period))
            if ran is not None and ran[3] - release <= rescue_ns:
                good = ran[7] if ues > 1 else ran[5]
                recovered[period] += bin(good & ~conv_good & ((1 << ues) - 1)).count("1")
    mask = activity_mask(config)
    gpus = int(config.get("num_gpus", 4))
    load = np.zeros((gpus, periods))
    own = np.zeros(gpus)
    for cell in config["cells"]:
        load[int(cell["gpu"])] += mask[int(cell["cell"])]
        own[int(cell["gpu"])] += 1
    return config, recovered, late, tbs, load / own[:, None]


def histogram(share_active: np.ndarray) -> np.ndarray:
    """Share of (GPU, slot) pairs in each bin of the active share of the GPU's cells."""
    bins = np.minimum(BINS - 1, (BINS * share_active).astype(int))
    return np.bincount(bins.ravel(), minlength=BINS) / bins.size


def samples(specs: list[str]):
    x, y = [], []
    for spec in specs:
        job, tag, cells, kind = (spec.split(":") + ["sp"])[:4]
        pattern = re.compile(rf"^{re.escape(tag)}(\d)(?:n|r(\d+)([{kind}])(\d+))_c{cells}_")
        refs, runs = {}, []
        for path in sorted(RAW.glob(f"{tag}[0-9]*_c{cells}_*_j{job}.json")):
            m = pattern.match(path.name)
            if not m:
                continue
            if m.group(2) is None:
                refs[int(m.group(1))] = per_period(path)
            else:
                runs.append((int(m.group(1)), int(m.group(4)), path))
        for seed, share, path in runs:
            if seed not in refs:
                continue
            _, rec0, _, _, _ = refs[seed]
            config, rec, late, tbs, load = per_period(path)
            for start in range(DECISION, int(config["periods"]) - DECISION + 1, DECISION):
                span = slice(start, start + DECISION)
                if rec0[span].sum() < 5:
                    continue
                x.append(np.append(histogram(load[:, span]), share / 100.0))
                y.append([rec[span].sum() / rec0[span].sum(), 1.0 - late[span].sum() / max(1.0, tbs[span].sum())])
    return np.array(x), np.array(y)


def train(x: np.ndarray, y: np.ndarray, hidden: int = 16, steps: int = 30000, seed: int = 0):
    """MLP with one hidden layer; outputs = quantiles of kept, then quantiles of l1."""
    rng = np.random.default_rng(seed)
    nq = len(QUANTILES)
    w1 = rng.normal(0, 0.5, (hidden, x.shape[1]))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, 0.1, (2 * nq, hidden))
    b2 = np.concatenate([np.full(nq, np.median(y[:, 0])), np.full(nq, np.median(y[:, 1]))])
    target = np.concatenate([np.repeat(y[:, :1], nq, axis=1), np.repeat(y[:, 1:], nq, axis=1)], axis=1)
    taus = np.array(QUANTILES * 2)
    scale = np.array([1.0] * nq + [50.0] * nq)        # l1 varies in the fourth decimal: weigh it up
    m = [np.zeros_like(p) for p in (w1, b1, w2, b2)]
    v = [np.zeros_like(p) for p in (w1, b1, w2, b2)]
    for step in range(1, steps + 1):
        h = np.maximum(0.0, x @ w1.T + b1)
        out = h @ w2.T + b2
        u = target - out
        grad_out = -np.where(u > 0, taus, taus - 1.0) * scale / len(x)       # pinball loss
        gw2, gb2 = grad_out.T @ h, grad_out.sum(axis=0)
        gh = (grad_out @ w2) * (h > 0)
        gw1, gb1 = gh.T @ x, gh.sum(axis=0)
        for i, (p, g) in enumerate(zip((w1, b1, w2, b2), (gw1, gb1, gw2, gb2))):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            # Step size from 1e-2 down to 1e-6: the L1 share is resolved in the fifth decimal.
            rate = 1e-2 * 1e-4 ** (step / steps)
            p -= rate * (m[i] / (1 - 0.9 ** step)) / (np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8)
    return [[w1.tolist(), b1.tolist()], [w2.tolist(), b2.tolist()]]


def predict(layers, x: np.ndarray) -> np.ndarray:
    (w1, b1), (w2, b2) = layers
    return np.maximum(0.0, x @ np.array(w1).T + np.array(b1)) @ np.array(w2).T + np.array(b2)


def main() -> None:
    out_path = Path(sys.argv[1])
    x, y = samples(sys.argv[2:])
    layers = train(x, y)
    nq = len(QUANTILES)
    estimator = {"bins": BINS, "quantiles": list(QUANTILES), "decision_periods": DECISION,
                 "outputs": {"kept": list(range(nq)), "l1": list(range(nq, 2 * nq))},
                 "layers": layers, "samples": int(len(x))}
    out_path.write_text(json.dumps(estimator))
    print(f"samples {len(x)} (decision periods), shares {sorted(set((x[:, -1] * 100).round().astype(int)))}")
    print("| mean active share | AI share | kept: measured median (q25) | kept: predicted q10 / q25 / q50 | l1: measured median (q25) | l1: predicted q10 / q25 |")
    print("|---|---|---|---|---|---|")
    level = (x[:, :BINS] * (np.arange(BINS) + 0.5) / BINS).sum(axis=1)
    for lv in sorted(set(level.round(1))):
        for share in sorted(set(x[:, -1].round(2))):
            pick = (level.round(1) == lv) & (x[:, -1].round(2) == share)
            if not pick.any():
                continue
            p = predict(layers, x[pick]).mean(axis=0)
            print(f"| {lv:.1f} | {100 * share:.0f}% | {np.median(y[pick, 0]):.3f} (q25 {np.quantile(y[pick, 0], 0.25):.3f}) | "
                  f"{p[0]:.3f} / {p[1]:.3f} / {p[2]:.3f} | "
                  f"{np.median(y[pick, 1]):.5f} (q25 {np.quantile(y[pick, 1], 0.25):.5f}) | {p[nq]:.5f} / {p[nq + 1]:.5f} |")


if __name__ == "__main__":
    main()
