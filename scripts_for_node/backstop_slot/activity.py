"""Which cells carry a PUSCH TB in which uplink period (partial load).

``activity.prob`` = 1 (default) is the full-load setting: every cell, every period.  Below 1,
each (cell, period) is active independently with that probability (``mode: bernoulli``), or
cells alternate between busy and idle runs with mean busy run ``burst_periods`` and the same
long-run probability (``mode: bursty``).  ``mode: phased`` alternates phases of
``phase_periods`` periods: every (cell, period) active with probability ``phase_high``
(default 1, full load), then with probability ``prob``, and so on.  ``mode: steps`` holds the
activity probability at one of ``levels`` for a random time (0.5-1.5 times ``phase_periods``),
then jumps to a different level, so the load changes at times a policy cannot know in advance.
Every process derives the same mask from the config.
"""

from __future__ import annotations

import numpy as np


def step_levels(spec: dict, periods: int) -> np.ndarray:
    """Activity probability of every period under ``mode: steps``."""
    levels = [float(v) for v in spec.get("levels", (0.25, 0.5, 0.75, 1.0))]
    mean = int(spec.get("phase_periods", 800))
    rng = np.random.default_rng(int(spec.get("seed", 0)) + 7777)
    out = np.empty(periods, dtype=np.float64)
    k, current = 0, int(rng.integers(len(levels)))
    while k < periods:
        run = int(rng.integers(max(1, mean // 2), mean + mean // 2 + 1))
        out[k:k + run] = levels[current]
        k += run
        current = (current + 1 + int(rng.integers(len(levels) - 1))) % len(levels) if len(levels) > 1 else 0
    return out


def activity_mask(config: dict) -> np.ndarray:
    cells, periods = len(config["cells"]), int(config["periods"])
    spec = config.get("activity") or {}
    prob = float(spec.get("prob", 1.0))
    if prob >= 1.0 and spec.get("mode") != "steps":
        return np.ones((cells, periods), dtype=bool)
    rng = np.random.default_rng(int(spec.get("seed", 0)) + 424242)
    if spec.get("mode") == "steps":
        return rng.random((cells, periods)) < step_levels(spec, periods)[None, :]
    if spec.get("mode", "bernoulli") == "bernoulli":
        return rng.random((cells, periods)) < prob
    if spec.get("mode") == "phased":
        draw = rng.random((cells, periods))
        busy = (np.arange(periods) // int(spec.get("phase_periods", 800))) % 2 == 0
        return np.where(busy[None, :], draw < float(spec.get("phase_high", 1.0)), draw < prob)
    busy = float(spec.get("burst_periods", 8.0))
    idle = busy * (1.0 - prob) / prob
    mask = np.zeros((cells, periods), dtype=bool)
    for cell in range(cells):
        on = rng.random() < prob
        k = 0
        while k < periods:
            run = max(1, int(rng.geometric(1.0 / (busy if on else idle))))
            mask[cell, k:k + run] = on
            k += run
            on = not on
    return mask


def gpu_load(config: dict, mask: np.ndarray) -> np.ndarray:
    """Active cells per (GPU, period)."""
    gpus = int(config.get("num_gpus", 4))
    load = np.zeros((gpus, mask.shape[1]), dtype=np.int32)
    for cell in config["cells"]:
        load[int(cell["gpu"])] += mask[int(cell["cell"])]
    return load
