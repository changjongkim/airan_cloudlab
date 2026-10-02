"""Which cells carry a PUSCH TB in which uplink period (partial load).

``activity.prob`` = 1 (default) is the full-load setting: every cell, every period.  Below 1,
each (cell, period) is active independently with that probability (``mode: bernoulli``), or
cells alternate between busy and idle runs with mean busy run ``burst_periods`` and the same
long-run probability (``mode: bursty``).  ``mode: phased`` alternates phases of
``phase_periods`` periods: every (cell, period) active with probability ``phase_high``
(default 1, full load), then with probability ``prob``, and so on.  Every process derives the same mask from the config.
"""

from __future__ import annotations

import numpy as np


def activity_mask(config: dict) -> np.ndarray:
    cells, periods = len(config["cells"]), int(config["periods"])
    spec = config.get("activity") or {}
    prob = float(spec.get("prob", 1.0))
    if prob >= 1.0:
        return np.ones((cells, periods), dtype=bool)
    rng = np.random.default_rng(int(spec.get("seed", 0)) + 424242)
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
