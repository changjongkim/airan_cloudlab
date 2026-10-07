#!/usr/bin/env python3
"""Channel states of a two-user cell under closed-loop link adaptation.

With ``la["datasets"]`` in the run configuration a cell holds one slot ring per (state, MCS level): every state is
a dataset of the same channel at another Es/No.  The cell stays ``la["phase"]`` uplink periods in a state and
then moves to another state chosen at random; the times of the changes differ between the cells.  The outer loop
keeps its pointer over the MCS levels across a change, as a rate controller that does not know the channel.
With ``la["phase"]`` = 0 the states are fixed: every cell stays in the state ``la["assign"][cell]`` (cells of
different channels on one server).

Rings and records use one flat index: state * (number of MCS levels) + MCS level.  With one state the flat index
is the MCS level, as before.
"""

from __future__ import annotations

import numpy as np


def state_count(la: dict | None) -> int:
    return max(1, len(la.get("datasets", []))) if la else 1


def state_sequence(la: dict, cell: int, periods: int) -> np.ndarray:
    """State of every uplink period of a cell."""
    states = state_count(la)
    if states == 1:
        return np.zeros(periods, dtype=np.int64)
    phase = int(la["phase"])
    if phase <= 0:
        return np.full(periods, int(la["assign"][str(int(cell))]), dtype=np.int64)
    rng = np.random.default_rng(int(la.get("state_seed", 0)) * 1000 + int(cell))
    offset = int(rng.integers(phase))
    seq = np.empty(periods // phase + 2, dtype=np.int64)
    state = int(rng.integers(states))
    for i in range(len(seq)):
        seq[i] = state
        state = (state + 1 + int(rng.integers(states - 1))) % states       # a different state
    return np.repeat(seq, phase)[offset:offset + periods]
