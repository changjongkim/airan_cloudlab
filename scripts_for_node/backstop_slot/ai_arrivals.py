"""AI request arrivals with BurstGPT prompt lengths (no GPU imports).

Inter-arrival times are exponential (Poisson) by default; ``arrival_cv`` > 1 draws them from a
gamma distribution with that coefficient of variation and the same mean (bursty arrivals).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def arrivals(ai: dict, gpu: int, horizon_ns: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(int(ai["seed"]) + 1000 * gpu)
    trace = json.loads(Path(ai["trace"]).read_text())
    lengths = [max(1, min(int(r["raw_request_tokens"]), int(ai["max_prompt"])))
               for r in trace["requests"]]
    rate = float(ai["rate_per_s"])
    cv = float(ai.get("arrival_cv", 1.0))
    out, t = [], 0.0
    while True:
        t += rng.exponential(1.0 / rate) if cv == 1.0 else rng.gamma(1.0 / cv ** 2, cv ** 2 / rate)
        if t * 1e9 >= horizon_ns:
            return out
        out.append((int(t * 1e9), int(lengths[rng.integers(len(lengths))])))
