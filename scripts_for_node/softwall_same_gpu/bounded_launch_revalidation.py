#!/usr/bin/env python3.11
"""Repeat launch-time certificate construction after a control-budget overrun."""

from __future__ import annotations

import time
from collections.abc import Callable


def bounded_revalidate(
    builder: Callable[[int], dict],
    *,
    release_ns: int,
    lower_bound_ns: int,
    budget_ms: float,
    max_attempts: int = 3,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> tuple[dict, int, int, list[dict]]:
    if budget_ms <= 0 or max_attempts <= 0:
        raise ValueError("budget and attempts must be positive")
    budget_ns = round(budget_ms * 1e6)
    attempts = []
    for attempt in range(1, max_attempts + 1):
        started_ns = clock()
        launch_now_ns = max(lower_bound_ns, started_ns - release_ns)
        plan = builder(launch_now_ns)
        completed_ns = clock()
        elapsed_ns = completed_ns - started_ns
        attempts.append({
            "attempt": attempt,
            "started_ns": started_ns,
            "completed_ns": completed_ns,
            "launch_now_ns": launch_now_ns,
            "elapsed_ms": elapsed_ns / 1e6,
            "within_budget": elapsed_ns <= budget_ns,
        })
        if elapsed_ns <= budget_ns:
            return plan, started_ns, completed_ns, attempts
    raise RuntimeError(
        f"launch revalidation exceeded {budget_ms} ms for {max_attempts} attempts"
    )

