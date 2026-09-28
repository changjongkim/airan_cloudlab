#!/usr/bin/env python3.11
"""Focused unit tests for C167 probability and boundary arithmetic."""

import math

from c167_reclaimable_capacity_model import (
    AI_CLASSES,
    CONTROL_MS,
    evaluate_realization,
    lane_failure_distribution,
    lane_potential_counts,
)


def main() -> None:
    for correlation in (0.0, 0.5, 1.0):
        distribution = lane_failure_distribution(4, 2, 1, 0.5, correlation)
        assert math.isclose(sum(probability for _, probability in distribution), 1.0)
    potential = lane_potential_counts(4, 2, 1)
    assert potential == (4,)
    e4 = evaluate_realization(
        potential=potential, failures=(2,), recovery_ms=25,
        window_ms=108, ai_service_ms=AI_CLASSES[256]["service_bound_ms"],
        ai_deadline_ms=1000, offered=1,
    )
    assert e4["static_global_safe"] == 0
    assert e4["softwall"] == 0
    assert e4["debt_blind_current_idle"] == 1
    assert e4["debt_blind_violation"] is True
    e6a = evaluate_realization(
        potential=potential, failures=(1,), recovery_ms=25,
        window_ms=65, ai_service_ms=AI_CLASSES[64]["service_bound_ms"],
        ai_deadline_ms=1000, offered=1,
    )
    assert CONTROL_MS + AI_CLASSES[64]["service_bound_ms"] + 25 == 65
    assert e6a["softwall"] == 1 and not e6a["debt_blind_violation"]
    e6b = evaluate_realization(
        potential=potential, failures=(1,), recovery_ms=25,
        window_ms=64, ai_service_ms=AI_CLASSES[64]["service_bound_ms"],
        ai_deadline_ms=1000, offered=1,
    )
    assert e6b["softwall"] == 0 and e6b["debt_blind_violation"]
    print("C167_RECLAIMABLE_CAPACITY_UNIT_PASS")


if __name__ == "__main__":
    main()
