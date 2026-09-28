#!/usr/bin/env python3.11
"""Regression checks for absolute trace-deadline semantics."""

from __future__ import annotations

from analyze_c173_deadline_correction import exact_weighted_matching_absolute_deadline


def main() -> None:
    requests = [{
        "context_length": 64,
        "arrival_ms": 100.0,
        "deadline_ms": 150.0,
        "value_tokens": 64,
    }]
    late_slot = [{
        "decision_ms": 120.0,
        "policy": {"completion_by_context_ms": {64: 151.0}},
    }]
    timely_slot = [{
        "decision_ms": 120.0,
        "policy": {"completion_by_context_ms": {64: 150.0}},
    }]
    assert exact_weighted_matching_absolute_deadline(requests, late_slot, "policy")["timely_requests"] == 0
    assert exact_weighted_matching_absolute_deadline(requests, timely_slot, "policy")["timely_requests"] == 1
    print("C173_ABSOLUTE_DEADLINE_UNIT_PASS")


if __name__ == "__main__":
    main()
