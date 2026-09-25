#!/usr/bin/env python3.11
"""Prespecified physical boundary cases for C162."""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class BoundaryCase:
    case_id: str
    success_count: int
    context_length: Optional[int]
    target_decision_ms: int
    expected_state: str
    expected_lease: bool


CASES = (
    BoundaryCase("E1_four_unresolved_no_ai", 0, None, 45, "QSN", False),
    BoundaryCase("E3_two_unresolved_context128", 2, 128, 45, "QSU", True),
    BoundaryCase("E4_two_unresolved_context256", 2, 256, 45, "QSN", False),
    BoundaryCase("E5_one_unresolved_context512", 3, 512, 45, "QSU", True),
    # wait_until(87 ms) normally yields ceil(observed_ms)=88, the latest
    # integer-ms safe point.  The analyzer uses the observed clock, not target.
    BoundaryCase("E6a_latest_safe_context64", 3, 64, 87, "QSU", True),
    BoundaryCase("E6b_first_unsafe_context64", 3, 64, 88, "QSN", False),
)


CASE_BY_ID = {case.case_id: case for case in CASES}


def case_for_sequence(sequence: int, reverse: bool = False) -> BoundaryCase:
    if sequence <= 0:
        raise ValueError("sequence must be positive")
    cases = tuple(reversed(CASES)) if reverse else CASES
    return cases[(sequence - 1) % len(cases)]

