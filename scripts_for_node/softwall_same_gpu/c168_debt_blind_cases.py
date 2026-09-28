#!/usr/bin/env python3.11
"""Prespecified C168 policy/case pairs for physical necessity witnesses."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class DiagnosticScenario:
    scenario_id: str
    case_id: str
    policy: str
    success_count: int
    context_length: int
    target_decision_ms: float
    integer_decision_ms: int
    expected_softwall_lease: bool
    expected_guard_violation: bool
    expected_deadline_violation: bool


SCENARIOS = (
    DiagnosticScenario(
        "E4_softwall_reject", "E4", "softwall", 2, 256,
        45.001, 46, False, False, False,
    ),
    DiagnosticScenario(
        "E4_debt_blind_launch", "E4", "debt_blind", 2, 256,
        45.001, 46, False, True, True,
    ),
    # Continuous targets are frozen inside their integer envelope cells.  They
    # give the padder measurable room on each side of the 88/89 ms frontier.
    DiagnosticScenario(
        "E6a_softwall_admit", "E6a", "softwall", 3, 64,
        87.75, 88, True, False, False,
    ),
    DiagnosticScenario(
        "E6b_softwall_reject", "E6b", "softwall", 3, 64,
        88.75, 89, False, False, False,
    ),
    DiagnosticScenario(
        "E6b_shadow_launch", "E6b", "shadow", 3, 64,
        88.75, 89, False, True, False,
    ),
)


def scenario_for_sequence(sequence: int, reverse: bool = False) -> DiagnosticScenario:
    if sequence <= 0:
        raise ValueError("sequence must be positive")
    rows = tuple(reversed(SCENARIOS)) if reverse else SCENARIOS
    return rows[(sequence - 1) % len(rows)]
