#!/usr/bin/env python3.11
"""Unit checks for C175 safe greedy and placement upper bound."""

from __future__ import annotations

from c175_lever_evaluation import (
    flexible_recovery_oracle_value,
    greedy_ai_first_value,
    weak_compositions,
)


def main() -> None:
    jobs = ((40.0, 50.0, 64, 0),)
    assert greedy_ai_first_value(jobs, (1,), 25.0) == 64
    assert greedy_ai_first_value(jobs, (4,), 25.0) == 0
    assert set(weak_compositions(2, 2)) == {(0, 2), (1, 1), (2, 0)}
    two_jobs = ((40.0, 50.0, 64, 0), (40.0, 50.0, 64, 1))
    fixed = flexible_recovery_oracle_value(two_jobs, (2, 0), 25.0)
    assert fixed == 128
    print("C175_LEVER_UNIT_PASS")


if __name__ == "__main__":
    main()
