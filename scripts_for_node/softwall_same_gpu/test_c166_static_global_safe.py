#!/usr/bin/env python3.11
"""Unit checks for the C166 static-safe completion semantics."""

from analyze_c166_static_global_safe import static_slot
from c159_q3_oracle_screen import BOUNDS_MS, CONTROL_MS, CUTOFF_MS, RECOVERY_BOUND_MS


def main() -> None:
    assert static_slot(0)["static_global_safe"]["completion_by_context_ms"] == {}
    assert CUTOFF_MS + 4 * RECOVERY_BOUND_MS + CONTROL_MS + min(BOUNDS_MS.values()) == 185
    assert static_slot(1)["decision_ms"] == 225
    print("C166_STATIC_GLOBAL_SAFE_UNIT_PASS")


if __name__ == "__main__":
    main()
