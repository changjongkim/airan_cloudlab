#!/usr/bin/env python3.11
"""Unit checks for the C174 exact structural scheduler."""

from __future__ import annotations

from c174_trace_oracle_screen import optimal_value


def main() -> None:
    short = ((40.0, 50.0, 64, 0),)
    assert optimal_value("recovery_first", short, (1,), 25.0) == 0
    assert optimal_value("offline_oracle", short, (1,), 25.0) == 64
    relaxed = ((40.0, 100.0, 64, 0),)
    assert optimal_value("recovery_first", relaxed, (1,), 25.0) == 64
    assert optimal_value("offline_oracle", relaxed, (1,), 25.0) == 64
    impossible = ((40.0, 50.0, 64, 0),)
    assert optimal_value("offline_oracle", impossible, (5,), 25.0) == 0
    parallel = ((40.0, 50.0, 64, 0), (40.0, 50.0, 64, 1))
    assert optimal_value("offline_oracle", parallel, (1, 1), 25.0) == 128
    print("C174_TRACE_ORACLE_UNIT_PASS")


if __name__ == "__main__":
    main()
