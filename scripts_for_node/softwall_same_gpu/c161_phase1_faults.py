#!/usr/bin/env python3.11
"""Prospectively fixed fault semantics for C161 phase 1."""

from __future__ import annotations


ARMS = ("no_fault", "correlated_all_fail", "latest_start_nonlaunch")


def parse_arm_order(text: str, iterations: int, arm_length: int) -> tuple[str, ...]:
    arms = tuple(value for value in text.split(",") if value)
    if (set(arms) != set(ARMS) or len(arms) != len(ARMS)
            or arm_length <= 0 or iterations != arm_length * len(arms)):
        raise ValueError("invalid C161 phase-1 arm order")
    return arms


def arm_for_index(index: int, arms: tuple[str, ...], arm_length: int) -> str:
    if index < 0 or index >= len(arms) * arm_length:
        raise IndexError("C161 phase-1 epoch outside arm schedule")
    return arms[index // arm_length]


def effective_successes(arm: str, actual_successes) -> tuple[tuple[str, str], ...]:
    values = tuple(tuple(key) for key in actual_successes)
    if arm == "correlated_all_fail":
        return ()
    if arm in {"no_fault", "latest_start_nonlaunch"}:
        return values
    raise ValueError(f"unknown C161 phase-1 arm {arm}")
