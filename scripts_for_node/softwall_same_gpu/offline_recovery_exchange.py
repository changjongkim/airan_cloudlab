#!/usr/bin/env python3.11
"""Small-state decision gate for conditional recovery-credit exchange.

All jobs in a DecisionState are observable at ``now_ms``. Recovery jobs are
mandatory, representing simultaneous failure of every unresolved NeuralRx.
The model is a *single qualified serial lane* after physical NRx completion;
it does not model MPS overlap, live GPU kernels, radio quality, or future AI.
It is an offline policy screen, not a physical deadline certificate.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Job:
    name: str
    release_ms: int
    duration_ms: int
    deadline_ms: int
    value: int = 0
    fixed_start_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.name or self.duration_ms <= 0 or self.release_ms > self.deadline_ms:
            raise ValueError(f"invalid job {self.name!r}")
        if self.value < 0:
            raise ValueError("negative AI value")
        if self.fixed_start_ms is not None and self.fixed_start_ms < self.release_ms:
            raise ValueError("fixed start precedes release")


@dataclass(frozen=True)
class State:
    now_ms: int
    recovery: tuple[Job, ...]
    ai: tuple[Job, ...]

    def __post_init__(self) -> None:
        names = [job.name for job in self.recovery + self.ai]
        if len(names) != len(set(names)):
            raise ValueError("job names must be distinct")
        if any(job.release_ms > self.now_ms for job in self.recovery + self.ai):
            raise ValueError("state contains a future arrival")
        if any(job.value != 0 for job in self.recovery):
            raise ValueError("recovery value must be zero")


def _schedule_order(now_ms: int, order: Iterable[Job]) -> list[dict] | None:
    cursor = now_ms
    rows: list[dict] = []
    for job in order:
        start = max(cursor, job.release_ms)
        if job.fixed_start_ms is not None:
            if start > job.fixed_start_ms:
                return None
            start = job.fixed_start_ms
        finish = start + job.duration_ms
        if finish > job.deadline_ms:
            return None
        rows.append({"job": job.name, "start_ms": start, "finish_ms": finish})
        cursor = finish
    return rows


def feasible_schedule(now_ms: int, jobs: tuple[Job, ...]) -> list[dict] | None:
    """Exact permutation search for at most eight visible non-preemptive jobs."""
    if len(jobs) > 8:
        raise ValueError("exact screen supports at most eight jobs")
    if not jobs:
        return []
    best: list[dict] | None = None
    for order in itertools.permutations(jobs):
        rows = _schedule_order(now_ms, order)
        if rows is None:
            continue
        if best is None or (rows[-1]["finish_ms"], [x["job"] for x in rows]) < (
            best[-1]["finish_ms"], [x["job"] for x in best]
        ):
            best = rows
    return best


def exact_joint(state: State) -> dict:
    """Maximize completed visible AI value subject to every recovery job."""
    best = {"value": -1, "admitted": [], "schedule": None}
    for mask in range(1 << len(state.ai)):
        selected = tuple(job for index, job in enumerate(state.ai) if mask & (1 << index))
        score = sum(job.value for job in selected)
        if score < best["value"]:
            continue
        rows = feasible_schedule(state.now_ms, state.recovery + selected)
        names = sorted(job.name for job in selected)
        if rows is not None and (score > best["value"] or names < best["admitted"]):
            best = {"value": score, "admitted": names, "schedule": rows}
    if best["value"] < 0:
        return {"value": None, "admitted": [], "schedule": None}
    return best


def greedy_retime(state: State, priority: str) -> dict:
    """AI-aware baseline: movable recovery and exact feasibility per insertion."""
    if priority == "density":
        ranked = sorted(state.ai, key=lambda j: (-j.value / j.duration_ms, j.deadline_ms, j.name))
    elif priority == "deadline":
        ranked = sorted(state.ai, key=lambda j: (j.deadline_ms, -j.value, j.name))
    elif priority == "value":
        ranked = sorted(state.ai, key=lambda j: (-j.value, j.deadline_ms, j.name))
    else:
        raise ValueError(f"unknown greedy priority {priority}")
    selected: tuple[Job, ...] = ()
    rows = feasible_schedule(state.now_ms, state.recovery)
    if rows is None:
        return {"value": None, "admitted": [], "schedule": None}
    for candidate in ranked:
        trial = feasible_schedule(state.now_ms, state.recovery + selected + (candidate,))
        if trial is not None:
            selected += (candidate,)
            rows = trial
    return {
        "value": sum(job.value for job in selected),
        "admitted": sorted(job.name for job in selected),
        "schedule": rows,
    }


def fixed_recovery(state: State) -> dict:
    """Fixed-calendar baseline with the same AI-aware insertion safety check."""
    if any(job.fixed_start_ms is None for job in state.recovery):
        raise ValueError("fixed baseline requires all recovery starts")
    return greedy_retime(state, "deadline")


def load_state(path: Path) -> State:
    source = json.loads(path.read_text(encoding="utf-8"))
    return State(
        now_ms=int(source["now_ms"]),
        recovery=tuple(Job(**item) for item in source["recovery"]),
        ai=tuple(Job(**item) for item in source["ai"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = load_state(args.state)
    fixed = fixed_recovery(state)
    movable = State(state.now_ms, tuple(Job(
        job.name, job.release_ms, job.duration_ms, job.deadline_ms, job.value
    ) for job in state.recovery), state.ai)
    report = {
        "schema": "softwall-offline-decision-screen-v1",
        "state": str(args.state),
        "scope": "visible jobs after physical NRx completion; one serial lane; no future AI",
        "fixed": fixed,
        "greedy_density": greedy_retime(movable, "density"),
        "greedy_deadline": greedy_retime(movable, "deadline"),
        "greedy_value": greedy_retime(movable, "value"),
        "joint_exact": exact_joint(movable),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value["value"] for key, value in report.items() if isinstance(value, dict)}))


if __name__ == "__main__":
    main()
