#!/usr/bin/env python3.11
"""Small-state screen for PHY rescue, contingent recovery and visible AI.

This is a deliberately conservative *two-stage model*, not a GPU deadline
certificate. All visible AI jobs are known at time zero. Selected NeuralRx
jobs finish at their declared event time; before that, stage-one AI and
conventional-only jobs can use one serial lane, subject to a separately
qualified co-run mode. After outcomes are observed, the remaining
conventional obligations and AI jobs are optimally scheduled on that lane.
Every stage-one choice must admit an all-NeuralRx-fail recovery schedule.
The model assumes independent NeuralRx outcomes for its *expected value*;
the all-fail safety check does not need that assumption.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from offline_recovery_exchange import Job, State, exact_joint, feasible_schedule


@dataclass(frozen=True)
class Cell:
    name: str
    deadline_ms: int
    conventional_ms: int
    nrx_ready_ms: int
    neural_success_probability: float
    incremental_rescue_probability: float

    def __post_init__(self) -> None:
        if not self.name or min(self.deadline_ms, self.conventional_ms,
                                self.nrx_ready_ms) <= 0:
            raise ValueError("invalid cell timing or name")
        if not (0 <= self.incremental_rescue_probability
                <= self.neural_success_probability <= 1):
            raise ValueError("incremental rescue must lie within neural success probability")


@dataclass(frozen=True)
class AiUnit:
    name: str
    deadline_ms: int
    isolated_ms: int
    nrx_overlap_ms: int
    value: int

    def __post_init__(self) -> None:
        if not self.name or min(self.deadline_ms, self.isolated_ms,
                                self.nrx_overlap_ms, self.value) <= 0:
            raise ValueError("invalid AI unit")


@dataclass(frozen=True)
class Problem:
    cells: tuple[Cell, ...]
    ai: tuple[AiUnit, ...]
    min_expected_radio_rescues: float
    nrx_endpoint_capacity: int

    def __post_init__(self) -> None:
        names = [item.name for item in self.cells + self.ai]
        if len(names) != len(set(names)) or not self.cells:
            raise ValueError("all jobs require distinct names and a nonempty radio set")
        if len(self.cells) > 4 or len(self.ai) > 3:
            raise ValueError("screen limited to at most four cells and three AI units")
        if not (0 <= self.min_expected_radio_rescues <= len(self.cells)):
            raise ValueError("invalid radio floor")
        if not (1 <= self.nrx_endpoint_capacity <= len(self.cells)):
            raise ValueError("invalid endpoint capacity")


def recovery_job(cell: Cell, release_ms: int) -> Job | None:
    if release_ms + cell.conventional_ms > cell.deadline_ms:
        return None
    return Job("conv_" + cell.name, release_ms, cell.conventional_ms,
               cell.deadline_ms)


def optimize_for_nrx_subset(problem: Problem, selected_names: frozenset[str]) -> dict | None:
    cells = {cell.name: cell for cell in problem.cells}
    if not selected_names <= cells.keys() or len(selected_names) > problem.nrx_endpoint_capacity:
        return None
    radio_gain = sum(cells[name].incremental_rescue_probability
                     for name in selected_names)
    if radio_gain + 1e-12 < problem.min_expected_radio_rescues:
        return None
    event_ms = max((cells[name].nrx_ready_ms for name in selected_names), default=0)
    if any(event_ms > cells[name].deadline_ms for name in selected_names):
        return None
    unselected = tuple(cell for cell in problem.cells
                       if cell.name not in selected_names)
    best: dict | None = None
    for conv_mask in range(1 << len(unselected)):
        early_conv = frozenset(
            cell.name for index, cell in enumerate(unselected)
            if conv_mask & (1 << index)
        )
        for ai_mask in range(1 << len(problem.ai)):
            early_ai = frozenset(
                unit.name for index, unit in enumerate(problem.ai)
                if ai_mask & (1 << index)
            )
            early_jobs = []
            for cell in unselected:
                if cell.name in early_conv:
                    if cell.conventional_ms > min(cell.deadline_ms, event_ms):
                        break
                    early_jobs.append(Job("conv_" + cell.name, 0,
                                          cell.conventional_ms,
                                          min(cell.deadline_ms, event_ms)))
            else:
                for unit in problem.ai:
                    if unit.name in early_ai:
                        duration = unit.nrx_overlap_ms if selected_names else unit.isolated_ms
                        if duration > min(unit.deadline_ms, event_ms):
                            break
                        early_jobs.append(Job("ai_" + unit.name, 0, duration,
                                              min(unit.deadline_ms, event_ms),
                                              unit.value))
                else:
                    early_schedule = feasible_schedule(0, tuple(early_jobs))
                    if early_schedule is None or (early_schedule and
                                                  early_schedule[-1]["finish_ms"] > event_ms):
                        continue
                    remaining_allfail = []
                    for cell in problem.cells:
                        if cell.name in early_conv:
                            continue
                        job = recovery_job(cell, event_ms)
                        if job is None:
                            break
                        remaining_allfail.append(job)
                    else:
                        certificate = feasible_schedule(event_ms, tuple(remaining_allfail))
                        if certificate is None:
                            continue
                        expectation = float(sum(unit.value for unit in problem.ai
                                                if unit.name in early_ai))
                        branches = []
                        selected = sorted(selected_names)
                        for success_bits in itertools.product((False, True), repeat=len(selected)):
                            outcome = dict(zip(selected, success_bits))
                            probability = 1.0
                            for name, success in outcome.items():
                                p = cells[name].neural_success_probability
                                probability *= p if success else 1 - p
                            recovery = []
                            for cell in problem.cells:
                                if cell.name in early_conv or outcome.get(cell.name, False):
                                    continue
                                job = recovery_job(cell, event_ms)
                                if job is None:
                                    raise AssertionError("all-fail certificate should dominate")
                                recovery.append(job)
                            remaining_ai = tuple(
                                Job("ai_" + unit.name, event_ms, unit.isolated_ms,
                                    unit.deadline_ms, unit.value)
                                for unit in problem.ai
                                if unit.name not in early_ai
                                and event_ms + unit.isolated_ms <= unit.deadline_ms
                            )
                            recourse = exact_joint(State(event_ms, tuple(recovery), remaining_ai))
                            if recourse["value"] is None:
                                raise AssertionError("all-fail certificate should dominate")
                            expectation += probability * recourse["value"]
                            branches.append({"success": outcome,
                                             "probability": probability,
                                             "ai_value_after_observation": recourse["value"]})
                        candidate = {
                            "selected_nrx": selected,
                            "event_ms": event_ms,
                            "radio_gain": radio_gain,
                            "expected_ai_value": expectation,
                            "early_conventional": sorted(early_conv),
                            "early_ai": sorted(early_ai),
                            "early_schedule": early_schedule,
                            "all_fail_recovery_certificate": certificate,
                            "branches": branches,
                        }
                        if best is None or (
                            candidate["expected_ai_value"],
                            candidate["radio_gain"],
                            -len(candidate["selected_nrx"]),
                        ) > (
                            best["expected_ai_value"],
                            best["radio_gain"],
                            -len(best["selected_nrx"]),
                        ):
                            best = candidate
    return best


def all_nrx_subsets(problem: Problem):
    for count in range(problem.nrx_endpoint_capacity + 1):
        for chosen in itertools.combinations((cell.name for cell in problem.cells), count):
            yield frozenset(chosen)


def exact_joint_policy(problem: Problem) -> dict | None:
    candidates = [optimize_for_nrx_subset(problem, subset)
                  for subset in all_nrx_subsets(problem)]
    feasible = [item for item in candidates if item is not None]
    if not feasible:
        return None
    return max(feasible, key=lambda item: (
        item["expected_ai_value"], item["radio_gain"],
        -len(item["selected_nrx"]), tuple(item["selected_nrx"])
    ))


def greedy_radio_then_ai(problem: Problem) -> dict | None:
    """Strong local comparator with the exact safety and recourse evaluator.

    Start from a radio-ranked feasible set, then take any improving add,
    drop, or one-for-one exchange. A positive gap against this comparator
    cannot come from fixed backup timing, lack of AI visibility, or an unsafe
    all-fail check; it still might be a generic local-search gap.
    """
    ranked = sorted(problem.cells, key=lambda cell: (
        -cell.incremental_rescue_probability /
        (cell.nrx_ready_ms + (1 - cell.neural_success_probability) * cell.conventional_ms),
        cell.name,
    ))
    rank = {cell.name: index for index, cell in enumerate(ranked)}
    feasible = [item for item in
                (optimize_for_nrx_subset(problem, subset)
                 for subset in all_nrx_subsets(problem))
                if item is not None]
    if not feasible:
        return None
    # Seed by PHY rank, not test-trace AI outcomes. Safety is exact for every
    # seed. This avoids declaring the comparator infeasible when a lower-
    # ranked but feasible radio subset exists.
    current = min(feasible, key=lambda item: (
        len(item["selected_nrx"]),
        sum(rank[name] for name in item["selected_nrx"]),
        -item["radio_gain"],
    ))
    def score(item: dict) -> tuple:
        # Give the comparator the same tie preference as the exact policy.
        # Otherwise an equal-AI radio improvement would be a weak-baseline
        # artifact rather than a meaningful joint-policy gap.
        return (item["expected_ai_value"], item["radio_gain"],
                -len(item["selected_nrx"]), tuple(item["selected_nrx"]))

    while True:
        selected = frozenset(current["selected_nrx"])
        neighbors = []
        for name in selected:
            neighbors.append(selected - {name})
        for cell in problem.cells:
            if cell.name in selected:
                continue
            if len(selected) < problem.nrx_endpoint_capacity:
                neighbors.append(selected | {cell.name})
            for old in selected:
                neighbors.append((selected - {old}) | {cell.name})
        candidates = [item for item in
                      (optimize_for_nrx_subset(problem, subset)
                       for subset in set(neighbors))
                      if item is not None]
        if not candidates:
            return current
        candidate = max(candidates, key=score)
        if score(candidate) <= score(current):
            return current
        current = candidate


def load_problem(path: Path) -> Problem:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Problem(
        cells=tuple(Cell(**item) for item in data["cells"]),
        ai=tuple(AiUnit(**item) for item in data["ai"]),
        min_expected_radio_rescues=float(data["min_expected_radio_rescues"]),
        nrx_endpoint_capacity=int(data["nrx_endpoint_capacity"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    problem = load_problem(args.state)
    exact = exact_joint_policy(problem)
    greedy = greedy_radio_then_ai(problem)
    report = {
        "schema": "softwall-two-stage-conditional-phy-ai-screen-v1",
        "state": str(args.state),
        "scope": "At most four radio TBs, three visible AI jobs and two stages; hypothetical qualified co-run durations. This is not an online GPU policy or deadline proof.",
        "exact_joint": exact,
        "ai_aware_greedy": greedy,
        "expected_ai_gap": (None if exact is None or greedy is None else
                            exact["expected_ai_value"] - greedy["expected_ai_value"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exact": None if exact is None else exact["expected_ai_value"],
                      "greedy": None if greedy is None else greedy["expected_ai_value"],
                      "gap": report["expected_ai_gap"]}))


if __name__ == "__main__":
    main()
