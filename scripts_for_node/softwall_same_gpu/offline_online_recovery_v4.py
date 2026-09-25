#!/usr/bin/env python3.11
"""Event-driven CPU screen for conditional recovery and bounded AI leases.

This module deliberately separates the duration used to *admit* an action
from the sampled duration used to advance a replay.  Every committed AI lease
is checked with ``ai.bound_ms`` and every conventional obligation with
``conv_bound_ms``.  If the replayed action returns earlier, the next event may
reuse that observed slack.  Future AI arrivals are never included in a
current decision.

The model is a policy screen, not a GPU or WCET certificate.  It represents
one four-cell radio release, one NeuralRx observation event, and multiple AI
arrival/completion events on the protected conventional/AI lane exercised by
the observe-first controller.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace


EPS = 1e-9


@dataclass(frozen=True)
class Cell:
    name: str
    p_neural_success: float
    q_incremental_rescue: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("cell name is required")
        if not (0.0 <= self.q_incremental_rescue <= self.p_neural_success <= 1.0):
            raise ValueError("require 0 <= q <= p <= 1")


@dataclass(frozen=True)
class AiJob:
    name: str
    release_ms: float
    deadline_ms: float
    bound_ms: float
    actual_ms: float
    value: int = 1

    def __post_init__(self) -> None:
        if (not self.name or self.release_ms < 0 or self.deadline_ms <= self.release_ms
                or min(self.bound_ms, self.actual_ms, self.value) <= 0
                or self.actual_ms > self.bound_ms + EPS):
            raise ValueError("invalid AI job")


@dataclass(frozen=True)
class Config:
    radio_deadline_ms: float
    nrx_observe_actual_ms: float
    nrx_bound_ms: float
    conv_bound_ms: float
    conv_actual_ms: float
    endpoint_capacity: int

    def __post_init__(self) -> None:
        if min(self.radio_deadline_ms, self.nrx_observe_actual_ms,
               self.nrx_bound_ms, self.conv_bound_ms,
               self.conv_actual_ms, self.endpoint_capacity) <= 0:
            raise ValueError("invalid configuration")
        if self.nrx_observe_actual_ms > self.nrx_bound_ms + EPS:
            raise ValueError("replayed NeuralRx observation exceeds admission bound")
        if self.conv_actual_ms > self.conv_bound_ms + EPS:
            raise ValueError("replayed conventional service exceeds admission bound")


@dataclass(frozen=True)
class BoundJob:
    name: str
    deadline_ms: float
    duration_ms: float
    mandatory: bool
    value: int
    kind: str


def exact_bound_plan(now_ms: float, jobs: tuple[BoundJob, ...]) -> dict | None:
    """Max-value feasible subset containing every mandatory job.

    Jobs are all released by ``now_ms``.  One earliest-finish state per subset
    is sufficient.  Optional jobs precede mandatory jobs in the caller's
    order so equal-value plans expose available AI slack before recovery, as
    in the atomic joint-lease path.  Every policy uses this same evaluator.
    """
    if len(jobs) > 12:
        raise ValueError("screen supports at most twelve visible jobs")
    mandatory_mask = sum((1 << i) for i, job in enumerate(jobs) if job.mandatory)
    full = (1 << len(jobs)) - 1
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (now_ms, ())}
    for mask in range(full + 1):
        state = states.get(mask)
        if state is None:
            continue
        cursor, path = state
        for index, job in enumerate(jobs):
            if mask & (1 << index):
                continue
            finish = cursor + job.duration_ms
            if finish > job.deadline_ms + EPS:
                continue
            next_mask = mask | (1 << index)
            candidate = (finish, path + (index,))
            incumbent = states.get(next_mask)
            if (incumbent is None or candidate[0] < incumbent[0] - EPS
                    or (abs(candidate[0] - incumbent[0]) <= EPS
                        and candidate[1] < incumbent[1])):
                states[next_mask] = candidate

    best = None
    for mask, (finish, path) in states.items():
        if mask & mandatory_mask != mandatory_mask:
            continue
        admitted = tuple(sorted(job.name for i, job in enumerate(jobs)
                                if not job.mandatory and mask & (1 << i)))
        value = sum(job.value for i, job in enumerate(jobs)
                    if not job.mandatory and mask & (1 << i))
        score = (value, len(admitted), -finish, tuple(reversed(admitted)))
        if best is None or score > best[0]:
            best = (score, mask, finish, path, admitted)
    if best is None:
        return None
    _, mask, finish, path, admitted = best
    return {
        "value": sum(job.value for i, job in enumerate(jobs)
                     if not job.mandatory and mask & (1 << i)),
        "admitted": list(admitted),
        "finish_bound_ms": finish,
        "order": [jobs[i].name for i in path],
        "kinds": [jobs[i].kind for i in path],
    }


def _mandatory_jobs(names: set[str], config: Config) -> tuple[BoundJob, ...]:
    return tuple(BoundJob(
        name=f"conv_{name}", deadline_ms=config.radio_deadline_ms,
        duration_ms=config.conv_bound_ms, mandatory=True, value=0,
        kind="conv",
    ) for name in sorted(names))


def _optional_jobs(pending: dict[str, AiJob], now_ms: float) -> tuple[BoundJob, ...]:
    return tuple(BoundJob(
        name=job.name, deadline_ms=job.deadline_ms,
        duration_ms=job.bound_ms, mandatory=False, value=job.value,
        kind="ai",
    ) for job in sorted(pending.values(), key=lambda item: item.name)
             if job.release_ms <= now_ms + EPS and job.deadline_ms > now_ms + EPS)


def _pre_ai_candidate(ai_jobs: tuple[AiJob, ...], cell_names: set[str],
                      config: Config) -> AiJob | None:
    """Choose one visible AI lease whose forced-first all-fail plan is safe."""
    candidates = []
    for job in ai_jobs:
        if job.release_ms > EPS or job.bound_ms > job.deadline_ms + EPS:
            continue
        after = job.bound_ms
        mandatory = _mandatory_jobs(cell_names, config)
        certificate = exact_bound_plan(after, mandatory)
        if certificate is None:
            continue
        candidates.append(job)
    if not candidates:
        return None
    return min(candidates, key=lambda job: (-job.value, job.deadline_ms,
                                           job.bound_ms, job.name))


def replay_branch(cells: tuple[Cell, ...], selected: frozenset[str],
                  outcomes: dict[str, bool], ai_jobs: tuple[AiJob, ...],
                  config: Config, force_all_mandatory: bool = False) -> dict:
    """Replay one observed NeuralRx branch with online, one-action commits."""
    cell_names = {cell.name for cell in cells}
    if not selected <= cell_names or set(outcomes) != set(selected):
        raise ValueError("selected/outcome mismatch")
    if len(selected) > config.endpoint_capacity:
        raise ValueError("endpoint capacity exceeded")

    now = 0.0
    observed = not selected
    mandatory = set(cell_names) if observed else set()
    pending: dict[str, AiJob] = {
        job.name: job for job in ai_jobs if job.release_ms <= EPS
    }
    unseen = sorted((job for job in ai_jobs if job.release_ms > EPS),
                    key=lambda job: (job.release_ms, job.name))
    completed_ai: list[str] = []
    completed_value = 0
    completed_conv: list[str] = []
    actions: list[dict] = []
    certificates: list[dict] = []
    busy: dict | None = None

    if selected:
        candidate = _pre_ai_candidate(ai_jobs, cell_names, config)
        if candidate is not None:
            pending.pop(candidate.name, None)
            busy = {
                "kind": "ai", "name": candidate.name,
                "start_ms": 0.0, "finish_ms": candidate.actual_ms,
                "job": candidate,
            }
            certificates.append({
                "time_ms": 0.0, "action": candidate.name,
                "assumption": "all selected NeuralRx fail or remain unresolved",
                "candidate_bound_finish_ms": candidate.bound_ms,
            })

    def next_external_time() -> float | None:
        values = []
        if not observed:
            values.append(config.nrx_observe_actual_ms)
        if unseen:
            values.append(unseen[0].release_ms)
        if busy is not None:
            values.append(busy["finish_ms"])
        return min(values) if values else None

    while True:
        event_time = next_external_time()
        if event_time is None:
            if not mandatory and not pending:
                break
            event_time = now
        now = max(now, event_time)

        while unseen and unseen[0].release_ms <= now + EPS:
            job = unseen.pop(0)
            pending[job.name] = job

        if not observed and config.nrx_observe_actual_ms <= now + EPS:
            observed = True
            mandatory = {
                name for name in cell_names
                if name not in selected or force_all_mandatory
                or not outcomes.get(name, False)
            }

        if busy is not None and busy["finish_ms"] <= now + EPS:
            action = dict(busy)
            job = action.pop("job", None)
            if action["kind"] == "ai":
                if now > job.deadline_ms + EPS:
                    raise AssertionError("bound-admitted AI missed its deadline")
                completed_ai.append(job.name)
                completed_value += job.value
            else:
                cell_name = action["name"].removeprefix("conv_")
                mandatory.remove(cell_name)
                completed_conv.append(cell_name)
            action["finish_ms"] = now
            actions.append(action)
            busy = None

        if busy is not None:
            continue

        if not observed:
            # The physical observe-first path permits only the already-issued
            # pre-observation lease.  It does not start conventional work or a
            # second AI unit before consuming the NeuralRx result.
            continue

        visible_optional = _optional_jobs(pending, now)
        plan = exact_bound_plan(
            now, visible_optional + _mandatory_jobs(mandatory, config)
        )
        if plan is None:
            return {
                "safe": False, "reason": "mandatory schedule infeasible",
                "time_ms": now, "selected": sorted(selected),
                "outcomes": outcomes,
            }
        if not plan["order"]:
            future = [job.release_ms for job in unseen]
            if future:
                now = min(future)
                continue
            break

        name, kind = plan["order"][0], plan["kinds"][0]
        certificates.append({
            "time_ms": now, "action": name,
            "bound_finish_ms": plan["finish_bound_ms"],
            "admitted_visible_ai": plan["admitted"],
            "mandatory_cells": sorted(mandatory),
        })
        if kind == "ai":
            job = pending.pop(name)
            busy = {
                "kind": "ai", "name": name, "start_ms": now,
                "finish_ms": now + job.actual_ms, "job": job,
            }
        else:
            busy = {
                "kind": "conv", "name": name, "start_ms": now,
                "finish_ms": now + config.conv_actual_ms,
            }

    return {
        "safe": True,
        "selected": sorted(selected),
        "outcomes": dict(sorted(outcomes.items())),
        "completed_ai": completed_ai,
        "completed_ai_value": completed_value,
        "completed_conv": completed_conv,
        "actions": actions,
        "certificates": certificates,
        "finish_ms": now,
    }


def branch_expectation(cells: tuple[Cell, ...], selected: frozenset[str],
                       ai_jobs: tuple[AiJob, ...], config: Config,
                       force_all_mandatory: bool = False) -> dict:
    by_name = {cell.name: cell for cell in cells}
    branches = []
    expectation = 0.0
    for bits in itertools.product((False, True), repeat=len(selected)):
        outcomes = dict(zip(sorted(selected), bits))
        probability = 1.0
        for name, success in outcomes.items():
            p = by_name[name].p_neural_success
            probability *= p if success else 1.0 - p
        replay = replay_branch(cells, selected, outcomes, ai_jobs, config,
                               force_all_mandatory=force_all_mandatory)
        if not replay["safe"]:
            return {"safe": False, "branches": branches}
        expectation += probability * replay["completed_ai_value"]
        branches.append({
            "outcomes": outcomes, "probability": probability,
            "completed_ai_value": replay["completed_ai_value"],
        })
    return {
        "safe": True,
        "expected_ai_value": expectation,
        "branches": branches,
    }


def subsets(cells: tuple[Cell, ...], capacity: int):
    names = tuple(cell.name for cell in cells)
    for count in range(capacity + 1):
        for chosen in itertools.combinations(names, count):
            yield frozenset(chosen)


def evaluate_subset(cells: tuple[Cell, ...], selected: frozenset[str],
                    visible_ai: tuple[AiJob, ...], config: Config,
                    radio_floor: float,
                    force_all_mandatory: bool = False) -> dict | None:
    by_name = {cell.name: cell for cell in cells}
    radio_gain = sum(by_name[name].q_incremental_rescue for name in selected)
    if radio_gain + EPS < radio_floor:
        return None
    result = branch_expectation(cells, selected, visible_ai, config,
                                force_all_mandatory=force_all_mandatory)
    if not result["safe"]:
        return None
    return {
        "selected_nrx": sorted(selected),
        "radio_gain": radio_gain,
        "expected_visible_ai_value": result["expected_ai_value"],
        "branches": result["branches"],
    }


def joint_score(plan: dict) -> tuple:
    return (
        round(plan["expected_visible_ai_value"], 12),
        round(plan["radio_gain"], 12),
        -len(plan["selected_nrx"]),
        tuple(plan["selected_nrx"]),
    )


def feasible_plans(cells: tuple[Cell, ...], visible_ai: tuple[AiJob, ...],
                   config: Config, radio_floor: float,
                   force_all_mandatory: bool = False) -> list[dict]:
    return [plan for plan in (
        evaluate_subset(cells, chosen, visible_ai, config, radio_floor,
                        force_all_mandatory=force_all_mandatory)
        for chosen in subsets(cells, config.endpoint_capacity)
    ) if plan is not None]


def exact_joint_policy(cells: tuple[Cell, ...], visible_ai: tuple[AiJob, ...],
                       config: Config, radio_floor: float,
                       force_all_mandatory: bool = False) -> dict | None:
    plans = feasible_plans(cells, visible_ai, config, radio_floor,
                           force_all_mandatory=force_all_mandatory)
    return max(plans, key=joint_score) if plans else None


def staged_min_policy(cells: tuple[Cell, ...], visible_ai: tuple[AiJob, ...],
                      config: Config, radio_floor: float) -> dict | None:
    plans = feasible_plans(cells, visible_ai, config, radio_floor)
    if not plans:
        return None
    minimum = min(len(plan["selected_nrx"]) for plan in plans)
    return max((plan for plan in plans if len(plan["selected_nrx"]) == minimum),
               key=lambda plan: (round(plan["radio_gain"], 12),
                                 round(plan["expected_visible_ai_value"], 12),
                                 tuple(plan["selected_nrx"])))


def max_radio_policy(cells: tuple[Cell, ...], visible_ai: tuple[AiJob, ...],
                     config: Config, radio_floor: float) -> dict | None:
    plans = feasible_plans(cells, visible_ai, config, radio_floor)
    if not plans:
        return None
    return max(plans, key=lambda plan: (
        round(plan["radio_gain"], 12),
        round(plan["expected_visible_ai_value"], 12),
        -len(plan["selected_nrx"]), tuple(plan["selected_nrx"]),
    ))


def radio_guarded_joint_policy(cells: tuple[Cell, ...],
                               visible_ai: tuple[AiJob, ...], config: Config,
                               radio_floor: float,
                               maximum_radio_loss: float) -> dict | None:
    """Maximize AI inside a fixed distance from the max-radio frontier."""
    if maximum_radio_loss < 0:
        raise ValueError("maximum radio loss must be nonnegative")
    max_radio = max_radio_policy(cells, visible_ai, config, radio_floor)
    if max_radio is None:
        return None
    effective_floor = max(radio_floor,
                          max_radio["radio_gain"] - maximum_radio_loss)
    result = exact_joint_policy(cells, visible_ai, config, effective_floor)
    if result is not None:
        result = dict(result)
        result["max_radio_gain"] = max_radio["radio_gain"]
        result["maximum_radio_loss"] = maximum_radio_loss
        result["effective_radio_floor"] = effective_floor
    return result


def radio_guarded_one_swap_policy(cells: tuple[Cell, ...],
                                  visible_ai: tuple[AiJob, ...],
                                  config: Config, radio_floor: float,
                                  maximum_radio_loss: float) -> dict | None:
    if maximum_radio_loss < 0:
        raise ValueError("maximum radio loss must be nonnegative")
    max_radio = max_radio_policy(cells, visible_ai, config, radio_floor)
    if max_radio is None:
        return None
    effective_floor = max(radio_floor,
                          max_radio["radio_gain"] - maximum_radio_loss)
    result = joint_one_swap_policy(cells, visible_ai, config, effective_floor)
    if result is not None:
        result = dict(result)
        result["max_radio_gain"] = max_radio["radio_gain"]
        result["maximum_radio_loss"] = maximum_radio_loss
        result["effective_radio_floor"] = effective_floor
    return result


def joint_one_swap_policy(cells: tuple[Cell, ...], visible_ai: tuple[AiJob, ...],
                          config: Config, radio_floor: float) -> dict | None:
    current = staged_min_policy(cells, visible_ai, config, radio_floor)
    if current is None:
        return None
    while True:
        chosen = frozenset(current["selected_nrx"])
        neighbors = set()
        for old in chosen:
            neighbors.add(chosen - {old})
        for cell in cells:
            if cell.name in chosen:
                continue
            if len(chosen) < config.endpoint_capacity:
                neighbors.add(chosen | {cell.name})
            for old in chosen:
                neighbors.add((chosen - {old}) | {cell.name})
        candidates = [plan for plan in (
            evaluate_subset(cells, neighbor, visible_ai, config, radio_floor)
            for neighbor in neighbors
        ) if plan is not None]
        if not candidates:
            return current
        best = max(candidates, key=joint_score)
        if joint_score(best) <= joint_score(current):
            return current
        current = best


def uniform_q(cells: tuple[Cell, ...]) -> tuple[Cell, ...]:
    # Preserve a comparable batch-level rescue scale while removing which
    # cell has more predicted incremental radio value.  q must remain <= p.
    value = min(sum(cell.q_incremental_rescue for cell in cells) / len(cells),
                min(cell.p_neural_success for cell in cells))
    return tuple(replace(cell, q_incremental_rescue=value) for cell in cells)
