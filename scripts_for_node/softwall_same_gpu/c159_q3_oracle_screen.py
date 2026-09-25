#!/usr/bin/env python3.11
"""C159-Q3 calibration-only offline upper-bound screen.

The two policies receive the same radio outcomes, AI requests, class bounds,
and an exact offline request selector.  They differ only in execution order:
SoftWall may atomically place AI before unresolved recoveries, whereas the
strong event-driven baseline drains the unresolved recoveries before AI.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


CONTEXTS = (16, 32, 64, 128, 256, 512)
BOUNDS_MS = {16: 35.0, 32: 35.0, 64: 35.0, 128: 40.0, 256: 65.0, 512: 75.0}
PERIOD_MS = 180.0
CUTOFF_MS = 45.0
RECOVERY_DEADLINE_MS = 153.0
RECOVERY_BOUND_MS = 25.0
CONTROL_MS = 5.0
WINDOW_MS = 60_000.0


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Edge:
    to: int
    reverse: int
    capacity: int
    cost: int


def add_edge(graph: list[list[Edge]], left: int, right: int,
             capacity: int, cost: int) -> Edge:
    forward = Edge(right, len(graph[right]), capacity, cost)
    backward = Edge(left, len(graph[left]), 0, -cost)
    graph[left].append(forward)
    graph[right].append(backward)
    return forward


def exact_weighted_matching(requests: list[dict], slots: list[dict],
                            policy: str) -> dict:
    """Maximum timely token value for the supplied policy-specific slots."""
    job_count = len(requests)
    slot_count = len(slots)
    source = 0
    first_job = 1
    first_slot = first_job + job_count
    sink = first_slot + slot_count
    graph: list[list[Edge]] = [[] for _ in range(sink + 1)]
    edge_refs: list[tuple[int, int, Edge]] = []
    for index in range(job_count):
        add_edge(graph, source, first_job + index, 1, 0)
    for index in range(slot_count):
        add_edge(graph, first_slot + index, sink, 1, 0)
    for job_index, request in enumerate(requests):
        context = int(request["context_length"])
        arrival = float(request["arrival_ms"])
        deadline = arrival + float(request["deadline_ms"])
        for slot_index, slot in enumerate(slots):
            completion = slot[policy]["completion_by_context_ms"].get(context)
            if completion is None:
                continue
            if arrival <= slot["decision_ms"] and completion <= deadline:
                edge = add_edge(
                    graph, first_job + job_index, first_slot + slot_index,
                    1, -int(request["value_tokens"]),
                )
                edge_refs.append((job_index, slot_index, edge))

    flow = 0
    total_cost = 0
    infinity = 10**18
    while True:
        distance = [infinity] * len(graph)
        previous_node = [-1] * len(graph)
        previous_edge = [-1] * len(graph)
        in_queue = [False] * len(graph)
        distance[source] = 0
        queue = collections.deque([source])
        in_queue[source] = True
        while queue:
            node = queue.popleft()
            in_queue[node] = False
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                candidate = distance[node] + edge.cost
                if candidate >= distance[edge.to]:
                    continue
                distance[edge.to] = candidate
                previous_node[edge.to] = node
                previous_edge[edge.to] = edge_index
                if not in_queue[edge.to]:
                    queue.append(edge.to)
                    in_queue[edge.to] = True
        if distance[sink] == infinity or distance[sink] >= 0:
            break
        node = sink
        while node != source:
            parent = previous_node[node]
            edge = graph[parent][previous_edge[node]]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent
        flow += 1
        total_cost += distance[sink]

    matched = [
        (job_index, slot_index) for job_index, slot_index, edge in edge_refs
        if edge.capacity == 0
    ]
    by_context = {context: 0 for context in CONTEXTS}
    for job_index, _ in matched:
        by_context[int(requests[job_index]["context_length"])] += 1
    return {
        "timely_value_tokens": -total_cost,
        "timely_requests": flow,
        "timely_by_context": {str(key): value for key, value in by_context.items()},
    }


def build_state_rows(coordinator: dict) -> list[dict]:
    recovery_ms = collections.defaultdict(float)
    for row in coordinator["physical_recoveries"]:
        recovery_ms[int(row["sequence"])] += (
            int(row["actual_completed_ns"]) - int(row["actual_start_ns"])
        ) / 1e6
    return [
        {
            "sequence": int(row["sequence"]),
            "unresolved": len(row["physical_recoveries"]),
            "success_keys": row["success_keys"],
            "empirical_recovery_ms": recovery_ms[int(row["sequence"])],
        }
        for row in coordinator["rounds"]
    ]


def policy_slot(epoch: int, state: dict) -> dict:
    base = epoch * PERIOD_MS
    decision = base + CUTOFF_MS
    unresolved = int(state["unresolved"])
    admission = {
        context: (
            CUTOFF_MS + unresolved * RECOVERY_BOUND_MS
            + CONTROL_MS + BOUNDS_MS[context]
            <= RECOVERY_DEADLINE_MS
        )
        for context in CONTEXTS
    }
    softwall = {
        context: decision + CONTROL_MS + BOUNDS_MS[context]
        for context in CONTEXTS if admission[context]
    }
    event_empirical = {
        context: (
            decision + float(state["empirical_recovery_ms"])
            + CONTROL_MS + BOUNDS_MS[context]
        )
        for context in CONTEXTS if admission[context]
    }
    event_contract = {
        context: (
            decision + unresolved * RECOVERY_BOUND_MS
            + CONTROL_MS + BOUNDS_MS[context]
        )
        for context in CONTEXTS if admission[context]
    }
    return {
        "epoch": epoch,
        "decision_ms": decision,
        "unresolved": unresolved,
        "empirical_recovery_ms": state["empirical_recovery_ms"],
        "softwall": {"completion_by_context_ms": softwall},
        "event_empirical": {"completion_by_context_ms": event_empirical},
        "event_contract": {"completion_by_context_ms": event_contract},
    }


def relative_gain(candidate: int, baseline: int) -> float | None:
    return 100.0 * (candidate - baseline) / baseline if baseline else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--prespec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    trace_path = root / "data/current/softwall_c159_calibration_burst_windows_v1.json"
    combined_path = root / "results/softwall_multigpu/confirm159_q2_variable_two_node.json"
    raw = root / "results/softwall_multigpu/raw"
    dev_path = raw / "confirm159q2f_batch_dev_j58857672_coordinator.json"
    holdout_path = raw / "confirm159q2g_batch_holdout_j58858194_coordinator.json"
    trace = load(trace_path)
    combined = load(combined_path)
    prespec = load(args.prespec)
    sources = {
        "trace": trace_path,
        "q2_combined": combined_path,
        "q2_development_coordinator": dev_path,
        "q2_holdout_coordinator": holdout_path,
        "screen_source": Path(__file__).resolve(),
    }
    observed_hashes = {
        key: sha256(path) for key, path in sources.items()
    }
    if observed_hashes != prespec["source_sha256"]:
        raise RuntimeError("Q3 source hash mismatch")
    state_streams = {
        "development": build_state_rows(load(dev_path)),
        "holdout": build_state_rows(load(holdout_path)),
    }
    mappings = (
        ("development", 0),
        ("development", 266),
        ("holdout", 0),
        ("holdout", 266),
    )
    rows = []
    for window, (stream_name, offset) in zip(trace["windows"], mappings):
        states = state_streams[stream_name]
        slot_count = sum(
            epoch * PERIOD_MS + CUTOFF_MS < WINDOW_MS
            for epoch in range(1000)
        )
        selected_states = states[offset:offset + slot_count]
        if len(selected_states) != slot_count:
            raise RuntimeError("insufficient frozen Q2 state stream")
        slots = [policy_slot(epoch, state)
                 for epoch, state in enumerate(selected_states)]
        policy_results = {
            policy: exact_weighted_matching(window["requests"], slots, policy)
            for policy in ("softwall", "event_empirical", "event_contract")
        }
        soft = policy_results["softwall"]["timely_value_tokens"]
        empirical = policy_results["event_empirical"]["timely_value_tokens"]
        contract = policy_results["event_contract"]["timely_value_tokens"]
        rows.append({
            "window_id": window["window_id"],
            "q2_state_stream": stream_name,
            "q2_state_offset": offset,
            "slots": slot_count,
            "offered_requests": window["summary"]["requests"],
            "offered_value_tokens": window["summary"]["offered_value_tokens"],
            "policies": policy_results,
            "softwall_gain_over_event_empirical_pct": relative_gain(soft, empirical),
            "softwall_gain_over_event_contract_pct": relative_gain(soft, contract),
        })
    totals = {
        policy: {
            "timely_value_tokens": sum(
                row["policies"][policy]["timely_value_tokens"] for row in rows
            ),
            "timely_requests": sum(
                row["policies"][policy]["timely_requests"] for row in rows
            ),
        }
        for policy in ("softwall", "event_empirical", "event_contract")
    }
    soft_total = totals["softwall"]["timely_value_tokens"]
    empirical_total = totals["event_empirical"]["timely_value_tokens"]
    contract_total = totals["event_contract"]["timely_value_tokens"]
    empirical_gain = relative_gain(soft_total, empirical_total)
    contract_gain = relative_gain(soft_total, contract_total)
    per_window_empirical = [
        row["softwall_gain_over_event_empirical_pct"] for row in rows
    ]
    mde = float(prespec["minimum_effect_pct"])
    gates = {
        "q2_variable_context_qualification_passed": combined.get("all_pass") is True,
        "calibration_only_and_holdout_unmaterialized": (
            trace["partition"]["name"] == "calibration"
            and prespec["confirmatory_holdout_materialized"] is False
        ),
        "frozen_source_hash_match": observed_hashes == prespec["source_sha256"],
        "all_four_calibration_windows_complete": len(rows) == 4,
        "same_exact_offline_selector_both_policies": True,
        "radio_admission_set_equal_by_construction": True,
        "empirical_upper_bound_reaches_mde": (
            empirical_gain is not None and empirical_gain >= mde
        ),
        "empirical_direction_positive_all_windows": all(
            value is not None and value > 0 for value in per_window_empirical
        ),
    }
    open_holdout = (
        all(gates[key] for key in (
            "q2_variable_context_qualification_passed",
            "calibration_only_and_holdout_unmaterialized",
            "frozen_source_hash_match",
            "all_four_calibration_windows_complete",
            "same_exact_offline_selector_both_policies",
            "radio_admission_set_equal_by_construction",
            "empirical_upper_bound_reaches_mde",
            "empirical_direction_positive_all_windows",
        ))
    )
    value = {
        "schema": "softwall-confirm159-q3-oracle-screen-v1",
        "status": (
            "C159_Q3_ORACLE_SCREEN_OPEN_HOLDOUT"
            if open_holdout else "C159_Q3_ORACLE_SCREEN_STOP_PERFORMANCE"
        ),
        "open_confirmatory_holdout": open_holdout,
        "gates": gates,
        "model": {
            "period_ms": PERIOD_MS,
            "cutoff_ms": CUTOFF_MS,
            "recovery_deadline_ms": RECOVERY_DEADLINE_MS,
            "recovery_bound_ms": RECOVERY_BOUND_MS,
            "launch_control_ms": CONTROL_MS,
            "class_bounds_ms": {str(k): v for k, v in BOUNDS_MS.items()},
            "capacity": "at most one Qwen unit per P180 radio epoch",
            "softwall_order": "AI first, then atomically retimed unresolved recoveries",
            "event_order": "physically drain unresolved recoveries, then admit AI",
            "event_empirical_delay": (
                "sum of Q2 observed full recovery path durations in that epoch"
            ),
            "event_contract_sensitivity": (
                "25 ms per unresolved recovery; intentionally conservative"
            ),
        },
        "windows": rows,
        "summary": {
            "offered_requests": trace["summary"]["selected_requests"],
            "offered_value_tokens": trace["summary"]["offered_value_tokens"],
            "slots": sum(row["slots"] for row in rows),
            "totals": totals,
            "softwall_gain_over_event_empirical_pct": empirical_gain,
            "softwall_gain_over_event_contract_pct": contract_gain,
            "minimum_effect_pct": mde,
            "per_window_empirical_gain_pct": per_window_empirical,
        },
        "interpretation": (
            "Calibration-only structural upper-bound screen. Both policies use "
            "the same exact offline weighted request selector and the same radio "
            "admission inequality. The measured difference can only come from "
            "SoftWall placing AI before unresolved recovery versus the strong "
            "event-driven baseline physically draining recovery first. This is "
            "not an online performance result or confirmatory confidence interval."
        ),
        "source_sha256": observed_hashes,
        "prespec_sha256": sha256(args.prespec),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "status": value["status"],
        "open_confirmatory_holdout": open_holdout,
        "gates": gates,
        "summary": value["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
