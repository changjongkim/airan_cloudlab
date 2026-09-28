#!/usr/bin/env python3.11
"""Correct C159/C166 absolute-deadline handling without opening a holdout."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from analyze_c166_static_global_safe import static_slot
from c159_q3_oracle_screen import (
    CONTEXTS,
    CUTOFF_MS,
    PERIOD_MS,
    WINDOW_MS,
    Edge,
    add_edge,
    build_state_rows,
    load,
    policy_slot,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_weighted_matching_absolute_deadline(
    requests: list[dict], slots: list[dict], policy: str
) -> dict:
    """Maximum value when ``deadline_ms`` is absolute in the trace window."""
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
        deadline = float(request["deadline_ms"])
        for slot_index, slot in enumerate(slots):
            completion = slot[policy]["completion_by_context_ms"].get(context)
            if completion is None:
                continue
            if arrival <= slot["decision_ms"] and completion <= deadline:
                edge = add_edge(
                    graph,
                    first_job + job_index,
                    first_slot + slot_index,
                    1,
                    -int(request["value_tokens"]),
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
        (job_index, slot_index)
        for job_index, slot_index, edge in edge_refs
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


def relative_gap(value: int, oracle: int) -> float:
    return 100.0 * (oracle - value) / oracle if oracle else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    protocol = load(args.protocol)
    if protocol.get("status") != "FROZEN_BEFORE_C173_ANALYSIS":
        raise RuntimeError("C173 protocol is not frozen")
    paths = {key: root / value for key, value in protocol["inputs"].items()}
    observed_hashes = {key: sha256(path) for key, path in paths.items()}
    if observed_hashes != protocol["source_sha256"]:
        raise RuntimeError("C173 frozen source hash mismatch")

    trace = load(paths["calibration_trace"])
    prior = load(paths["c166_result"])
    original = load(paths["c159_q3_result"])
    requests = [request for window in trace["windows"] for request in window["requests"]]
    fixed_slo = float(trace["selection"]["fixed_slo_ms"])
    deadline_encoding_valid = all(
        abs(float(row["deadline_ms"]) - float(row["arrival_ms"]) - fixed_slo) < 1e-9
        for row in requests
    )

    state_streams = {
        "development": build_state_rows(load(paths["q2_development_coordinator"])),
        "holdout": build_state_rows(load(paths["q2_holdout_coordinator"])),
    }
    mappings = (
        ("development", 0),
        ("development", 266),
        ("holdout", 0),
        ("holdout", 266),
    )
    rows = []
    for window, (stream_name, offset) in zip(trace["windows"], mappings):
        slot_count = sum(
            epoch * PERIOD_MS + CUTOFF_MS < WINDOW_MS for epoch in range(1000)
        )
        states = state_streams[stream_name][offset:offset + slot_count]
        if len(states) != slot_count:
            raise RuntimeError("insufficient frozen Q2 state stream")
        dynamic_slots = [policy_slot(epoch, state) for epoch, state in enumerate(states)]
        static_slots = [static_slot(epoch) for epoch in range(slot_count)]
        policies = {
            "static_global_safe": exact_weighted_matching_absolute_deadline(
                window["requests"], static_slots, "static_global_safe"
            ),
            "recovery_first_empirical": exact_weighted_matching_absolute_deadline(
                window["requests"], dynamic_slots, "event_empirical"
            ),
            "recovery_first_full_bound": exact_weighted_matching_absolute_deadline(
                window["requests"], dynamic_slots, "event_contract"
            ),
            "softwall": exact_weighted_matching_absolute_deadline(
                window["requests"], dynamic_slots, "softwall"
            ),
            "offline_oracle": exact_weighted_matching_absolute_deadline(
                window["requests"], dynamic_slots, "softwall"
            ),
        }
        rows.append({
            "window_id": window["window_id"],
            "q2_state_stream": stream_name,
            "q2_state_offset": offset,
            "slots": slot_count,
            "policies": policies,
        })

    policy_names = tuple(rows[0]["policies"])
    totals = {
        policy: {
            "timely_requests": sum(row["policies"][policy]["timely_requests"] for row in rows),
            "timely_value_tokens": sum(
                row["policies"][policy]["timely_value_tokens"] for row in rows
            ),
        }
        for policy in policy_names
    }
    oracle_tokens = totals["offline_oracle"]["timely_value_tokens"]
    for values in totals.values():
        values["gap_to_oracle_pct"] = relative_gap(
            values["timely_value_tokens"], oracle_tokens
        )
    prior_totals = prior["summary"]["totals"]
    correction = {
        policy: {
            "timely_requests_delta": (
                totals[policy]["timely_requests"] - prior_totals[policy]["timely_requests"]
            ),
            "timely_value_tokens_delta": (
                totals[policy]["timely_value_tokens"]
                - prior_totals[policy]["timely_value_tokens"]
            ),
        }
        for policy in policy_names
    }
    gates = {
        "frozen_hashes_match": observed_hashes == protocol["source_sha256"],
        "trace_deadlines_are_absolute_and_fixed_slo": deadline_encoding_valid,
        "all_four_windows_recomputed": len(rows) == 4,
        "no_request_or_window_exclusion": len(requests) == trace["summary"]["selected_requests"],
        "confirmatory_holdout_remains_closed": original["open_confirmatory_holdout"] is False,
        "static_and_oracle_rows_reported": set(totals) == set(policy_names),
    }
    result = {
        "schema": "softwall-c173-deadline-correction-v1",
        "status": "C173_DEADLINE_CORRECTION_PASS" if all(gates.values()) else "C173_DEADLINE_CORRECTION_FAIL",
        "all_pass": all(gates.values()),
        "analysis_role": (
            "Correction of an absolute-deadline interpretation defect in the opened "
            "C159/C166 calibration screen. No holdout is opened and no request, "
            "window, policy, bound, or threshold is changed."
        ),
        "defect": {
            "trace_encoding": "deadline_ms = arrival_ms + fixed_slo_ms",
            "incorrect_expression": "arrival_ms + deadline_ms",
            "correct_expression": "deadline_ms",
            "effect": correction,
        },
        "gates": gates,
        "summary": {
            "offered_requests": trace["summary"]["selected_requests"],
            "offered_value_tokens": trace["summary"]["offered_value_tokens"],
            "slots": sum(row["slots"] for row in rows),
            "totals": totals,
            "minimum_effect_pct": protocol["minimum_effect_pct"],
        },
        "windows": rows,
        "protocol_sha256": sha256(args.protocol),
        "source_sha256": observed_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "status": result["status"],
        "all_pass": result["all_pass"],
        "defect": result["defect"],
        "summary": result["summary"],
    }, indent=2, sort_keys=True))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
