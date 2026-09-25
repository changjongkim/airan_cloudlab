#!/usr/bin/env python3
"""Audit C135 exchanges against static and conditional recovery contracts."""

import argparse
import json
from pathlib import Path


def counterfactual_terms(protocol, cells, successful_nrx, live_recoveries,
                         ai_bound_ms):
    """Return the declared-bound comparison for one target branch."""
    deadline = float(protocol["deadline_ms"])
    radio_guard = float(protocol["guard_ms"])
    recovery = float(protocol["conv_bound_ms"])
    decision = float(protocol["nrx_bound_ms"])
    control = float(protocol["global_broker_transaction_budget_ms"])
    admission_guard = float(protocol["admission_ai_guard_ms"])
    ai_completion_guard = admission_guard - control
    if ai_completion_guard < 0:
        raise ValueError("admission guard is smaller than broker budget")
    capacity = deadline - radio_guard
    static_slack = capacity - cells * recovery
    recovery_start = capacity - live_recoveries * recovery
    conditional_window = recovery_start - decision
    effective = float(ai_bound_ms) + admission_guard
    return {
        "cells": cells,
        "successful_nrx": successful_nrx,
        "live_recoveries": live_recoveries,
        "radio_commit_guard_ms": radio_guard,
        "ai_completion_guard_ms": ai_completion_guard,
        "broker_control_budget_ms": control,
        "admission_guard_ms": admission_guard,
        "all_fail_base_slack_ms": static_slack,
        "conditional_recovery_start_ms": recovery_start,
        "decision_time_bound_ms": decision,
        "conditional_transaction_window_ms": conditional_window,
        "raw_ai_bound_ms": float(ai_bound_ms),
        "effective_transaction_bound_ms": effective,
        "static_margin_ms": static_slack - effective,
        "conditional_margin_ms": conditional_window - effective,
    }


def analyze_arm(result_path):
    result_path = Path(result_path).resolve()
    result = json.loads(result_path.read_text())
    protocol_path = Path(result["protocol"])
    protocol = json.loads(protocol_path.read_text())
    target_bound = float(protocol["target_ai_bound_ms"])
    events = []
    branch_count = 0
    safety_pass = bool(result["all_pass"])
    for home in result["homes"]:
        controller_path = Path(home["controller"])
        controller = json.loads(controller_path.read_text())
        by_index = {index: [] for index in range(protocol["iterations"])}
        for row in controller["records"]:
            by_index[row["index"]].append(row)
        background = {
            row["request_id"]: row for row in controller["background_records"]
        }
        for decision in controller["recovery_decisions"]:
            rows = by_index[decision["release_index"]]
            admitted = [row for row in rows if row["admitted"]]
            successful = [row for row in admitted if row["nrx_commit"]]
            rejected = [row for row in rows if not row["admitted"]]
            target_branch = (
                len(admitted) == 2
                and len(successful) == 2
                and len(rejected) == 2
            )
            branch_count += int(target_branch)
            record = background.get(decision.get("selected_request_id"))
            if not (
                target_branch
                and decision.get("ai_lease")
                and record is not None
                and float(record["bound_ms"]) == target_bound
            ):
                continue
            release_ns = rows[0]["release_ns"]
            terms = counterfactual_terms(
                protocol, controller["cells"], len(successful), len(rejected),
                record["bound_ms"],
            )
            horizon_offset_ms = (record["horizon_ns"] - release_ns) / 1e6
            physical_margin_ms = (
                record["horizon_ns"] - record["returned_ns"]
                - round(float(protocol["admission_ai_guard_ms"]) * 1e6)
            ) / 1e6
            latest_observation_ms = max(
                (row["nrx_observed_ns"] - release_ns) / 1e6
                for row in successful
            )
            events.append({
                "arm_result": str(result_path),
                "home": home["home"],
                "release_index": decision["release_index"],
                "request_id": record["request_id"],
                "phase": record["phase"],
                "terms": terms,
                "observed": {
                    "latest_nrx_observation_ms": latest_observation_ms,
                    "ai_admitted_ms": (record["admitted_ns"] - release_ns) / 1e6,
                    "ai_returned_ms": (record["returned_ns"] - release_ns) / 1e6,
                    "reserved_horizon_ms": horizon_offset_ms,
                    "physical_guarded_horizon_margin_ms": physical_margin_ms,
                },
                "checks": {
                    "static_contract_rejects": terms["static_margin_ms"] < 0,
                    "conditional_contract_accepts": terms["conditional_margin_ms"] >= 0,
                    "runtime_horizon_matches_model": abs(
                        horizon_offset_ms
                        - terms["conditional_recovery_start_ms"]
                    ) < 1e-6,
                    "nrx_observation_within_bound": (
                        latest_observation_ms <= terms["decision_time_bound_ms"]
                    ),
                    "physical_guarded_horizon_pass": physical_margin_ms >= 0,
                    "correct_execution_phase": (
                        record["phase"] == "after_nrx_before_recovery"
                    ),
                    "global_complete_confirmed": bool(
                        record["global_complete_confirmed"]
                    ),
                },
            })
    expected_exchanges = result["candidate_evidence"]["exchanges"]
    return {
        "result": str(result_path),
        "protocol": str(protocol_path),
        "target_branch_count": branch_count,
        "expected_candidate_exchanges": expected_exchanges,
        "audited_candidate_exchanges": len(events),
        "arm_result_pass": safety_pass,
        "events": events,
    }


def analyze(arm_paths):
    arms = [analyze_arm(path) for path in arm_paths]
    events = [event for arm in arms for event in arm["events"]]
    all_event_checks = [
        value for event in events for value in event["checks"].values()
    ]
    gates = {
        "two_independent_arms": len(arms) == 2,
        "arm_results_pass": all(arm["arm_result_pass"] for arm in arms),
        "candidate_counts_match_frozen_results": all(
            arm["audited_candidate_exchanges"]
            == arm["expected_candidate_exchanges"]
            for arm in arms
        ),
        "candidate_exchange_each_arm": all(
            arm["audited_candidate_exchanges"] > 0 for arm in arms
        ),
        "all_event_checks_pass": bool(events) and all(all_event_checks),
        "all_static_contracts_reject": bool(events) and all(
            event["terms"]["static_margin_ms"] < 0 for event in events
        ),
        "all_conditional_contracts_accept": bool(events) and all(
            event["terms"]["conditional_margin_ms"] >= 0 for event in events
        ),
    }
    return {
        "schema": "softwall-confirm135-static-counterfactual-audit-v1",
        "arms": arms,
        "totals": {
            "target_branches": sum(
                arm["target_branch_count"] for arm in arms
            ),
            "candidate_exchanges": len(events),
            "minimum_physical_guarded_horizon_margin_ms": min(
                event["observed"]["physical_guarded_horizon_margin_ms"]
                for event in events
            ) if events else None,
        },
        "gates": gates,
        "all_pass": all(gates.values()),
        "interpretation": (
            "Each event is a model counterfactual over the same physical C135 "
            "execution: the complete 57 ms AI-plus-control-plus-completion-guard "
            "transaction exceeds 53 ms static all-fail slack but fits the 58 ms "
            "conditional decision window. This is mechanism evidence, not a "
            "measured static-baseline throughput comparison."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = analyze(args.arms)
    output = Path(args.output).resolve()
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(output)
    if not value["all_pass"]:
        raise SystemExit("C135 static counterfactual audit failed")


if __name__ == "__main__":
    main()
