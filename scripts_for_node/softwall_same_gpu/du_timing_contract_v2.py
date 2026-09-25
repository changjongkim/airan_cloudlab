#!/usr/bin/env python3.11
"""Validate production DU evidence and request-specific SoftWall decisions."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

from c163_production_envelope_bridge import classify_document
from c163_build_du_contract_v2 import build_contract


SCHEMA = "softwall-du-timing-contract-v2"
ALLOWED_SOURCE_TYPES = {"du-fapi-trace", "du-mac-trace"}
ALLOWED_EXPIRY_SOURCES = {
    "du_configuration", "mac_scheduler_trace", "fapi_contract",
}
ALLOWED_CLOCK_METHODS = {
    "same-host-clock-monotonic-raw", "ptp-phc-calibrated", "tai-calibrated",
}
OUTCOMES = {"timely_success", "failed_or_late", "not_admitted"}
REQUIRED_TIMES = (
    "iq_ready_ns", "release_ns", "phy_submit_ns", "crc_visible_ns",
    "fapi_publish_ns", "mac_consume_ns", "expiry_ns",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact_ok(reference, artifacts):
    return (
        isinstance(reference, dict)
        and isinstance(reference.get("path"), str)
        and reference["path"] in artifacts
        and isinstance(reference.get("sha256"), str)
        and len(reference["sha256"]) == 64
        and digest(artifacts[reference["path"]]) == reference["sha256"]
    )


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1,
                              math.ceil(fraction * len(ordered)) - 1))]


def summarize(values):
    return {
        "count": len(values), "min_ns": min(values) if values else None,
        "p50_ns": percentile(values, .50), "p99_ns": percentile(values, .99),
        "max_ns": max(values) if values else None,
    }


def validate_document(document: dict, artifacts: dict[str, bytes]) -> dict:
    errors = []
    gates = {}
    gates["schema"] = document.get("schema") == SCHEMA
    source = document.get("source", {})
    gates["production_trace_provenance"] = (
        document.get("contract_kind") == "production-du"
        and source.get("type") in ALLOWED_SOURCE_TYPES
        and bool(source.get("identifier"))
        and _artifact_ok(source.get("artifact"), artifacts)
    )
    clock = document.get("clock", {})
    calibration = clock.get("calibration", {})
    uncertainty = calibration.get("max_error_ns")
    gates["qualified_clock_evidence"] = (
        bool(clock.get("domain")) and clock.get("timestamp_unit") == "ns"
        and clock.get("synchronized") is True
        and calibration.get("method") in ALLOWED_CLOCK_METHODS
        and isinstance(uncertainty, int) and not isinstance(uncertainty, bool)
        and uncertainty >= 0
        and _artifact_ok(calibration.get("artifact"), artifacts)
    )
    expiry = document.get("expiry_contract", {})
    gates["explicit_gnb_expiry_evidence"] = (
        expiry.get("source_type") in ALLOWED_EXPIRY_SOURCES
        and bool(expiry.get("source_reference"))
        and expiry.get("synthetic") is False
        and expiry.get("derived_from_ue_k2_n2") is False
        and _artifact_ok(expiry.get("artifact"), artifacts)
    )
    mode = document.get("mode", {})
    ai_bounds = mode.get("ai_class_bounds_ns")
    mode_fields = (
        mode.get("lifecycle") in {"warm", "cold", "restart", "long_idle"}
        and mode.get("node_bound_status") in {"qualified", "failed", "unknown"}
        and all(isinstance(mode.get(name), int) and not isinstance(mode.get(name), bool)
                and mode[name] > 0 for name in (
                    "recovery_capacity", "recovery_bound_ns", "nrx_bound_ns",
                    "control_bound_ns",
                ))
        and isinstance(mode.get("guard_ns"), int)
        and not isinstance(mode.get("guard_ns"), bool) and mode["guard_ns"] >= 0
        and isinstance(ai_bounds, dict) and bool(ai_bounds)
        and all(str(key).isdigit() and isinstance(value, int)
                and not isinstance(value, bool) and value > 0
                for key, value in ai_bounds.items())
    )
    gates["mode_fields"] = bool(mode_fields)
    gates["qualified_mode_provenance"] = (
        bool(mode.get("qualification_reference"))
        and _artifact_ok(mode.get("qualification_artifact"), artifacts)
    )
    gates["artifact_reconstruction_matches"] = False
    provenance_refs = {
        "raw": source.get("artifact"),
        "clock": calibration.get("artifact"),
        "expiry": expiry.get("artifact"),
        "mode": mode.get("qualification_artifact"),
    }
    if all(_artifact_ok(reference, artifacts)
           for reference in provenance_refs.values()):
        try:
            decoded = {
                name: json.loads(artifacts[reference["path"]])
                for name, reference in provenance_refs.items()
            }
            rebuilt = build_contract(
                decoded["raw"], decoded["clock"], decoded["expiry"],
                decoded["mode"], provenance_refs,
            )
            gates["artifact_reconstruction_matches"] = rebuilt == document
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    records = document.get("records")
    decisions = document.get("decisions")
    gates["nonempty_records_and_decisions"] = (
        isinstance(records, list) and bool(records)
        and isinstance(decisions, list) and bool(decisions)
    )
    if not gates["nonempty_records_and_decisions"]:
        records, decisions = [], []
    record_ids = set()
    records_by_batch = {}
    structure_errors = duplicate_ids = clock_errors = order_errors = 0
    commit_errors = expiry_errors = deadline_misses = outcome_errors = 0
    release_to_expiry = []
    release_to_consume = []
    uncertainty_value = uncertainty if isinstance(uncertainty, int) else 0
    for index, record in enumerate(records):
        required = (
            "request_id", "batch_id", "home_id", "cell_id", "slot_id",
            "clock_domain", "radio_commit_count", "nrx_admitted",
            "nrx_outcome_at_decision", "nrx_outcome_observed_ns",
        ) + REQUIRED_TIMES
        if any(name not in record for name in required):
            structure_errors += 1
            continue
        request_id = record["request_id"]
        if not all(isinstance(record[name], str) and record[name]
                   for name in ("request_id", "batch_id", "home_id")):
            structure_errors += 1
            continue
        if (not isinstance(record["cell_id"], int)
                or isinstance(record["cell_id"], bool)
                or record["cell_id"] < 0
                or not isinstance(record["slot_id"], int)
                or isinstance(record["slot_id"], bool)
                or record["slot_id"] < 0
                or not isinstance(record["radio_commit_count"], int)
                or isinstance(record["radio_commit_count"], bool)):
            structure_errors += 1
            continue
        if request_id in record_ids:
            duplicate_ids += 1
        record_ids.add(request_id)
        records_by_batch.setdefault(record["batch_id"], []).append(record)
        if record["clock_domain"] != clock.get("domain"):
            clock_errors += 1
        values = [record[name] for name in REQUIRED_TIMES]
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
               for value in values):
            structure_errors += 1
            continue
        iq, release, submit, crc, fapi, consume, request_expiry = values
        if not (iq <= release <= submit <= crc <= fapi <= consume):
            order_errors += 1
        if record["radio_commit_count"] != 1:
            commit_errors += 1
        if request_expiry - uncertainty_value <= release:
            expiry_errors += 1
        else:
            release_to_expiry.append(request_expiry - release)
        release_to_consume.append(consume - release)
        if consume + uncertainty_value > request_expiry:
            deadline_misses += 1
        admitted = record["nrx_admitted"]
        outcome = record["nrx_outcome_at_decision"]
        outcome_observed = record["nrx_outcome_observed_ns"]
        if (not isinstance(admitted, bool) or outcome not in OUTCOMES
                or not isinstance(outcome_observed, int)
                or isinstance(outcome_observed, bool)
                or not release <= outcome_observed <= consume
                or (admitted and outcome == "not_admitted")
                or ((not admitted) and outcome != "not_admitted")):
            outcome_errors += 1
    gates.update({
        "record_structure": structure_errors == 0,
        "unique_request_ids": duplicate_ids == 0,
        "record_clock_matches": clock_errors == 0,
        "timestamp_order": order_errors == 0,
        "single_radio_commit": commit_errors == 0,
        "positive_effective_expiry": expiry_errors == 0,
        "observed_mac_before_expiry_with_clock_error": deadline_misses == 0,
        "nrx_outcome_semantics": outcome_errors == 0,
    })

    decision_errors = coverage_errors = actual_ai_errors = 0
    outcome_visibility_errors = 0
    decision_ids = set()
    ai_bound_keys = {int(key) for key in ai_bounds} if isinstance(ai_bounds, dict) else set()
    for decision in decisions:
        required = (
            "batch_id", "decision_ns", "mandatory_request_ids",
            "unresolved_request_ids", "ai_context_length", "ai_deadline_ns",
            "ai_lease_accepted", "ai_complete_ns",
        )
        if any(name not in decision for name in required):
            decision_errors += 1
            continue
        batch_id = decision["batch_id"]
        if batch_id in decision_ids:
            decision_errors += 1
        decision_ids.add(batch_id)
        batch_records = records_by_batch.get(batch_id, [])
        batch_ids = {row["request_id"] for row in batch_records}
        mandatory = decision["mandatory_request_ids"]
        unresolved = decision["unresolved_request_ids"]
        id_lists_valid = (
            isinstance(mandatory, list) and isinstance(unresolved, list)
            and all(isinstance(value, str) and value for value in mandatory)
            and all(isinstance(value, str) and value for value in unresolved)
        )
        if (not isinstance(batch_id, str) or not batch_id
                or not isinstance(decision["decision_ns"], int)
                or isinstance(decision["decision_ns"], bool)
                or decision["decision_ns"] < 0
                or not id_lists_valid
                or len(mandatory) != len(set(mandatory))
                or len(unresolved) != len(set(unresolved))):
            decision_errors += 1
            continue
        expected_unresolved = {
            row["request_id"] for row in batch_records
            if row["nrx_outcome_at_decision"] != "timely_success"
        }
        if (set(mandatory) != batch_ids or not set(unresolved).issubset(batch_ids)
                or set(unresolved) != expected_unresolved):
            coverage_errors += 1
        if any(row["nrx_outcome_observed_ns"] > decision["decision_ns"]
               for row in batch_records):
            outcome_visibility_errors += 1
        context = decision["ai_context_length"]
        accepted = decision["ai_lease_accepted"]
        complete = decision["ai_complete_ns"]
        deadline = decision["ai_deadline_ns"]
        if context is None:
            if deadline is not None or accepted is not False or complete is not None:
                actual_ai_errors += 1
        elif (not isinstance(context, int) or isinstance(context, bool)
              or context not in ai_bound_keys or not isinstance(deadline, int)
              or isinstance(deadline, bool) or deadline <= decision["decision_ns"]
              or not isinstance(accepted, bool)
              or (accepted and (not isinstance(complete, int)
                                or isinstance(complete, bool)
                                or complete < decision["decision_ns"]
                                or complete > deadline))
              or ((not accepted) and complete is not None)):
            actual_ai_errors += 1
    gates["one_decision_per_batch"] = (
        decision_errors == 0 and decision_ids == set(records_by_batch)
    )
    gates["mandatory_and_unresolved_coverage"] = coverage_errors == 0
    gates["nrx_outcomes_visible_by_decision"] = outcome_visibility_errors == 0
    gates["physical_ai_outcome_fields"] = actual_ai_errors == 0

    structural_names = (
        "schema", "production_trace_provenance", "qualified_clock_evidence",
        "explicit_gnb_expiry_evidence", "mode_fields",
        "qualified_mode_provenance",
        "nonempty_records_and_decisions", "record_structure",
        "unique_request_ids", "record_clock_matches", "timestamp_order",
        "single_radio_commit", "positive_effective_expiry",
        "nrx_outcome_semantics", "one_decision_per_batch",
        "mandatory_and_unresolved_coverage", "nrx_outcomes_visible_by_decision",
        "physical_ai_outcome_fields",
    )
    structurally_valid = all(gates.get(name, False) for name in structural_names)
    predictions = []
    if structurally_valid:
        predictions = classify_document(document)
    gates["verified_all_fail_certificate"] = (
        bool(predictions)
        and all(row.mandatory_all_fail_safe for row in predictions)
    )
    gates["model_matches_physical_ai_decision"] = (
        bool(predictions) and all(row.decision_matches for row in predictions)
    )
    trace_valid = structurally_valid and gates["artifact_reconstruction_matches"]
    all_pass = trace_valid and all(gates.values())
    errors = [name for name, passed in gates.items() if not passed]
    return {
        "schema": "softwall-du-timing-validation-v2",
        "status": "QUALIFIED" if all_pass else "UNQUALIFIED",
        "trace_valid": trace_valid, "all_pass": all_pass, "gates": gates,
        "counts": {
            "records": len(records), "batches": len(records_by_batch),
            "deadline_misses": deadline_misses,
            "model_decision_mismatches": sum(
                not row.decision_matches for row in predictions
            ),
            "states": {state: sum(row.state == state for row in predictions)
                       for state in ("QSU", "QSN", "MI", "UQ")},
        },
        "timing": {
            "release_to_expiry": summarize(release_to_expiry),
            "release_to_mac_consume": summarize(release_to_consume),
        },
        "predictions": [dataclasses.asdict(row) for row in predictions],
        "errors": errors,
        "claim_boundary": (
            "QUALIFIED proves schema/provenance consistency, request-specific exact "
            "all-fail feasibility, observed decision agreement, and finite observed "
            "deadlines for the supplied mode. It is not a WCET proof."
        ),
    }


def load_artifacts(document: dict, root: Path) -> dict[str, bytes]:
    root = root.resolve()
    references = (
        document.get("source", {}).get("artifact"),
        document.get("clock", {}).get("calibration", {}).get("artifact"),
        document.get("expiry_contract", {}).get("artifact"),
        document.get("mode", {}).get("qualification_artifact"),
    )
    artifacts = {}
    for reference in references:
        if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
            continue
        path = (root / reference["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("artifact path escapes artifact root")
        artifacts[reference["path"]] = path.read_bytes()
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    document = json.loads(raw)
    result = validate_document(document, load_artifacts(document, args.artifact_root))
    result["input"] = str(args.input.resolve())
    result["input_sha256"] = digest(raw)
    result["validator_sha256"] = digest(Path(__file__).read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, indent=2))
    raise SystemExit(0 if result["all_pass"] else 2)


if __name__ == "__main__":
    main()
