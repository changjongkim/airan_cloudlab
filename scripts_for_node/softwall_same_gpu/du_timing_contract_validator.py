#!/usr/bin/env python3
"""Validate a production DU/FAPI timing trace before SoftWall qualification."""

import argparse
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "softwall-du-timing-contract-v1"
REQUIRED_TIMES = (
    "iq_ready_ns",
    "release_ns",
    "phy_submit_ns",
    "crc_visible_ns",
    "fapi_publish_ns",
    "mac_consume_ns",
    "expiry_ns",
)
ALLOWED_EXPIRY_SOURCES = {
    "du_configuration",
    "mac_scheduler_trace",
    "fapi_contract",
}


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def validate_document(document):
    errors = []
    gates = {}

    gates["schema"] = document.get("schema") == SCHEMA
    if not gates["schema"]:
        errors.append("schema must be " + SCHEMA)

    source = document.get("source", {})
    source_ok = (
        document.get("contract_kind") == "production-du"
        and source.get("type") in {"du-fapi-trace", "du-mac-trace"}
        and bool(source.get("identifier"))
        and bool(source.get("artifact_sha256"))
    )
    gates["production_trace_provenance"] = source_ok
    if not source_ok:
        errors.append("production DU/FAPI trace provenance is incomplete")

    clock = document.get("clock", {})
    clock_ok = (
        bool(clock.get("domain"))
        and clock.get("timestamp_unit") == "ns"
        and clock.get("synchronized") is True
    )
    gates["single_synchronized_clock"] = clock_ok
    if not clock_ok:
        errors.append("one synchronized nanosecond clock domain is required")

    expiry = document.get("expiry_contract", {})
    expiry_ok = (
        expiry.get("source_type") in ALLOWED_EXPIRY_SOURCES
        and bool(expiry.get("source_reference"))
        and expiry.get("synthetic") is False
        and expiry.get("derived_from_ue_k2_n2") is False
    )
    gates["explicit_gnb_expiry_contract"] = expiry_ok
    if not expiry_ok:
        errors.append("expiry must come from an explicit gNB DU/MAC/FAPI contract, not synthetic D or UE K2/N2")

    records = document.get("records")
    records_ok = isinstance(records, list) and len(records) > 0
    gates["nonempty_records"] = records_ok
    if not records_ok:
        errors.append("records must be a nonempty list")
        records = []

    identifiers = set()
    duplicate_ids = 0
    structural_errors = 0
    clock_mismatches = 0
    monotonic_errors = 0
    commit_errors = 0
    invalid_expiries = 0
    deadline_misses = 0
    release_to_expiry = []
    iq_to_release = []
    release_to_mac = []

    for index, record in enumerate(records):
        request_id = record.get("request_id")
        required = ("request_id", "cell_id", "slot_id", "clock_domain", "radio_commit_count") + REQUIRED_TIMES
        if any(field not in record for field in required):
            structural_errors += 1
            errors.append("record {} is missing required fields".format(index))
            continue
        if request_id in identifiers:
            duplicate_ids += 1
        identifiers.add(request_id)
        if record["clock_domain"] != clock.get("domain"):
            clock_mismatches += 1
        values = [record[name] for name in REQUIRED_TIMES]
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            structural_errors += 1
            errors.append("record {} has a non-integer or negative timestamp".format(index))
            continue
        iq, release, submit, crc, fapi, consume, request_expiry = values
        if not (iq <= release <= submit <= crc <= fapi <= consume):
            monotonic_errors += 1
        if record["radio_commit_count"] != 1:
            commit_errors += 1
        if request_expiry <= release:
            invalid_expiries += 1
        else:
            release_to_expiry.append(request_expiry - release)
        iq_to_release.append(release - iq)
        release_to_mac.append(consume - release)
        if consume > request_expiry:
            deadline_misses += 1

    gates["record_structure"] = structural_errors == 0
    gates["unique_request_ids"] = duplicate_ids == 0
    gates["record_clock_matches"] = clock_mismatches == 0
    gates["timestamp_order"] = monotonic_errors == 0
    gates["single_radio_commit"] = commit_errors == 0
    gates["positive_request_expiry"] = invalid_expiries == 0
    gates["observed_mac_consumption_before_expiry"] = deadline_misses == 0

    mandatory = document.get("mandatory_contract", {})
    home_cells = mandatory.get("home_cell_counts", [])
    recovery = mandatory.get("recovery_bound_ns")
    guard = mandatory.get("guard_ns")
    mandatory_fields_ok = (
        isinstance(home_cells, list)
        and bool(home_cells)
        and all(isinstance(count, int) and count > 0 for count in home_cells)
        and isinstance(recovery, int) and not isinstance(recovery, bool) and recovery > 0
        and isinstance(guard, int) and not isinstance(guard, bool) and guard >= 0
    )
    gates["mandatory_contract_fields"] = mandatory_fields_ok
    mandatory_capacity_ok = False
    home_demand = []
    if mandatory_fields_ok and release_to_expiry:
        minimum_deadline = min(release_to_expiry)
        home_demand = [count * recovery + guard for count in home_cells]
        mandatory_capacity_ok = all(value <= minimum_deadline for value in home_demand)
    gates["conservative_all_fail_capacity"] = mandatory_capacity_ok
    if not mandatory_capacity_ok:
        errors.append("declared mandatory all-fail capacity does not fit the minimum observed request deadline")

    structural_gate_names = (
        "schema",
        "production_trace_provenance",
        "single_synchronized_clock",
        "explicit_gnb_expiry_contract",
        "nonempty_records",
        "record_structure",
        "unique_request_ids",
        "record_clock_matches",
        "timestamp_order",
        "single_radio_commit",
        "positive_request_expiry",
        "mandatory_contract_fields",
    )
    trace_valid = all(gates.get(name, False) for name in structural_gate_names)
    all_pass = trace_valid and all(gates.values())

    def summary(values):
        if not values:
            return {"count": 0, "min_ns": None, "p50_ns": None, "p99_ns": None, "max_ns": None}
        return {
            "count": len(values),
            "min_ns": min(values),
            "p50_ns": percentile(values, 0.50),
            "p99_ns": percentile(values, 0.99),
            "max_ns": max(values),
        }

    return {
        "schema": "softwall-du-timing-validation-v1",
        "input_schema": document.get("schema"),
        "gates": gates,
        "trace_valid": trace_valid,
        "all_pass": all_pass,
        "status": "QUALIFIED" if all_pass else "UNQUALIFIED",
        "counts": {
            "records": len(records),
            "duplicate_request_ids": duplicate_ids,
            "structural_errors": structural_errors,
            "clock_mismatches": clock_mismatches,
            "timestamp_order_errors": monotonic_errors,
            "radio_commit_count_errors": commit_errors,
            "invalid_expiries": invalid_expiries,
            "deadline_misses": deadline_misses,
        },
        "timing": {
            "iq_ready_to_release": summary(iq_to_release),
            "release_to_expiry": summary(release_to_expiry),
            "release_to_mac_consume": summary(release_to_mac),
        },
        "mandatory_capacity": {
            "home_demand_with_guard_ns": home_demand,
            "minimum_request_deadline_ns": min(release_to_expiry) if release_to_expiry else None,
        },
        "errors": errors,
        "claim_boundary": (
            "QUALIFIED validates trace provenance, timestamp structure, observed deadline outcomes, "
            "and a conservative mandatory capacity check; it does not turn finite observations into WCET."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    result = validate_document(document)
    result["input"] = str(args.input.resolve())
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    result["validator_sha256"] = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, indent=2))
    if not result["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
