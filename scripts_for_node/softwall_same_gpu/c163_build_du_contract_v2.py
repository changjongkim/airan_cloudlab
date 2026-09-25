#!/usr/bin/env python3.11
"""Join raw DU events, expiry, clock, and mode evidence into contract v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RAW_SCHEMA = "softwall-c163-raw-du-trace-v1"
CLOCK_SCHEMA = "softwall-c163-clock-calibration-v1"
EXPIRY_SCHEMA = "softwall-c163-expiry-contract-v1"
MODE_SCHEMA = "softwall-c163-mode-qualification-v1"
EVENT_TO_FIELD = {
    "iq_ready": "iq_ready_ns",
    "release": "release_ns",
    "phy_submit": "phy_submit_ns",
    "crc_visible": "crc_visible_ns",
    "fapi_publish": "fapi_publish_ns",
    "mac_consume": "mac_consume_ns",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reference(path: str, data: bytes) -> dict:
    return {"path": path, "sha256": digest(data)}


def build_contract(raw: dict, clock: dict, expiry: dict, mode: dict,
                   artifact_references: dict[str, dict]) -> dict:
    if raw.get("schema") != RAW_SCHEMA:
        raise ValueError("invalid raw DU trace schema")
    if clock.get("schema") != CLOCK_SCHEMA:
        raise ValueError("invalid clock calibration schema")
    if expiry.get("schema") != EXPIRY_SCHEMA:
        raise ValueError("invalid expiry contract schema")
    if mode.get("schema") != MODE_SCHEMA:
        raise ValueError("invalid mode qualification schema")
    clock_domain = raw.get("clock_domain")
    if not isinstance(clock_domain, str) or not clock_domain:
        raise ValueError("raw trace requires clock_domain")
    if clock.get("domain") != clock_domain:
        raise ValueError("raw trace and clock calibration domains differ")

    expiry_rows = expiry.get("entries")
    if not isinstance(expiry_rows, list) or not expiry_rows:
        raise ValueError("expiry entries must be nonempty")
    expiry_by_id = {}
    for row in expiry_rows:
        request_id = row.get("request_id")
        timestamp = row.get("expiry_ns")
        if (not isinstance(request_id, str) or not request_id
                or request_id in expiry_by_id
                or not isinstance(timestamp, int) or isinstance(timestamp, bool)
                or timestamp < 0):
            raise ValueError("invalid or duplicate expiry entry")
        expiry_by_id[request_id] = timestamp

    groups = {}
    identities = {}
    for event in raw.get("events", []):
        request_id = event.get("request_id")
        identity = tuple(event.get(name) for name in (
            "batch_id", "home_id", "cell_id", "slot_id", "clock_domain"
        ))
        if (not isinstance(request_id, str) or not request_id
                or identity[-1] != clock_domain):
            raise ValueError("invalid request event identity")
        if request_id in identities and identities[request_id] != identity:
            raise ValueError("request identity changed across events")
        identities[request_id] = identity
        groups.setdefault(request_id, []).append(event)
    if not groups or set(groups) != set(expiry_by_id):
        raise ValueError("trace requests and expiry requests differ")

    records = []
    for request_id in sorted(groups):
        events = groups[request_id]
        by_name = {}
        for event in events:
            by_name.setdefault(event.get("event"), []).append(event)
        if any(len(by_name.get(name, [])) != 1 for name in EVENT_TO_FIELD):
            raise ValueError(f"request {request_id} lacks a unique timing event")
        if len(by_name.get("nrx_outcome", [])) != 1:
            raise ValueError(f"request {request_id} lacks a unique NRx outcome")
        commits = by_name.get("radio_commit", [])
        batch_id, home_id, cell_id, slot_id, event_clock = identities[request_id]
        record = {
            "request_id": request_id, "batch_id": batch_id,
            "home_id": home_id, "cell_id": cell_id, "slot_id": slot_id,
            "clock_domain": event_clock, "expiry_ns": expiry_by_id[request_id],
            "radio_commit_count": len(commits),
        }
        for event_name, field in EVENT_TO_FIELD.items():
            record[field] = by_name[event_name][0].get("timestamp_ns")
        outcome = by_name["nrx_outcome"][0]
        record["nrx_admitted"] = outcome.get("nrx_admitted")
        record["nrx_outcome_at_decision"] = outcome.get(
            "nrx_outcome_at_decision"
        )
        record["nrx_outcome_observed_ns"] = outcome.get("timestamp_ns")
        records.append(record)

    decisions = raw.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("raw trace requires decisions")
    known_batches = {record["batch_id"] for record in records}
    if {row.get("batch_id") for row in decisions} != known_batches:
        raise ValueError("decision batches and request batches differ")

    return {
        "schema": "softwall-du-timing-contract-v2",
        "contract_kind": "production-du",
        "source": {
            "type": raw.get("source_type"),
            "identifier": raw.get("identifier"),
            "artifact": artifact_references["raw"],
        },
        "clock": {
            "domain": clock_domain, "timestamp_unit": "ns",
            "synchronized": clock.get("synchronized"),
            "calibration": {
                "method": clock.get("method"),
                "max_error_ns": clock.get("max_error_ns"),
                "artifact": artifact_references["clock"],
            },
        },
        "expiry_contract": {
            "source_type": expiry.get("source_type"),
            "source_reference": expiry.get("source_reference"),
            "synthetic": expiry.get("synthetic"),
            "derived_from_ue_k2_n2": expiry.get("derived_from_ue_k2_n2"),
            "artifact": artifact_references["expiry"],
        },
        "mode": {
            name: mode.get(name) for name in (
                "lifecycle", "node_bound_status", "recovery_capacity",
                "recovery_bound_ns", "nrx_bound_ns", "control_bound_ns",
                "guard_ns", "ai_class_bounds_ns", "qualification_reference",
            )
        } | {"qualification_artifact": artifact_references["mode"]},
        "records": records,
        "decisions": decisions,
    }


def _load_inside(root: Path, path: Path) -> tuple[dict, str, bytes]:
    root = root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("input artifact escapes artifact root")
    data = resolved.read_bytes()
    return json.loads(data), str(resolved.relative_to(root)), data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--raw-trace", type=Path, required=True)
    parser.add_argument("--clock-calibration", type=Path, required=True)
    parser.add_argument("--expiry-contract", type=Path, required=True)
    parser.add_argument("--mode-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = {}
    refs = {}
    for key, path in (
        ("raw", args.raw_trace), ("clock", args.clock_calibration),
        ("expiry", args.expiry_contract), ("mode", args.mode_qualification),
    ):
        loaded[key], relative, data = _load_inside(args.artifact_root, path)
        refs[key] = reference(relative, data)
    contract = build_contract(
        loaded["raw"], loaded["clock"], loaded["expiry"], loaded["mode"], refs
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "status": "BUILT", "records": len(contract["records"]),
        "decisions": len(contract["decisions"]),
    }, indent=2))


if __name__ == "__main__":
    main()
