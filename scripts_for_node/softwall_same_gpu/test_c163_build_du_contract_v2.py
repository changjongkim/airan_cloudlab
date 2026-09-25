#!/usr/bin/env python3.11

import hashlib
import json
import unittest

from c163_build_du_contract_v2 import build_contract
from du_timing_contract_v2 import validate_document


def fixtures():
    clock_domain = "monotonic-raw-host0"
    identity = {
        "request_id": "r0", "batch_id": "b0", "home_id": "h0",
        "cell_id": 0, "slot_id": 7, "clock_domain": clock_domain,
    }
    events = []
    for name, timestamp in (
        ("iq_ready", 5), ("release", 10), ("phy_submit", 20),
        ("crc_visible", 100), ("fapi_publish", 110),
        ("mac_consume", 120), ("radio_commit", 100),
    ):
        events.append(identity | {"event": name, "timestamp_ns": timestamp})
    events.append(identity | {
        "event": "nrx_outcome", "timestamp_ns": 60,
        "nrx_admitted": True, "nrx_outcome_at_decision": "failed_or_late",
    })
    raw = {
        "schema": "softwall-c163-raw-du-trace-v1",
        "source_type": "du-fapi-trace", "identifier": "unit-only",
        "clock_domain": clock_domain, "events": events,
        "decisions": [{
            "batch_id": "b0", "decision_ns": 60,
            "mandatory_request_ids": ["r0"],
            "unresolved_request_ids": ["r0"],
            "ai_context_length": 128, "ai_deadline_ns": 150,
            "ai_lease_accepted": True, "ai_complete_ns": 100,
        }],
    }
    clock = {
        "schema": "softwall-c163-clock-calibration-v1",
        "domain": clock_domain, "synchronized": True,
        "method": "same-host-clock-monotonic-raw", "max_error_ns": 1,
    }
    expiry = {
        "schema": "softwall-c163-expiry-contract-v1",
        "source_type": "du_configuration", "source_reference": "unit-only",
        "synthetic": False, "derived_from_ue_k2_n2": False,
        "entries": [{"request_id": "r0", "expiry_ns": 200}],
    }
    mode = {
        "schema": "softwall-c163-mode-qualification-v1",
        "lifecycle": "warm", "node_bound_status": "qualified",
        "recovery_capacity": 1, "recovery_bound_ns": 25,
        "nrx_bound_ns": 45, "control_bound_ns": 5, "guard_ns": 2,
        "ai_class_bounds_ns": {"128": 40},
        "qualification_reference": "unit-only",
    }
    contents = {
        name: (json.dumps(value, sort_keys=True) + "\n").encode()
        for name, value in (
            ("raw", raw), ("clock", clock),
            ("expiry", expiry), ("mode", mode),
        )
    }
    refs = {name: {
        "path": name + ".json", "sha256": hashlib.sha256(data).hexdigest()
    } for name, data in contents.items()}
    artifacts = {refs[name]["path"]: contents[name] for name in refs}
    return raw, clock, expiry, mode, refs, artifacts


class BuildDUContractV2Test(unittest.TestCase):
    def test_built_contract_passes_v2_validator(self):
        raw, clock, expiry, mode, refs, artifacts = fixtures()
        contract = build_contract(raw, clock, expiry, mode, refs)
        result = validate_document(contract, artifacts)
        self.assertTrue(result["all_pass"])
        self.assertEqual(contract["records"][0]["radio_commit_count"], 1)

    def test_duplicate_timing_event_is_rejected(self):
        raw, clock, expiry, mode, refs, _ = fixtures()
        raw["events"].append(dict(raw["events"][0]))
        with self.assertRaisesRegex(ValueError, "unique timing event"):
            build_contract(raw, clock, expiry, mode, refs)

    def test_identity_change_is_rejected(self):
        raw, clock, expiry, mode, refs, _ = fixtures()
        raw["events"][1]["home_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "identity changed"):
            build_contract(raw, clock, expiry, mode, refs)

    def test_request_and_expiry_sets_must_match(self):
        raw, clock, expiry, mode, refs, _ = fixtures()
        expiry["entries"].append({"request_id": "missing", "expiry_ns": 200})
        with self.assertRaisesRegex(ValueError, "requests differ"):
            build_contract(raw, clock, expiry, mode, refs)

    def test_clock_domains_must_match(self):
        raw, clock, expiry, mode, refs, _ = fixtures()
        clock["domain"] = "other"
        with self.assertRaisesRegex(ValueError, "domains differ"):
            build_contract(raw, clock, expiry, mode, refs)


if __name__ == "__main__":
    unittest.main()
