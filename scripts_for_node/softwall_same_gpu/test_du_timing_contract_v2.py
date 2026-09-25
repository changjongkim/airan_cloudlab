#!/usr/bin/env python3.11

import copy
import hashlib
import json
from pathlib import Path
import unittest

from du_timing_contract_v2 import load_artifacts, validate_document


CLOCK_ARTIFACT = {
    "schema": "softwall-c163-clock-calibration-v1",
    "domain": "monotonic-raw-host0", "synchronized": True,
    "method": "same-host-clock-monotonic-raw", "max_error_ns": 1,
}
EXPIRY_ARTIFACT = {
    "schema": "softwall-c163-expiry-contract-v1",
    "source_type": "du_configuration", "source_reference": "unit-only-config",
    "synthetic": False, "derived_from_ue_k2_n2": False,
    "entries": [
        {"request_id": "r0", "expiry_ns": 200},
        {"request_id": "r1", "expiry_ns": 201},
    ],
}
MODE_ARTIFACT = {
    "schema": "softwall-c163-mode-qualification-v1",
    "lifecycle": "warm", "node_bound_status": "qualified",
    "recovery_capacity": 1, "recovery_bound_ns": 25,
    "nrx_bound_ns": 45, "control_bound_ns": 5,
    "guard_ns": 2, "ai_class_bounds_ns": {"128": 40},
    "qualification_reference": "unit-only mode qualification",
}


def raw_artifact():
    events = []
    for index, outcome in enumerate(("timely_success", "failed_or_late")):
        identity = {
            "request_id": f"r{index}", "batch_id": "b0",
            "home_id": f"h{index}", "cell_id": index, "slot_id": 7,
            "clock_domain": "monotonic-raw-host0",
        }
        for name, timestamp in (
            ("iq_ready", 5 + index), ("release", 10 + index),
            ("phy_submit", 20 + index), ("crc_visible", 140 + index),
            ("fapi_publish", 145 + index), ("mac_consume", 150 + index),
            ("radio_commit", 140 + index),
        ):
            events.append(identity | {"event": name, "timestamp_ns": timestamp})
        events.append(identity | {
            "event": "nrx_outcome", "timestamp_ns": 60,
            "nrx_admitted": True, "nrx_outcome_at_decision": outcome,
        })
    return {
        "schema": "softwall-c163-raw-du-trace-v1",
        "source_type": "du-fapi-trace", "identifier": "unit-only-fixture",
        "clock_domain": "monotonic-raw-host0", "events": events,
        "decisions": [{
            "batch_id": "b0", "decision_ns": 60,
            "mandatory_request_ids": ["r0", "r1"],
            "unresolved_request_ids": ["r1"],
            "ai_context_length": 128, "ai_deadline_ns": 150,
            "ai_lease_accepted": True, "ai_complete_ns": 100,
        }],
    }


ARTIFACT_DOCUMENTS = {
    "trace.ndjson": raw_artifact(), "clock.json": CLOCK_ARTIFACT,
    "expiry.json": EXPIRY_ARTIFACT, "mode.json": MODE_ARTIFACT,
}
ARTIFACTS = {
    name: (json.dumps(value, sort_keys=True) + "\n").encode()
    for name, value in ARTIFACT_DOCUMENTS.items()
}


def reference(path):
    return {"path": path, "sha256": hashlib.sha256(ARTIFACTS[path]).hexdigest()}


def valid_document():
    records = []
    for index, outcome in enumerate(("timely_success", "failed_or_late")):
        records.append({
            "request_id": f"r{index}", "batch_id": "b0",
            "home_id": f"h{index}", "cell_id": index, "slot_id": 7,
            "clock_domain": "monotonic-raw-host0",
            "iq_ready_ns": 5 + index, "release_ns": 10 + index,
            "phy_submit_ns": 20 + index, "crc_visible_ns": 140 + index,
            "fapi_publish_ns": 145 + index, "mac_consume_ns": 150 + index,
            "expiry_ns": 200 + index, "radio_commit_count": 1,
            "nrx_admitted": True, "nrx_outcome_at_decision": outcome,
            "nrx_outcome_observed_ns": 60,
        })
    return {
        "schema": "softwall-du-timing-contract-v2",
        "contract_kind": "production-du",
        "source": {
            "type": "du-fapi-trace", "identifier": "unit-only-fixture",
            "artifact": reference("trace.ndjson"),
        },
        "clock": {
            "domain": "monotonic-raw-host0", "timestamp_unit": "ns",
            "synchronized": True,
            "calibration": {
                "method": "same-host-clock-monotonic-raw", "max_error_ns": 1,
                "artifact": reference("clock.json"),
            },
        },
        "expiry_contract": {
            "source_type": "du_configuration",
            "source_reference": "unit-only-config",
            "synthetic": False, "derived_from_ue_k2_n2": False,
            "artifact": reference("expiry.json"),
        },
        "mode": {
            "lifecycle": "warm", "node_bound_status": "qualified",
            "recovery_capacity": 1, "recovery_bound_ns": 25,
            "nrx_bound_ns": 45, "control_bound_ns": 5,
            "guard_ns": 2, "ai_class_bounds_ns": {"128": 40},
            "qualification_reference": "unit-only mode qualification",
            "qualification_artifact": reference("mode.json"),
        },
        "records": records,
        "decisions": [{
            "batch_id": "b0", "decision_ns": 60,
            "mandatory_request_ids": ["r0", "r1"],
            "unresolved_request_ids": ["r1"],
            "ai_context_length": 128, "ai_deadline_ns": 150,
            "ai_lease_accepted": True, "ai_complete_ns": 100,
        }],
    }


class TimingContractV2Test(unittest.TestCase):
    def test_valid_contract_qualifies_and_predicts_qsu(self):
        result = validate_document(valid_document(), ARTIFACTS)
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["counts"]["states"]["QSU"], 1)
        self.assertEqual(result["counts"]["model_decision_mismatches"], 0)

    def test_artifact_content_must_match_hash(self):
        artifacts = dict(ARTIFACTS)
        artifacts["trace.ndjson"] = b"changed\n"
        result = validate_document(valid_document(), artifacts)
        self.assertFalse(result["gates"]["production_trace_provenance"])

    def test_contract_cannot_change_bound_without_changing_mode_artifact(self):
        value = valid_document()
        value["mode"]["recovery_bound_ns"] = 24
        result = validate_document(value, ARTIFACTS)
        self.assertFalse(result["gates"]["artifact_reconstruction_matches"])
        self.assertFalse(result["trace_valid"])

    def test_synthetic_or_k2_n2_expiry_is_rejected(self):
        for field in ("synthetic", "derived_from_ue_k2_n2"):
            value = valid_document()
            value["expiry_contract"][field] = True
            with self.subTest(field=field):
                self.assertFalse(validate_document(
                    value, ARTIFACTS
                )["gates"]["explicit_gnb_expiry_evidence"])

    def test_clock_uncertainty_is_applied_to_observed_deadline(self):
        value = valid_document()
        value["records"][0]["mac_consume_ns"] = 200
        result = validate_document(value, ARTIFACTS)
        self.assertFalse(result["gates"][
            "observed_mac_before_expiry_with_clock_error"
        ])

    def test_mandatory_coverage_cannot_omit_a_request(self):
        value = valid_document()
        value["decisions"][0]["mandatory_request_ids"] = ["r1"]
        result = validate_document(value, ARTIFACTS)
        self.assertFalse(result["gates"]["mandatory_and_unresolved_coverage"])

    def test_unresolved_set_is_derived_from_observed_outcomes(self):
        value = valid_document()
        value["decisions"][0]["unresolved_request_ids"] = []
        result = validate_document(value, ARTIFACTS)
        self.assertFalse(result["gates"]["mandatory_and_unresolved_coverage"])

    def test_nrx_outcome_must_be_observed_before_decision(self):
        value = valid_document()
        value["records"][0]["nrx_outcome_observed_ns"] = 61
        result = validate_document(value, ARTIFACTS)
        self.assertFalse(result["gates"]["nrx_outcomes_visible_by_decision"])

    def test_all_fail_infeasible_batch_is_not_qualified(self):
        value = valid_document()
        value["records"] = []
        for index in range(5):
            row = copy.deepcopy(valid_document()["records"][1])
            row.update({
                "request_id": f"r{index}", "home_id": f"h{index}",
                "cell_id": index, "release_ns": 10, "phy_submit_ns": 20,
                "crc_visible_ns": 120, "fapi_publish_ns": 125,
                "mac_consume_ns": 130, "expiry_ns": 150,
            })
            value["records"].append(row)
        ids = [row["request_id"] for row in value["records"]]
        value["decisions"][0].update({
            "mandatory_request_ids": ids, "unresolved_request_ids": ids,
            "ai_lease_accepted": False, "ai_complete_ns": None,
        })
        result = validate_document(value, ARTIFACTS)
        self.assertFalse(result["gates"]["verified_all_fail_certificate"])
        self.assertEqual(result["counts"]["states"]["MI"], 1)

    def test_observed_ai_decision_must_match_certificate(self):
        value = valid_document()
        value["decisions"][0].update({
            "ai_lease_accepted": False, "ai_complete_ns": None,
        })
        result = validate_document(value, ARTIFACTS)
        self.assertFalse(result["gates"]["model_matches_physical_ai_decision"])
        self.assertEqual(result["counts"]["model_decision_mismatches"], 1)

    def test_ai_completion_cannot_precede_decision(self):
        value = valid_document()
        value["decisions"][0]["ai_complete_ns"] = 59
        result = validate_document(value, ARTIFACTS)
        self.assertFalse(result["gates"]["physical_ai_outcome_fields"])

    def test_artifact_loader_rejects_parent_escape(self):
        value = valid_document()
        value["source"]["artifact"]["path"] = "../outside.ndjson"
        with self.assertRaisesRegex(ValueError, "escapes artifact root"):
            load_artifacts(value, Path(__file__).resolve().parent)


if __name__ == "__main__":
    unittest.main()
