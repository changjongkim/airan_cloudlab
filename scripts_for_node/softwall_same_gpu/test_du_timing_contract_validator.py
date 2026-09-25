import copy
import unittest

from du_timing_contract_validator import validate_document


def valid_document():
    return {
        "schema": "softwall-du-timing-contract-v1",
        "contract_kind": "production-du",
        "source": {
            "type": "du-fapi-trace",
            "identifier": "unit-only-fixture",
            "artifact_sha256": "0" * 64,
        },
        "clock": {
            "domain": "clock-monotonic-raw",
            "timestamp_unit": "ns",
            "synchronized": True,
        },
        "expiry_contract": {
            "source_type": "du_configuration",
            "source_reference": "unit-only-config",
            "synthetic": False,
            "derived_from_ue_k2_n2": False,
        },
        "mandatory_contract": {
            "home_cell_counts": [2],
            "recovery_bound_ns": 20,
            "guard_ns": 5,
        },
        "records": [
            {
                "request_id": "r0",
                "cell_id": 0,
                "slot_id": 100,
                "clock_domain": "clock-monotonic-raw",
                "iq_ready_ns": 100,
                "release_ns": 110,
                "phy_submit_ns": 120,
                "crc_visible_ns": 130,
                "fapi_publish_ns": 140,
                "mac_consume_ns": 150,
                "expiry_ns": 200,
                "radio_commit_count": 1,
            },
            {
                "request_id": "r1",
                "cell_id": 1,
                "slot_id": 100,
                "clock_domain": "clock-monotonic-raw",
                "iq_ready_ns": 101,
                "release_ns": 111,
                "phy_submit_ns": 121,
                "crc_visible_ns": 131,
                "fapi_publish_ns": 141,
                "mac_consume_ns": 151,
                "expiry_ns": 201,
                "radio_commit_count": 1,
            },
        ],
    }


class TimingContractValidatorTest(unittest.TestCase):
    def test_valid_contract_qualifies(self):
        self.assertTrue(validate_document(valid_document())["all_pass"])

    def test_synthetic_contract_is_rejected(self):
        value = valid_document()
        value["expiry_contract"]["synthetic"] = True
        self.assertFalse(validate_document(value)["gates"]["explicit_gnb_expiry_contract"])

    def test_ue_k2_n2_derivation_is_rejected(self):
        value = valid_document()
        value["expiry_contract"]["derived_from_ue_k2_n2"] = True
        self.assertFalse(validate_document(value)["gates"]["explicit_gnb_expiry_contract"])

    def test_clock_mismatch_is_rejected(self):
        value = valid_document()
        value["records"][0]["clock_domain"] = "other-clock"
        self.assertFalse(validate_document(value)["gates"]["record_clock_matches"])

    def test_duplicate_request_is_rejected(self):
        value = valid_document()
        value["records"][1]["request_id"] = "r0"
        self.assertFalse(validate_document(value)["gates"]["unique_request_ids"])

    def test_nonmonotonic_timestamps_are_rejected(self):
        value = valid_document()
        value["records"][0]["fapi_publish_ns"] = 125
        self.assertFalse(validate_document(value)["gates"]["timestamp_order"])

    def test_multiple_radio_commits_are_rejected(self):
        value = valid_document()
        value["records"][0]["radio_commit_count"] = 2
        self.assertFalse(validate_document(value)["gates"]["single_radio_commit"])

    def test_observed_deadline_miss_is_rejected(self):
        value = valid_document()
        value["records"][0]["mac_consume_ns"] = 210
        self.assertFalse(validate_document(value)["gates"]["observed_mac_consumption_before_expiry"])

    def test_mandatory_capacity_failure_is_rejected(self):
        value = valid_document()
        value["mandatory_contract"]["recovery_bound_ns"] = 50
        self.assertFalse(validate_document(value)["gates"]["conservative_all_fail_capacity"])


if __name__ == "__main__":
    unittest.main()
