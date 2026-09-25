#!/usr/bin/env python3.11
"""Run C163 contract tests and emit a hash-pinned readiness artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest


TEST_MODULES = (
    "test_c163_build_du_contract_v2",
    "test_c163_production_envelope_bridge",
    "test_du_timing_contract_v2",
)
FILES = (
    "scripts_for_node/softwall_same_gpu/build_c163_timing_readiness_v2.py",
    "scripts_for_node/softwall_same_gpu/c163_build_du_contract_v2.py",
    "scripts_for_node/softwall_same_gpu/c163_production_envelope_bridge.py",
    "scripts_for_node/softwall_same_gpu/c162_certified_scheduler_v1.py",
    "scripts_for_node/softwall_same_gpu/du_timing_contract_v2.py",
    "scripts_for_node/softwall_same_gpu/shared_recovery_certificate_v1.py",
    "scripts_for_node/softwall_same_gpu/test_c163_build_du_contract_v2.py",
    "scripts_for_node/softwall_same_gpu/test_c163_production_envelope_bridge.py",
    "scripts_for_node/softwall_same_gpu/test_du_timing_contract_v2.py",
    "results/softwall_multigpu/softwall_du_timing_contract_template_v2.json",
    "results/softwall_multigpu/c163_raw_du_trace_template_v1.json",
    "results/softwall_multigpu/c163_clock_calibration_template_v1.json",
    "results/softwall_multigpu/c163_expiry_contract_template_v1.json",
    "results/softwall_multigpu/c163_mode_qualification_template_v1.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    suite = unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
    count = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    output = {
        "schema": "softwall-c163-du-timing-readiness-v2",
        "status": "UQ_NO_PRODUCTION_TRACE",
        "verification": {"unit_tests": count, "unit_tests_passed": count},
        "ready_components": {
            "raw_event_to_contract_joiner": True,
            "content_hash_provenance_for_trace_clock_expiry_and_mode": True,
            "artifact_to_contract_reconstruction_gate": True,
            "calibrated_clock_uncertainty_gate": True,
            "request_specific_verified_all_fail_certificate": True,
            "qsu_qsn_mi_uq_production_bridge": True,
            "physical_ai_lease_decision_parity_gate": True,
            "mac_consumption_deadline_gate_with_clock_error": True,
            "single_commit_and_nrx_outcome_consistency": True,
            "nrx_outcome_visible_before_decision_gate": True,
        },
        "missing_evidence": [
            "target DU/FAPI or DU/MAC raw event trace",
            "explicit target gNB MAC expiry artifact",
            "clock calibration artifact for every timestamp producer",
            "same-topology whole-path NRx/recovery/control/AI mode qualification",
            "independent-node production holdout",
        ],
        "files_sha256": {name: sha256(root / name) for name in FILES},
        "claim_boundary": (
            "The C163 ingestion and decision-audit path is executable and unit "
            "tested. It does not qualify a production DU mode until the listed "
            "external artifacts are captured and pass the frozen validator."
        ),
    }
    path = root / "results/softwall_multigpu/c163_du_timing_readiness_v2.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": output["status"], "unit_tests": count}, indent=2))


if __name__ == "__main__":
    main()
