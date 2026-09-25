#!/usr/bin/env python3
"""Build the C141 and current two-node corrected V13 hash manifests."""

import json
from pathlib import Path

from build_softwall_v13_artifact_manifests import (
    matching,
    protocol_sources,
    relative_hashes,
    write_manifest,
)


def main():
    root = Path(__file__).resolve().parents[2]
    result_dir = root / "results/softwall_multigpu"
    docs = root / "docs/current"
    scripts = root / "scripts_for_node/softwall_same_gpu"
    protocols = [
        result_dir / "confirm141_control_protocol.json",
        result_dir / "confirm141_ai35_protocol.json",
    ]
    sources = []
    mismatches = []
    for protocol in protocols:
        current_sources, current_mismatches = protocol_sources(root, protocol)
        sources.extend(current_sources)
        mismatches.extend(current_mismatches)
    if mismatches:
        raise ValueError("frozen source mismatch: " + json.dumps(mismatches, sort_keys=True))

    c141_paths = matching(root, "confirm141") + sources + [
        docs / "SOFTWALL_CONFIRM141_V13_INDEPENDENT_RESULT_KO.md",
        scripts / "analyze_confirm141_v13_independent_requalification.py",
        scripts / "test_analyze_confirm141_v13_independent_requalification.py",
        scripts / "run_confirm141_v13_independent_requalification.sh",
        scripts / "normalize_confirm141_subcampaign.py",
        scripts / "test_normalize_confirm141_subcampaign.py",
        root / "data/current/softwall_context64_mechanism_trace_v1.json",
    ]
    c141 = {
        "schema": "softwall-confirm141-artifact-manifest-v1",
        "status": "PASS",
        "campaign_integrity": {
            "combined_all_pass": True,
            "source_hash_mismatches": mismatches,
            "independent_from_corrected_node": "nid001372 != nid001252",
        },
        "physical_scope": {
            "job_id": 58832462,
            "node": "nid001372",
            "a100_gpus": 4,
            "control_arms": 6,
            "ai35_arms": 2,
            "radio_records": 18880,
            "attempted_fault_campaign_rpcs": 2010,
            "maximum_rpc_wall_ms": 5.775660,
            "ai35_candidate_exchanges": 7,
        },
        "claim_boundary": (
            "Independent-node A100-family finite-sample requalification of the "
            "corrected V13 control bound and AI35 class; not WCET, production d_MAC, "
            "cross-family, or throughput-superiority evidence."
        ),
        "files": relative_hashes(root, c141_paths),
    }
    c141_manifest = result_dir / "confirm141_artifact_manifest.json"
    write_manifest(c141_manifest, c141)

    current_paths = [
        result_dir / "confirm138_artifact_manifest.json",
        result_dir / "confirm139_artifact_manifest.json",
        result_dir / "confirm140_artifact_manifest.json",
        c141_manifest,
        result_dir / "confirm141_v13_independent_requalification_result.json",
        result_dir / "softwall_service_bound_qualification_v3.json",
        result_dir / "softwall_service_bound_qualification_v4.json",
        result_dir / "softwall_envelope_v13_validation_summary.json",
        result_dir / "softwall_sharded_home_envelope_grid_v13.json",
        result_dir / "softwall_sharded_home_envelope_prediction_v13.json",
        root / "results/softwall_same_gpu/EXPERIMENT_GATE_LEDGER_KO.md",
        root / "data/README.md",
        docs / "CURRENT_RESEARCH_INDEX_KO.md",
        docs / "SOFTWALL_COMPLETION_AUDIT_KO.md",
        docs / "SOFTWALL_CONFIRM138_140_CONTROL_BOUND_CORRECTION_KO.md",
        docs / "SOFTWALL_CONFIRM141_V13_INDEPENDENT_RESULT_KO.md",
        docs / "SOFTWALL_CURRENT_SCHEME_AND_EVIDENCE_KO.md",
        docs / "SOFTWALL_ENDPOINT_OFFLOAD_ENVELOPE_RESULT_KO.md",
        docs / "SOFTWALL_FORMAL_MODEL_KO.md",
        docs / "SOFTWALL_MULTIGPU_MODELING_ROADMAP_KO.md",
        docs / "SOFTWALL_PAPER_STRUCTURE_KO.md",
        docs / "SOFTWALL_RELATED_WORK_AUDIT_KO.md",
        docs / "SOFTWALL_SERVICE_BOUND_QUALIFICATION_KO.md",
        scripts / "analyze_softwall_service_bound_qualification_v4.py",
        scripts / "test_analyze_softwall_service_bound_qualification_v4.py",
        scripts / "build_softwall_v13_cross_node_manifest.py",
    ]
    current = {
        "schema": "softwall-v13-cross-node-manifest-v1",
        "status": "TWO_NODE_CORRECTED_CONTROL_CONTRACT_PASS",
        "classification_counts": {"QSU": 5, "QSN": 0, "MI": 3, "UQ": 11},
        "qualification": {
            "nodes": ["nid001252", "nid001372"],
            "control_fault_arms": 12,
            "attempted_fault_campaign_rpcs": 4030,
            "maximum_rpc_wall_ms": 5.775660,
            "seven_ms_exceedances": 0,
            "ai35_arms": 4,
            "ai35_candidate_exchanges": 15,
            "declared_safety_violations": 0,
        },
        "remaining_gates": [
            "deterministic or explicitly probabilistic control-path bound",
            "production DU d_MAC",
            "cross-family hardware qualification",
        ],
        "claim_boundary": (
            "Current two-node A100-family finite-sample V13 artifact set. It does "
            "not establish WCET, production timing, or cross-family generalization."
        ),
        "files": relative_hashes(root, current_paths),
    }
    write_manifest(result_dir / "softwall_v13_cross_node_manifest.json", current)


if __name__ == "__main__":
    main()
