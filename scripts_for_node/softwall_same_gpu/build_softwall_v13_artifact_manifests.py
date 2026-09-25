#!/usr/bin/env python3
"""Build immutable hash manifests for C138--C140 and the V13 correction chain."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_hashes(root, paths):
    unique = sorted({path.resolve() for path in paths})
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise ValueError("missing artifact(s): " + ", ".join(missing))
    return {str(path.relative_to(root)): sha256(path) for path in unique}


def protocol_sources(root, protocol):
    document = json.loads(protocol.read_text(encoding="utf-8"))
    frozen = document["source_sha256"]
    mismatches = []
    paths = []
    for relative, expected in frozen.items():
        path = root / relative
        paths.append(path)
        actual = sha256(path)
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    return paths, mismatches


def matching(root, prefix):
    result_dir = root / "results/softwall_multigpu"
    paths = list(result_dir.glob(prefix + "*")) + list((result_dir / "raw").glob(prefix + "*"))
    return [path for path in paths if not path.name.endswith("_artifact_manifest.json")]


def write_manifest(path, payload):
    payload["formal_artifact_count"] = len(payload["files"])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[2]
    result_dir = root / "results/softwall_multigpu"
    docs = root / "docs/current"
    script_dir = root / "scripts_for_node/softwall_same_gpu"

    specs = {
        "confirm138": {
            "protocol": result_dir / "confirm138_service_bound_fault_telemetry_protocol.json",
            "status": "EXPECTED_TIMING_FAILURE_RETAINED",
            "schema": "softwall-confirm138-artifact-manifest-v1",
            "output": result_dir / "confirm138_artifact_manifest.json",
            "extras": [docs / "SOFTWALL_CONFIRM138_140_CONTROL_BOUND_CORRECTION_KO.md"],
            "integrity": {"campaign_all_pass": False, "expected_failed_gate": "rpc_wall_bound"},
            "scope": {"job_id": 58831304, "node": "nid001144", "arms_run": 1,
                      "radio_records": 2720, "rpc_records": 351, "five_ms_exceedances": 2},
        },
        "confirm139": {
            "protocol": result_dir / "confirm139_qualified_control_bound_protocol.json",
            "status": "PASS",
            "schema": "softwall-confirm139-artifact-manifest-v1",
            "output": result_dir / "confirm139_artifact_manifest.json",
            "extras": [docs / "SOFTWALL_CONFIRM138_140_CONTROL_BOUND_CORRECTION_KO.md"],
            "integrity": {"campaign_all_pass": True, "admission_bound_exceedances": 0},
            "scope": {"job_id": 58831575, "node": "nid001252", "arms": 6,
                      "radio_records": 16320, "rpc_records": 2020,
                      "maximum_rpc_wall_ms": 5.653235, "admission_bound_ms": 7.0},
        },
        "confirm140": {
            "protocol": result_dir / "confirm140_v13_ai35_protocol.json",
            "status": "PASS",
            "schema": "softwall-confirm140-artifact-manifest-v1",
            "output": result_dir / "confirm140_artifact_manifest.json",
            "extras": [
                docs / "SOFTWALL_CONFIRM138_140_CONTROL_BOUND_CORRECTION_KO.md",
                root / "data/current/softwall_context64_mechanism_trace_v1.json",
            ],
            "integrity": {"campaign_all_pass": True, "conditional_exchanges": 8},
            "scope": {"job_id": 58831575, "node": "nid001252", "arms": 2,
                      "radio_records": 2560, "effective_transaction_bound_ms": 58.0,
                      "static_margin_ms": -5.0, "conditional_margin_ms": 0.0},
        },
    }

    for prefix, spec in specs.items():
        sources, mismatches = protocol_sources(root, spec["protocol"])
        if mismatches:
            raise ValueError("frozen source mismatch: " + json.dumps(mismatches, sort_keys=True))
        paths = matching(root, prefix) + sources + spec["extras"]
        payload = {
            "schema": spec["schema"],
            "status": spec["status"],
            "campaign_integrity": dict(spec["integrity"], source_hash_mismatches=mismatches),
            "physical_scope": spec["scope"],
            "claim_boundary": (
                "Finite-sample A100/MPS evidence for the C138--C140 control-bound "
                "correction chain; not WCET, production d_MAC, cross-family, "
                "independent-node C140, or throughput-superiority evidence."
            ),
            "files": relative_hashes(root, paths),
        }
        write_manifest(spec["output"], payload)

    final_paths = [
        specs[key]["output"] for key in ("confirm138", "confirm139", "confirm140")
    ] + [
        root / "data/README.md",
        root / "data/current/softwall_context64_mechanism_trace_v1.json",
        docs / "CURRENT_RESEARCH_INDEX_KO.md",
        docs / "SOFTWALL_COMPLETION_AUDIT_KO.md",
        docs / "SOFTWALL_CONFIRM138_140_CONTROL_BOUND_CORRECTION_KO.md",
        docs / "SOFTWALL_CURRENT_SCHEME_AND_EVIDENCE_KO.md",
        docs / "SOFTWALL_ENDPOINT_OFFLOAD_ENVELOPE_RESULT_KO.md",
        docs / "SOFTWALL_FORMAL_MODEL_KO.md",
        docs / "SOFTWALL_MULTIGPU_MODELING_ROADMAP_KO.md",
        docs / "SOFTWALL_PAPER_STRUCTURE_KO.md",
        docs / "SOFTWALL_RELATED_WORK_AUDIT_KO.md",
        docs / "SOFTWALL_SERVICE_BOUND_QUALIFICATION_KO.md",
        root / "results/softwall_same_gpu/EXPERIMENT_GATE_LEDGER_KO.md",
        result_dir / "confirm138_prepare1_job58831304_result.json",
        result_dir / "confirm138_service_bound_fault_telemetry_result.json",
        result_dir / "confirm139_qualified_control_bound_result.json",
        result_dir / "confirm140_v13_ai35_result.json",
        result_dir / "softwall_envelope_v12_v13_control_bound_regression_v1.json",
        result_dir / "softwall_envelope_v13_validation_summary.json",
        result_dir / "softwall_service_bound_qualification_v3.json",
        result_dir / "softwall_sharded_home_envelope_grid_v13.json",
        result_dir / "softwall_sharded_home_envelope_prediction_v13.json",
        script_dir / "analyze_softwall_service_bound_qualification_v3.py",
        script_dir / "test_analyze_softwall_service_bound_qualification_v3.py",
        script_dir / "build_softwall_v13_artifact_manifests.py",
        script_dir / "build_softwall_envelope_v13.py",
        script_dir / "test_build_softwall_envelope_v13.py",
        script_dir / "compare_softwall_envelope_v12_v13.py",
        script_dir / "analyze_softwall_envelope_v13_validation.py",
    ]
    final = {
        "schema": "softwall-v13-corrected-control-manifest-v1",
        "status": "CORRECTED_CONTROL_CONTRACT_PASS",
        "classification_counts": {"QSU": 5, "QSN": 0, "MI": 3, "UQ": 11},
        "correction_chain": {
            "c138": "5 ms timeout-equals-wall-bound assumption falsified",
            "c139": "7 ms per-RPC admission bound passed six fault arms",
            "c140": "AI35+control21+guard2 conditional class passed two physical arms",
        },
        "remaining_gates": [
            "independent-node corrected V13 requalification",
            "deterministic or explicitly probabilistic control-path bound",
            "production DU d_MAC",
            "cross-family hardware qualification",
        ],
        "claim_boundary": (
            "Current authoritative finite-sample V13 artifact set. Historical V12 "
            "physical executions remain mechanism evidence, but the 15 ms control-time "
            "contract is superseded."
        ),
        "files": relative_hashes(root, final_paths),
    }
    write_manifest(result_dir / "softwall_v13_corrected_control_manifest.json", final)


if __name__ == "__main__":
    main()
