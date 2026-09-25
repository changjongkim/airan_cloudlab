#!/usr/bin/env python3
"""Build the immutable C142--C144 V14 pipelined-control artifact manifest."""

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
    paths = []
    mismatches = []
    for relative, expected in document["source_sha256"].items():
        path = root / relative
        paths.append(path)
        actual = sha256(path)
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    return paths, mismatches


def campaign_files(root, prefixes, output_name):
    result_dir = root / "results/softwall_multigpu"
    paths = []
    for directory in (result_dir, result_dir / "raw"):
        for prefix in prefixes:
            paths.extend(directory.glob(prefix + "*"))
    return [path for path in paths if path.name != output_name]


def require_result(path, schema, expected_node):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != schema:
        raise ValueError("unexpected schema in {}: {}".format(path, document.get("schema")))
    if document.get("all_pass") is not True:
        raise ValueError("campaign did not pass: {}".format(path))
    if document.get("nodes") != [expected_node]:
        raise ValueError("unexpected node in {}: {}".format(path, document.get("nodes")))
    return document


def build(root):
    result_dir = root / "results/softwall_multigpu"
    script_dir = root / "scripts_for_node/softwall_same_gpu"
    output_name = "softwall_v14_pipelined_control_manifest.json"

    protocols = [
        result_dir / "confirm142_v14_pipelined_protocol.json",
        result_dir / "confirm143_v14_control_faults_protocol.json",
        result_dir / "confirm144_v14_independent_node_protocol.json",
    ]
    source_paths = []
    source_mismatches = []
    for protocol in protocols:
        paths, mismatches = protocol_sources(root, protocol)
        source_paths.extend(paths)
        source_mismatches.extend(mismatches)
    if source_mismatches:
        raise ValueError("frozen source mismatch: " + json.dumps(source_mismatches, sort_keys=True))

    c142 = require_result(
        result_dir / "confirm142_v14_pipelined_result.json",
        "softwall-confirm142-v14-pipelined-ai45-v1",
        "nid001361",
    )
    c143 = require_result(
        result_dir / "confirm143_v14_control_faults_result.json",
        "softwall-confirm143-v14-pipelined-control-faults-v1",
        "nid001361",
    )
    c144 = require_result(
        result_dir / "confirm144_v14_independent_node_result.json",
        "softwall-confirm144-v14-independent-node-v1",
        "nid002049",
    )
    model = json.loads((result_dir / "softwall_pipelined_control_model_v1.json").read_text(encoding="utf-8"))
    validation = json.loads((result_dir / "softwall_envelope_v14_validation_summary.json").read_text(encoding="utf-8"))
    if model.get("all_pass") is not True or model.get("violations") != []:
        raise ValueError("finite model is not clean")
    if validation.get("status") != "PASS":
        raise ValueError("V14 envelope validation did not pass")

    expected = {
        "c142_ai45_exchanges": 9,
        "c143_fault_arms": 6,
        "c144_ai45_exchanges": 2,
        "c144_fault_arms": 3,
        "finite_states": 16,
        "finite_edges": 20,
    }
    observed = {
        "c142_ai45_exchanges": c142["totals"]["ai45_candidate_exchanges"],
        "c143_fault_arms": c143["totals"]["arms"],
        "c144_ai45_exchanges": c144["totals"]["ai45_exchanges"],
        "c144_fault_arms": c144["totals"]["fault_arms"],
        "finite_states": model["states"],
        "finite_edges": model["edges"],
    }
    if observed != expected:
        raise ValueError("authoritative count mismatch: {} != {}".format(observed, expected))

    extra_scripts = [
        script_dir / "analyze_confirm142_v14_pipelined.py",
        script_dir / "analyze_confirm143_v14_control_faults.py",
        script_dir / "analyze_confirm144_v14_independent_node.py",
        script_dir / "analyze_softwall_envelope_v14_validation.py",
        script_dir / "build_softwall_envelope_v14.py",
        script_dir / "softwall_envelope_checker_v9.py",
        script_dir / "test_build_softwall_envelope_v14.py",
        script_dir / "test_global_trace_pipelined_control_controller.py",
        script_dir / "test_pipelined_global_trace_client.py",
        script_dir / "test_verify_pipelined_control_model.py",
        script_dir / "verify_pipelined_control_model.py",
        script_dir / "run_confirm142_v14_pipelined.sh",
        script_dir / "run_confirm143_v14_control_faults.sh",
        script_dir / "run_confirm144_v14_independent_node.sh",
        script_dir / "build_softwall_v14_manifest.py",
        script_dir / "test_build_softwall_v14_manifest.py",
    ]
    immutable_inputs = [
        root / "data/current/softwall_context64_mechanism_trace_v1.json",
        result_dir / "softwall_pipelined_control_model_v1.json",
        result_dir / "softwall_sharded_home_envelope_grid_v14.json",
        result_dir / "softwall_sharded_home_envelope_prediction_v14.json",
        result_dir / "softwall_envelope_v14_validation_summary.json",
    ]
    files = campaign_files(root, ("confirm142", "confirm143", "confirm144"), output_name)
    files.extend(protocols + source_paths + extra_scripts + immutable_inputs)

    payload = {
        "schema": "softwall-v14-pipelined-control-manifest-v1",
        "status": "TWO_NODE_PIPELINED_CONTROL_PASS",
        "campaign_integrity": {
            "source_hash_mismatches": source_mismatches,
            "result_gates_pass": True,
            "finite_model_pass": True,
            "envelope_validation_pass": True,
        },
        "classification_counts": validation["model_counts"],
        "physical_scope": {
            "nodes": ["nid001361", "nid002049"],
            "ai45_arms": 3,
            "ai45_radio_records": c142["totals"]["radio_records"] + 1280,
            "ai45_candidate_exchanges": 11,
            "control_fault_arms": 9,
            "control_fault_radio_records": c143["totals"]["radio_records"] + 4800,
            "radio_records_after_fault": c143["totals"]["radio_records_after_fault"] + c144["totals"]["radio_after_fault"],
            "c143_commit_records": c143["totals"]["commit_records"],
            "c143_maximum_commit_ms": c143["totals"]["maximum_commit_ms"],
            "c143_deferred_over_7ms": c143["totals"]["deferred_over_7ms"],
            "declared_safety_or_duplicate_violations": 0,
        },
        "finite_model": {
            "states": model["states"],
            "edges": model["edges"],
            "terminal_states": model["terminal_states"],
            "violations": len(model["violations"]),
        },
        "claim_boundary": (
            "Two-node A100-family finite-sample evidence for V14 pipelined global control, "
            "AI45 conditional exchange, and three control fault points. This is not WCET, "
            "production d_MAC, cross-family, durable-restart, or throughput-superiority evidence."
        ),
        "mutable_documents_excluded": True,
        "files": relative_hashes(root, files),
    }
    payload["formal_artifact_count"] = len(payload["files"])
    return payload


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "results/softwall_multigpu/softwall_v14_pipelined_control_manifest.json"
    output.write_text(json.dumps(build(root), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
