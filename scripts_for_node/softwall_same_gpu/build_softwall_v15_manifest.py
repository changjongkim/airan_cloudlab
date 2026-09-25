#!/usr/bin/env python3
"""Build the immutable C145-C146 V15 single-token artifact manifest."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_hashes(root, paths):
    unique = sorted({Path(path).resolve() for path in paths})
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise ValueError("missing artifact(s): " + ", ".join(missing))
    return {str(path.relative_to(root)): sha256(path) for path in unique}


def protocol_sources(root, protocol):
    document = json.loads(Path(protocol).read_text(encoding="utf-8"))
    paths = []
    mismatches = []
    for relative, expected in document["source_sha256"].items():
        path = root / relative
        paths.append(path)
        actual = sha256(path)
        if actual != expected:
            mismatches.append({
                "path": relative, "expected": expected, "actual": actual,
            })
    return paths, mismatches


def campaign_files(root, prefixes, output_name):
    result_dir = root / "results/softwall_multigpu"
    paths = []
    for directory in (result_dir, result_dir / "raw"):
        for prefix in prefixes:
            paths.extend(directory.glob(prefix + "*"))
    return [path for path in paths if path.name != output_name]


def require_result(path, expected_node):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != "softwall-v15-single-token-node-requalification-v1":
        raise ValueError("unexpected result schema: " + str(path))
    if document.get("all_pass") is not True:
        raise ValueError("campaign did not pass: " + str(path))
    if document.get("nodes") != [expected_node]:
        raise ValueError("unexpected node in {}: {}".format(path, document.get("nodes")))
    if document["totals"]["maximum_unlaunched_tokens"] != 1:
        raise ValueError("ownership maximum is not one: " + str(path))
    return document


def build(root):
    root = Path(root).resolve()
    result_dir = root / "results/softwall_multigpu"
    script_dir = root / "scripts_for_node/softwall_same_gpu"
    output_name = "softwall_v15_single_token_manifest.json"
    protocols = [
        result_dir / "confirm145_v15_single_token_protocol.json",
        result_dir / "confirm146_v15_single_token_protocol.json",
    ]
    source_paths = []
    source_mismatches = []
    for protocol in protocols:
        paths, mismatches = protocol_sources(root, protocol)
        source_paths.extend(paths)
        source_mismatches.extend(mismatches)
    if source_mismatches:
        raise ValueError("frozen source mismatch: "
                         + json.dumps(source_mismatches, sort_keys=True))

    c145 = require_result(
        result_dir / "confirm145_v15_single_token_result.json", "nid001244")
    c146 = require_result(
        result_dir / "confirm146_v15_single_token_result.json", "nid003417")
    model = json.loads((result_dir / "softwall_pipelined_control_model_v2.json").read_text())
    regression = json.loads((result_dir / "softwall_v14_v15_staged_ownership_regression_v1.json").read_text())
    reproducibility = json.loads((result_dir / "softwall_pipelined_control_model_v2_reproducibility_v1.json").read_text())
    validation = json.loads((result_dir / "softwall_envelope_v15_validation_summary.json").read_text())
    if (not model.get("all_pass") or not regression.get("all_pass")
            or not reproducibility.get("all_pass")):
        raise ValueError("model, reproducibility, or deterministic regression did not pass")
    if validation.get("status") != "PASS":
        raise ValueError("V15 envelope validation did not pass")

    node_results = (c145, c146)
    observed = {
        "nodes": sorted(c145["nodes"] + c146["nodes"]),
        "radio_records": sum(row["totals"]["radio_records"] for row in node_results),
        "ai45_exchanges": sum(row["totals"]["ai45_exchanges"] for row in node_results),
        "fault_arms": sum(row["totals"]["fault_arms"] for row in node_results),
        "radio_after_fault": sum(row["totals"]["radio_after_fault"] for row in node_results),
        "suppressed_prepares": sum(
            row["totals"]["suppressed_prepare_due_owned_token"]
            for row in node_results
        ),
        "maximum_unlaunched_tokens": max(
            row["totals"]["maximum_unlaunched_tokens"] for row in node_results
        ),
    }
    expected = {
        "nodes": ["nid001244", "nid003417"],
        "radio_records": 12160,
        "ai45_exchanges": 12,
        "fault_arms": 6,
        "radio_after_fault": 7463,
        "suppressed_prepares": 537,
        "maximum_unlaunched_tokens": 1,
    }
    if observed != expected:
        raise ValueError("authoritative count mismatch: {} != {}".format(
            observed, expected))

    extra_scripts = [
        script_dir / "analyze_softwall_envelope_v15_validation.py",
        script_dir / "build_softwall_envelope_v15.py",
        script_dir / "softwall_envelope_checker_v10.py",
        script_dir / "test_build_softwall_envelope_v15.py",
        script_dir / "test_softwall_envelope_checker_v10.py",
        script_dir / "test_analyze_softwall_envelope_v15_validation.py",
        script_dir / "audit_pipelined_control_model_v2_reproducibility.py",
        script_dir / "test_audit_pipelined_control_model_v2_reproducibility.py",
        script_dir / "build_softwall_v15_manifest.py",
        script_dir / "test_build_softwall_v15_manifest.py",
    ]
    immutable_inputs = [
        root / "data/current/softwall_context64_mechanism_trace_v1.json",
        result_dir / "softwall_pipelined_control_model_v2.json",
        result_dir / "softwall_v14_v15_staged_ownership_regression_v1.json",
        result_dir / "softwall_pipelined_control_model_v2_reproducibility_v1.json",
        result_dir / "softwall_sharded_home_envelope_grid_v15.json",
        result_dir / "softwall_sharded_home_envelope_prediction_v15.json",
        result_dir / "softwall_envelope_v15_validation_summary.json",
    ]
    files = campaign_files(root, ("confirm145", "confirm146"), output_name)
    files.extend(protocols + source_paths + extra_scripts + immutable_inputs)

    payload = {
        "schema": "softwall-v15-single-token-manifest-v1",
        "status": "TWO_NODE_SINGLE_TOKEN_PIPELINED_CONTROL_PASS",
        "supersedes_for_current_claim": (
            "V14 physical evidence remains valid for observed traces, but its "
            "envelope mode is UQ because the retained-token ownership branch "
            "was not gated. V15 closes and physically exercises that branch."
        ),
        "campaign_integrity": {
            "source_hash_mismatches": source_mismatches,
            "result_gates_pass": True,
            "finite_model_pass": True,
            "finite_model_semantic_reproducibility_pass": True,
            "deterministic_regression_pass": True,
            "envelope_validation_pass": True,
        },
        "classification_counts": validation["model_counts"],
        "physical_scope": dict(observed, declared_safety_or_duplicate_violations=0),
        "finite_model": validation["finite_model"],
        "claim_boundary": validation["claim_boundary"],
        "mutable_documents_excluded": True,
        "files": relative_hashes(root, files),
    }
    payload["formal_artifact_count"] = len(payload["files"])
    return payload


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "results/softwall_multigpu/softwall_v15_single_token_manifest.json"
    output.write_text(json.dumps(build(root), indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
