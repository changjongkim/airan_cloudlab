#!/usr/bin/env python3
"""Build the immutable V16 extension manifest over the sealed V15 base."""

import json
from pathlib import Path

from build_softwall_v15_manifest import relative_hashes, sha256


def protocol_sources(root, protocol):
    """Resolve both workspace-relative and frozen container-mount source keys."""
    document = json.loads(Path(protocol).read_text(encoding="utf-8"))
    paths = []
    mismatches = []
    mounts = {
        "/softwall/": root / "scripts_for_node/softwall_same_gpu",
        "/softwall_task1/": root / "scripts_for_node/task1",
        "/softwall_runtime/": root / "runtime/softwall_same_gpu",
    }
    for source_name, expected in document["source_sha256"].items():
        if source_name.startswith("/"):
            matches = [
                (prefix, base) for prefix, base in mounts.items()
                if source_name.startswith(prefix)
            ]
            if len(matches) != 1:
                raise ValueError("unknown frozen mount path: " + source_name)
            prefix, base = matches[0]
            path = base / source_name[len(prefix):]
        else:
            path = root / source_name
        paths.append(path)
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches.append({
                "path": source_name, "expected": expected, "actual": actual,
            })
    return paths, mismatches


def verify_base_manifest(root, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    mismatches = []
    for relative, expected in manifest["files"].items():
        path = root / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches.append({
                "path": relative, "expected": expected, "actual": actual,
            })
    return manifest, mismatches


def require_node(path, schema, expected_node):
    value = json.loads(Path(path).read_text())
    if value.get("schema") != schema or value.get("all_pass") is not True:
        raise ValueError("node result did not pass: " + str(path))
    if value.get("nodes") != [expected_node]:
        raise ValueError("unexpected node: " + str(path))
    return value


def build(root):
    root = Path(root).resolve()
    result_dir = root / "results/softwall_multigpu"
    script_dir = root / "scripts_for_node/softwall_same_gpu"
    base_path = result_dir / "softwall_v15_single_token_manifest.json"
    base, base_mismatches = verify_base_manifest(root, base_path)
    if base_mismatches:
        raise ValueError("V15 base manifest mismatch: "
                         + json.dumps(base_mismatches, sort_keys=True))

    node_specs = (
        (147, "nid003197"),
        (148, "nid001005"),
    )
    nodes = []
    files = [base_path]
    source_mismatches = []
    source_paths = []
    for number, expected_node in node_specs:
        outer_protocol = result_dir / f"confirm{number}_v16_abort_protocol.json"
        outer_result = result_dir / f"confirm{number}_v16_abort_result.json"
        node = require_node(
            outer_result, "softwall-v16-abort-node-requalification-v1",
            expected_node,
        )
        nodes.append(node)
        outer_sources, mismatches = protocol_sources(root, outer_protocol)
        source_paths.extend(outer_sources)
        source_mismatches.extend(mismatches)
        outer = json.loads(outer_protocol.read_text())
        label = outer["fault_arm"]["label"]
        arm_protocol = result_dir / f"{label}_protocol.json"
        arm_result = result_dir / f"{label}_result.json"
        arm_sources, mismatches = protocol_sources(root, arm_protocol)
        source_paths.extend(arm_sources)
        source_mismatches.extend(mismatches)
        files.extend([
            outer_protocol, outer_result, arm_protocol, arm_result,
            result_dir / f"{label}.log",
        ])
        files.extend((result_dir / "raw").glob(label + "*"))
    if source_mismatches:
        raise ValueError("V16 frozen source mismatch: "
                         + json.dumps(source_mismatches, sort_keys=True))

    validation_path = result_dir / "softwall_envelope_v16_validation_summary.json"
    validation = json.loads(validation_path.read_text())
    if not validation.get("all_pass") or validation.get("status") != "PASS":
        raise ValueError("V16 validation did not pass")
    matrix = validation["physical_fault_matrix"]
    expected = {
        "operation_counts": {
            "prepare": 2, "abort": 2, "commit": 2, "complete": 2,
        },
        "distinct_nodes": ["nid001005", "nid001244", "nid003197", "nid003417"],
        "fault_arms": 8,
        "radio_records": 15360,
        "radio_after_fault": 10125,
        "ai45_exchanges": 12,
        "suppressed_prepares": 666,
        "maximum_unlaunched_tokens": 1,
    }
    for key, expected_value in expected.items():
        if matrix.get(key) != expected_value:
            raise ValueError("authoritative {} mismatch: {} != {}".format(
                key, matrix.get(key), expected_value
            ))

    immutable_inputs = [
        result_dir / "softwall_sharded_home_envelope_grid_v16.json",
        result_dir / "softwall_sharded_home_envelope_prediction_v16.json",
        validation_path,
    ]
    extra_scripts = [
        script_dir / "four_point_crash_global_trace_lease_broker.py",
        script_dir / "test_four_point_crash_broker.py",
        script_dir / "run_v16_abort_fault_arm.sh",
        script_dir / "run_v16_abort_node_requalification.sh",
        script_dir / "analyze_v16_abort_node.py",
        script_dir / "build_softwall_envelope_v16.py",
        script_dir / "test_build_softwall_envelope_v16.py",
        script_dir / "softwall_envelope_checker_v11.py",
        script_dir / "test_softwall_envelope_checker_v11.py",
        script_dir / "analyze_softwall_envelope_v16_validation.py",
        script_dir / "build_softwall_v16_manifest.py",
        script_dir / "test_build_softwall_v16_manifest.py",
    ]
    files.extend(source_paths + immutable_inputs + extra_scripts)
    extension = relative_hashes(root, files)
    transitive_paths = set(base["files"]) | set(extension)
    payload = {
        "schema": "softwall-v16-four-point-manifest-v1",
        "status": "FOUR_POINT_TWO_ARMS_PER_OPERATION_PASS",
        "base_manifest": {
            "path": str(base_path.relative_to(root)),
            "sha256": sha256(base_path),
            "status": base["status"],
            "artifact_count": base["formal_artifact_count"],
            "integrity_mismatches": base_mismatches,
        },
        "campaign_integrity": {
            "source_hash_mismatches": source_mismatches,
            "node_results_pass": all(node["all_pass"] for node in nodes),
            "envelope_validation_pass": True,
            "base_manifest_integrity_pass": not base_mismatches,
        },
        "classification_counts": validation["classification_counts"],
        "physical_scope": dict(
            matrix, declared_safety_or_duplicate_violations=0
        ),
        "claim_boundary": validation["claim_boundary"],
        "supersedes_for_current_claim": (
            "V15 remains the sealed timing and ownership base, but its envelope "
            "row is UQ under the new all-four physical-fault requirement. V16 "
            "adds two independent abort arms and qualifies all four broker "
            "state-changing operations with two arms each."
        ),
        "mutable_documents_excluded": True,
        "files": extension,
        "extension_artifact_count": len(extension),
        "formal_artifact_count": len(transitive_paths),
    }
    return payload


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "results/softwall_multigpu/softwall_v16_four_point_manifest.json"
    output.write_text(
        json.dumps(build(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
