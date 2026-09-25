#!/usr/bin/env python3.11
"""Build the bounded C162 predictive-envelope artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXTRA_SOURCES = (
    "scripts_for_node/softwall_same_gpu/run_c162_feasibility_grid_v1.py",
    "scripts_for_node/softwall_same_gpu/test_c162_feasibility_model_v1.py",
    "scripts_for_node/softwall_same_gpu/c162_validate_q2_predictions.py",
    "scripts_for_node/softwall_same_gpu/analyze_c162_boundary_two_node.py",
    "scripts_for_node/softwall_same_gpu/c162_certified_scheduler_v1.py",
    "scripts_for_node/softwall_same_gpu/test_c162_certified_scheduler_v1.py",
    "scripts_for_node/softwall_same_gpu/run_c162_scheduler_scalability_v1.py",
    "scripts_for_node/softwall_same_gpu/build_c162_paper_figure.py",
    "scripts_for_node/softwall_same_gpu/build_c162_manifest.py",
)
RESULTS = (
    "results/softwall_multigpu/c162_feasibility_grid_v1.json",
    "results/softwall_multigpu/c162_q2_retrospective_validation.json",
    "results/softwall_multigpu/c162a_boundary_dev_j58860486_protocol.json",
    "results/softwall_multigpu/c162a_boundary_dev_j58860486_peer_spec.json",
    "results/softwall_multigpu/c162a_boundary_dev_j58860486_result.json",
    "results/softwall_multigpu/c162b_boundary_holdout_j58860535_protocol.json",
    "results/softwall_multigpu/c162b_boundary_holdout_j58860535_peer_spec.json",
    "results/softwall_multigpu/c162b_boundary_holdout_j58860535_result.json",
    "results/softwall_multigpu/c162_boundary_two_node.json",
    "results/softwall_multigpu/c162_scheduler_scalability_v1.json",
    "results/softwall_multigpu/c162_figure_data/frontier.tsv",
    "results/softwall_multigpu/c162_figure_data/physical_safe.tsv",
    "results/softwall_multigpu/c162_figure_data/physical_reject.tsv",
    "results/softwall_multigpu/c162_figure_data/latency.tsv",
    "results/softwall_multigpu/c162_figure_data/c162_figure.gnuplot",
)
DOCS = (
    "docs/current/SOFTWALL_C162_PREDICTIVE_ENVELOPE_RESULT_KO.md",
    "docs/current/figures/softwall_c162_envelope_scalability.svg",
    "docs/current/figures/softwall_c162_envelope_scalability.pdf",
    "docs/current/figures/softwall_c162_envelope_scalability.png",
)
LABELS = (
    "c162a_boundary_dev_j58860486",
    "c162b_boundary_holdout_j58860535",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    dev_protocol = load(root / RESULTS[2])
    hold_protocol = load(root / RESULTS[5])
    frozen = dev_protocol["source_sha256"]
    files = {root / name for name in EXTRA_SOURCES + RESULTS + DOCS}
    for logical in frozen:
        prefix, relative = logical.split("/", 1)
        base = (root / "scripts_for_node/softwall_same_gpu") if prefix == "scripts" else (root / "scripts_for_node/task1")
        files.add(base / relative)
    raw = root / "results/softwall_multigpu/raw"
    raw_files = []
    for label in LABELS:
        matched = sorted(raw.glob(f"{label}_*"))
        raw_files.extend(matched)
        files.update(matched)
    missing = sorted(str(path) for path in files if not path.is_file())
    if missing:
        raise FileNotFoundError(f"missing manifest files: {missing}")
    source_match = frozen == hold_protocol["source_sha256"]
    for logical, expected in frozen.items():
        prefix, relative = logical.split("/", 1)
        base = (root / "scripts_for_node/softwall_same_gpu") if prefix == "scripts" else (root / "scripts_for_node/task1")
        source_match &= sha256(base / relative) == expected
    documents = {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size, "sha256": sha256(path)
        }
        for path in sorted(files)
    }
    grid, retrospective = load(root / RESULTS[0]), load(root / RESULTS[1])
    dev, holdout, combined = (load(root / RESULTS[index]) for index in (4, 7, 8))
    scalability = load(root / RESULTS[9])
    gates = {
        "model_grid_pass": grid["all_pass"] is True,
        "retrospective_pass": retrospective["all_pass"] is True,
        "development_pass": dev["all_pass"] is True,
        "holdout_pass": holdout["all_pass"] is True,
        "two_node_combined_pass": combined["all_pass"] is True,
        "scheduler_scalability_pass": scalability["all_pass"] is True,
        "frozen_source_equality_and_current_match": source_match,
        "exactly_30_raw_campaign_files": len(raw_files) == 30,
        "all_paths_bounded_below_workspace": all(path.is_relative_to(root) for path in files),
    }
    value = {
        "schema": "softwall-c162-artifact-manifest-v1",
        "status": "C162_MANIFEST_PASS" if all(gates.values()) else "C162_MANIFEST_FAIL",
        "all_pass": all(gates.values()), "gates": gates,
        "file_count": len(documents), "files": documents,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(args.output)
    print(json.dumps({
        "status": value["status"], "file_count": value["file_count"],
        "gates": gates,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
