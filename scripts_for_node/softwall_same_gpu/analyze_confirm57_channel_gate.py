#!/usr/bin/env python3.11
"""Fresh-seed, prewarmed observable-feature calibration audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from analyze_channel_feature_gate import FEATURE, auc, gate, load, select_threshold


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = root / "results/softwall_same_gpu"
    raw = result / "raw"
    protocol = read(result / "confirm57_disjoint_feature_gate_protocol.json")
    train_seed = protocol["radio"]["train_seed"]
    test_seed = protocol["radio"]["test_seed"]

    def path(seed: int, mode: str) -> Path:
        return raw / f"confirm57_feature_job{args.job}_seed{seed}_{mode}.json"

    documents = {
        (seed, mode): read(path(seed, mode))
        for seed in (train_seed, test_seed) for mode in ("pre_fading", "post_fading")
    }
    train_pre = load(path(train_seed, "pre_fading"), "pre_fading", train_seed)
    train_post = load(path(train_seed, "post_fading"), "post_fading", train_seed)
    test_pre = load(path(test_seed, "pre_fading"), "pre_fading", test_seed)
    test_post = load(path(test_seed, "post_fading"), "post_fading", test_seed)
    train_channels = {row["channel_seed"] for row in train_pre}
    test_channels = {row["channel_seed"] for row in test_pre}
    gates: dict[str, bool] = {}
    gates["source_hashes"] = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in protocol["source_sha256_before_run"].items()
    )
    gates["fresh_disjoint_channel_seeds"] = (
        len(train_channels) == len(test_channels) == 500
        and not train_channels.intersection(test_channels)
        and train_channels.isdisjoint(set(range(20356001, 20356502)))
        and test_channels.isdisjoint(set(range(20356001, 20356502)))
    )
    gates["pre_post_paired"] = all(
        [(a["trial"], a["channel_seed"]) for a in pre]
        == [(b["trial"], b["channel_seed"]) for b in post]
        for pre, post in ((train_pre, train_post), (test_pre, test_post))
    )
    gates["prewarmed_before_timed_trials"] = all(
        doc.get("observed_features_prewarmed") is True for doc in documents.values()
    )
    threshold, train_gate = select_threshold(train_pre)
    test_gate = gate(test_pre, threshold)
    test_neural_only = sum(
        row["neural_correct"] and not row["conventional_correct"]
        for row in test_pre
    )
    feature_times = [
        row["observed_features"]["gpu_ms"]
        for rows in (train_pre, train_post, test_pre, test_post)
        for row in rows
    ]
    frozen = protocol["frozen_gates"]
    gates["train_neural_only_lost_at_most"] = train_gate["neural_only_lost"] <= frozen["train_neural_only_lost_at_most"]
    gates["heldout_skip_at_least"] = test_gate["skipped_nrx"] >= frozen["heldout_skip_at_least"]
    gates["heldout_neural_only_lost_at_most"] = test_gate["neural_only_lost"] <= frozen["heldout_neural_only_lost_at_most"]
    gates["heldout_neural_only_present"] = test_neural_only > 0
    gates["feature_gpu_max_below_ms"] = max(feature_times) < frozen["feature_gpu_max_below_ms"]
    report = {
        "schema": "softwall-confirm57-disjoint-feature-gate-audit-v1",
        "job": args.job,
        "train_seed": train_seed,
        "test_seed": test_seed,
        "train_channel_range": [min(train_channels), max(train_channels)],
        "test_channel_range": [min(test_channels), max(test_channels)],
        "feature": FEATURE,
        "threshold_selected_on_train": threshold,
        "train_auc_conventional_correct": auc(train_pre),
        "test_auc_conventional_correct": auc(test_pre),
        "train_gate": train_gate,
        "test_gate": test_gate,
        "heldout_neural_only_correct": test_neural_only,
        "feature_gpu_ms": {
            "max": max(feature_times),
            "median": sorted(feature_times)[len(feature_times) // 2],
        },
        "gates": gates,
        "all_pass": all(gates.values()),
        "raw_sha256": {
            str(path(seed, mode).relative_to(root)): hashlib.sha256(path(seed, mode).read_bytes()).hexdigest()
            for seed in (train_seed, test_seed) for mode in ("pre_fading", "post_fading")
        },
        "interpretation": "Synthetic fixed pre-fading-noise channel-gate calibration only. No online admission, MPS co-run, production deadline, or joint-policy result.",
    }
    output = result / f"confirm57_disjoint_feature_gate_job{args.job}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_pass": report["all_pass"],
        "failed": [name for name, passed in gates.items() if not passed],
        "train_gate": train_gate,
        "test_gate": test_gate,
        "heldout_neural_only_correct": test_neural_only,
        "feature_gpu_ms": report["feature_gpu_ms"],
    }, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
