#!/usr/bin/env python3
"""Audit the frozen channel-feature run without reinterpreting failed gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, data["results"][0]["records"]


def summary(rows: list[dict]) -> dict:
    feature_ms = sorted(row["observed_features"]["gpu_ms"] for row in rows)
    return {
        "conventional_correct": sum(bool(row["conventional_correct"]) for row in rows),
        "neural_correct": sum(bool(row["neural_correct"]) for row in rows),
        "union_correct": sum(
            bool(row["conventional_correct"] or row["neural_correct"]) for row in rows
        ),
        "neural_only_correct": sum(
            bool(row["neural_correct"] and not row["conventional_correct"])
            for row in rows
        ),
        "feature_gpu_ms_median": statistics.median(feature_ms),
        "feature_gpu_ms_p99_observed": feature_ms[494],
        "feature_gpu_ms_max": feature_ms[-1],
        "first_feature_gpu_ms": rows[0]["observed_features"]["gpu_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_root = args.root / "results/softwall_same_gpu"
    protocol = json.loads(
        (result_root / "channel_feature_gate_protocol.json").read_text(encoding="utf-8")
    )
    arms = {}
    raw_hashes = {}
    source_hashes_match_snapshot = {}
    for name, expected in protocol["source_sha256_before_run"].items():
        snapshot = result_root / "frozen_source/channel_feature_gate" / Path(name).name
        source_hashes_match_snapshot[name] = digest(snapshot) == expected
    for seed in (20356001, 20356002):
        for mode in ("pre_fading", "post_fading"):
            key = f"{seed}_{mode}"
            path = result_root / "raw" / f"channel_feature_job{args.job}_seed{seed}_{mode}.json"
            data, rows = load(path)
            if (
                data["seed"] != seed or data["noise_reference"] != mode
                or data["results"][0]["snr_db"] != -8.5 or len(rows) != 500
            ):
                raise ValueError(f"frozen input mismatch: {path}")
            arms[key] = {"records": rows, "summary": summary(rows)}
            raw_hashes[path.name] = digest(path)
    train_seeds = {r["channel_seed"] for r in arms["20356001_pre_fading"]["records"]}
    test_seeds = {r["channel_seed"] for r in arms["20356002_pre_fading"]["records"]}
    overlap = train_seeds & test_seeds
    paired = {}
    for seed in (20356001, 20356002):
        pre = arms[f"{seed}_pre_fading"]["records"]
        post = arms[f"{seed}_post_fading"]["records"]
        if any(a["channel_seed"] != b["channel_seed"] for a, b in zip(pre, post)):
            raise ValueError("within-seed channel pairing broke")
        paired[str(seed)] = {
            "neural_post_only_correct": sum(
                not a["neural_correct"] and b["neural_correct"]
                for a, b in zip(pre, post)
            ),
            "neural_pre_only_correct": sum(
                a["neural_correct"] and not b["neural_correct"]
                for a, b in zip(pre, post)
            ),
            "conventional_post_only_correct": sum(
                not a["conventional_correct"] and b["conventional_correct"]
                for a, b in zip(pre, post)
            ),
            "conventional_pre_only_correct": sum(
                a["conventional_correct"] and not b["conventional_correct"]
                for a, b in zip(pre, post)
            ),
        }
    audit = {
        "schema": "softwall-channel-feature-integrity-audit-v1",
        "job": args.job,
        "status": "INVALID_HELDOUT; within-seed noise-mode comparisons descriptive only",
        "train_test_channel_seed_overlap": len(overlap),
        "train_test_total_each": 500,
        "train_channel_seed_range": [min(train_seeds), max(train_seeds)],
        "test_channel_seed_range": [min(test_seeds), max(test_seeds)],
        "source_snapshot_hashes_match_frozen_protocol": source_hashes_match_snapshot,
        "raw_sha256": raw_hashes,
        "arm_summaries": {key: value["summary"] for key, value in arms.items()},
        "within_seed_paired_differences": paired,
        "interpretation": "The threshold test is not independent. A pre/post fading noise-reference change materially alters this synthetic workload; neither mode alone proves field-channel behavior. The observed first feature call includes cold compilation cost and must not be excluded from the frozen gate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
