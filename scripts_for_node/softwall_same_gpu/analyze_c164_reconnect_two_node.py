#!/usr/bin/env python3.11
"""Combine frozen development and independent-node reconnect holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COUNT_KEYS = (
    "tokens", "prepare_loss", "post_fence_loss", "physical_qwen_launches",
    "terminal_fences", "mandatory_releases", "cell_decodes",
    "overlap_releases", "deadline_misses", "component_bound_violations",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def combine(dev: dict, holdout: dict, dev_protocol: dict,
            holdout_protocol: dict) -> dict:
    checks = {
        "both_node_results_pass": dev.get("all_pass") is True
            and holdout.get("all_pass") is True,
        "campaign_roles": dev.get("campaign") == "development"
            and holdout.get("campaign") == "holdout",
        "independent_jobs_and_nodes": dev.get("host") != holdout.get("host")
            and dev.get("slurm_job_id") != holdout.get("slurm_job_id"),
        "holdout_excludes_development_node": dev.get("host")
            in holdout_protocol.get("excluded_nodes", []),
        "frozen_source_equality": dev_protocol.get("source_sha256")
            == holdout_protocol.get("source_sha256"),
        "same_mode_and_sample_size": dev_protocol.get("mode")
            == holdout_protocol.get("mode")
            and dev_protocol.get("tokens") == holdout_protocol.get("tokens") == 60,
        "independent_seed_epoch": dev_protocol.get("seed")
            != holdout_protocol.get("seed"),
        "combined_safety_zero": all(
            dev.get("counts", {}).get(key) == 0
            and holdout.get("counts", {}).get(key) == 0
            for key in ("deadline_misses", "component_bound_violations")
        ),
    }
    counts = {
        key: dev.get("counts", {}).get(key, 0)
            + holdout.get("counts", {}).get(key, 0)
        for key in COUNT_KEYS
    }
    return {
        "schema": "softwall-c164-reconnect-two-node-v1",
        "status": "C164_RECONNECT_TWO_NODE_PASS" if all(checks.values()) else "C164_RECONNECT_TWO_NODE_FAIL",
        "all_pass": all(checks.values()), "checks": checks,
        "nodes": [dev.get("host"), holdout.get("host")],
        "jobs": [dev.get("slurm_job_id"), holdout.get("slurm_job_id")],
        "counts": counts,
        "maxima_ms": {
            "qwen_gpu": max(dev["qwen_gpu_ms"]["max"], holdout["qwen_gpu_ms"]["max"]),
            "mandatory_response": max(dev["mandatory_response_ms"]["max"], holdout["mandatory_response_ms"]["max"]),
            "mandatory_component_gpu": max(dev["mandatory_component_gpu_ms"]["max"], holdout["mandatory_component_gpu_ms"]["max"]),
        },
        "claim_boundary": (
            "Two independent A100 nodes qualify the same-process, same-worker-epoch "
            "channel reconnect subset. Worker-process replacement, arbitrary crash "
            "windows, direct kernel-overlap tracing, production d_MAC and WCET remain UQ."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--dev-protocol", type=Path, required=True)
    parser.add_argument("--holdout-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = combine(load(args.dev), load(args.holdout),
                    load(args.dev_protocol), load(args.holdout_protocol))
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
