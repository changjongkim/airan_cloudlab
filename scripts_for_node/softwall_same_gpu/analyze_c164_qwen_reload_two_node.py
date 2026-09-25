#!/usr/bin/env python3.11
"""Combine two independent C164 Qwen-reload continuity campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def max_metric(development: dict, holdout: dict,
               group: str, subset: str, metric: str = "max"):
    return max(development[group][subset][metric],
               holdout[group][subset][metric])


def combine(development: dict, holdout: dict,
            development_protocol: dict, holdout_protocol: dict) -> dict:
    count_keys = (
        "reload_episodes", "mandatory_releases", "overlap_releases",
        "quiet_releases", "cell_decodes", "deadline_misses",
        "component_bound_violations", "optional_inference_units",
    )
    counts = {key: development["counts"][key] + holdout["counts"][key]
              for key in count_keys}
    gates = {
        "both_campaigns_pass": development.get("all_pass") is True and holdout.get("all_pass") is True,
        "development_holdout_roles": (
            development.get("campaign") == "development"
            and holdout.get("campaign") == "holdout"
        ),
        "independent_job_and_node": (
            development.get("host") != holdout.get("host")
            and development.get("slurm_job_id") != holdout.get("slurm_job_id")
        ),
        "frozen_source_equality": (
            bool(development_protocol.get("source_sha256"))
            and development_protocol.get("source_sha256")
                == holdout_protocol.get("source_sha256")
        ),
        "independent_seed": development_protocol.get("seed") != holdout_protocol.get("seed"),
        "development_node_excluded_from_holdout": (
            development.get("host") in holdout_protocol.get("excluded_nodes", [])
        ),
        "same_reload_contract": (
            development_protocol.get("mode") == holdout_protocol.get("mode")
            and development_protocol.get("contract") == holdout_protocol.get("contract")
            and development_protocol.get("episodes")
                == holdout_protocol.get("episodes") == 30
        ),
        "all_sixty_reload_episodes": counts["reload_episodes"] == 60,
        "combined_mandatory_safety": (
            counts["deadline_misses"] == 0
            and counts["component_bound_violations"] == 0
            and counts["optional_inference_units"] == 0
            and counts["overlap_releases"] > 0
            and all(development.get("gates", {}).values())
            and all(holdout.get("gates", {}).values())
        ),
    }
    return {
        "schema": "softwall-c164-qwen-reload-two-node-result-v1",
        "status": "C164_QWEN_RELOAD_TWO_NODE_PASS" if all(gates.values()) else "C164_QWEN_RELOAD_TWO_NODE_FAIL",
        "all_pass": all(gates.values()), "gates": gates,
        "nodes": [development["host"], holdout["host"]],
        "jobs": [development["slurm_job_id"], holdout["slurm_job_id"]],
        "counts": counts,
        "maxima_ms": {
            "reload_load_and_warmup": max(
                development["reload_ms"]["max"], holdout["reload_ms"]["max"]
            ),
            "mandatory_response_all": max_metric(
                development, holdout, "response_ms", "all"
            ),
            "mandatory_response_reload_overlap": max_metric(
                development, holdout, "response_ms", "reload_overlap"
            ),
            "mandatory_response_quiet": max_metric(
                development, holdout, "response_ms", "quiet"
            ),
            "cell_gpu_all": max_metric(
                development, holdout, "component_gpu_ms", "all"
            ),
            "cell_gpu_reload_overlap": max_metric(
                development, holdout, "component_gpu_ms", "reload_overlap"
            ),
        },
        "node_summaries": {
            development["host"]: {
                "counts": development["counts"],
                "reload_ms": development["reload_ms"],
                "response_ms": development["response_ms"],
            },
            holdout["host"]: {
                "counts": holdout["counts"],
                "reload_ms": holdout["reload_ms"],
                "response_ms": holdout["response_ms"],
            },
        },
        "claim_boundary": (
            "Two-node finite-sample mandatory four-cell continuity during 60 "
            "fresh Qwen load/warmup episodes on the same MPS GPU. Optional "
            "inference was closed. Optional-work availability, a reload WCET, "
            "production d_MAC, and cross-family generalization remain UQ."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--development-protocol", type=Path, required=True)
    parser.add_argument("--holdout-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = combine(load(args.development), load(args.holdout),
                    load(args.development_protocol), load(args.holdout_protocol))
    value["artifact_sha256"] = {
        "development": sha256(args.development), "holdout": sha256(args.holdout),
        "development_protocol": sha256(args.development_protocol),
        "holdout_protocol": sha256(args.holdout_protocol),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
