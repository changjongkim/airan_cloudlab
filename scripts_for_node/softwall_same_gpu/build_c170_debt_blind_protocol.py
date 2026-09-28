#!/usr/bin/env python3.11
"""Freeze one C170 two-node physical diagnostic protocol before outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

from build_c162_boundary_protocol import source_hashes as c162_source_hashes
from c170_debt_blind_cases import SCENARIOS
from integrated_shared_recovery_plan_v1 import IntegratedScenarioConfig, REQUEST_ORDER


SOURCE_FILES = (
    "actual_nrx_integrated_coordinator.py",
    "actual_nrx_repeated_control.py",
    "analyze_c170_debt_blind.py",
    "bounded_launch_revalidation.py",
    "build_c170_debt_blind_protocol.py",
    "c159_q2_classes.py",
    "c159_q2_qwen_worker.py",
    "c162_boundary_plan.py",
    "c170_debt_blind_cases.py",
    "c170_debt_blind_coordinator.py",
    "c170_gpu_bound_pad.py",
    "c170_qwen_worker.py",
    "dual_receiver_phy.py",
    "integrated_shared_recovery_plan_v1.py",
    "run_c170_debt_blind.sh",
    "shared_conventional_worker.py",
    "shared_recovery_certificate_v1.py",
    "test_c170_debt_blind.py",
)

EXCLUDED_NODES = (
    "nid001025", "nid001044", "nid001064", "nid001069", "nid001085",
    "nid001109", "nid001124", "nid001145", "nid001177", "nid001204",
    "nid001265", "nid001308", "nid001632", "nid001824", "nid002100",
    "nid002288",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root: Path, task1_root: Path) -> dict[str, str]:
    values = c162_source_hashes(root, task1_root)
    values.update({f"c170/{name}": digest(root / name) for name in SOURCE_FILES})
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--capacity-protocol", type=Path, required=True)
    parser.add_argument("--peer-spec", type=Path, required=True)
    parser.add_argument("--peer-tsv", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--campaign", choices=("development", "holdout"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse-scenarios", action="store_true")
    parser.add_argument("--samples-per-scenario", type=int, default=20)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--period-ms", type=float, default=180.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--release-lead-ms", type=float, default=3000.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol {args.output}")
    if args.samples_per_scenario <= 0:
        parser.error("samples must be positive")
    if args.seed_base <= 0:
        parser.error("seed base must be positive")
    config = IntegratedScenarioConfig(period_ms=round(args.period_ms))
    config.validate()
    host = platform.node()
    accepted = REQUEST_ORDER[:4]
    peers = []
    for index, key in enumerate(accepted):
        home_index = int(key[0].removeprefix("home"))
        safe = f"{key[0]}_{key[1]}"
        peers.append({
            "key": list(key),
            "source_device": home_index,
            "receiver_seed": args.seed_base + index * 100_000 + 11,
            "channel_seed_base": args.seed_base + index * 100_000 + 10_001,
            "snr_db": 20.0,
            "nrx_tag": f"{args.label}_{safe}_nrx",
            "recovery_tag": f"{args.label}_{safe}_recovery",
            "ready_file": str(args.state_dir / f"{safe}_ready.json"),
            "decision_control": str(args.state_dir / f"{safe}_decision.ctrl"),
            "owner_output": str(args.raw_dir / f"{args.label}_{safe}_owner.json"),
        })
    args.peer_spec.parent.mkdir(parents=True, exist_ok=True)
    args.peer_spec.write_text(json.dumps({
        "schema": "softwall-actual-nrx-peer-spec-v1",
        "label": args.label,
        "peers": peers,
    }, indent=2) + "\n")
    args.peer_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.peer_tsv.write_text("".join(
        "\t".join(map(str, (
            row["key"][0].removeprefix("home"), row["key"][1],
            row["source_device"], row["receiver_seed"],
            row["channel_seed_base"], row["snr_db"], row["nrx_tag"],
            row["recovery_tag"], row["ready_file"],
            row["decision_control"], row["owner_output"],
        ))) + "\n" for row in peers
    ))
    value = {
        "schema": "softwall-c170-debt-blind-protocol-v1",
        "status": "frozen-before-run",
        "campaign": args.campaign,
        "label": args.label,
        "host": host,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "analysis_role": (
            "Prespecified C168/C169-failure-informed mechanism correction. The "
            "controller pads the complete acceptance-to-observation AI transaction "
            "after the worker response. Diagnostic only; forbidden for "
            "qualification or production-rate claims."
        ),
        "hypotheses": {
            "E4": (
                "With integer decision 45 ms, control 5 ms, context-256 AI "
                "65 ms, and two 25 ms recoveries, debt-blind completion may "
                "reach 165 ms: 12 ms beyond the 153 ms guard and 10 ms beyond D155."
            ),
            "E6": (
                "With one debt and context-64 AI, integer decision 88 ms is "
                "safe at the 153 ms guard while 89 ms reaches 154 ms."
            ),
        },
        "service_vector_ms": {
            "launch_control_upper": 5.0,
            "launch_send_target": 3.80,
            "launch_accept_target": 4.85,
            "qwen_context64_upper": 35.0,
            "qwen_context256_upper": 65.0,
            "conventional_upper": 25.0,
            "ai_controller_padding_margin": 0.50,
            "recovery_padding_margin": 0.50,
            "qwen_lower_accuracy_tolerance": 0.90,
            "recovery_lower_accuracy_tolerance": 0.80,
            "radio_expiry": 155.0,
            "completion_guard": 2.0,
        },
        "mode": {
            "period_ms": args.period_ms,
            "expiry_ms": config.expiry_ms,
            "nrx_bound_ms": config.nrx_bound_ms,
            "conventional_bound_ms": config.conventional_bound_ms,
            "guard_ms": config.guard_ms,
            "warmup": args.warmup,
            "release_lead_ms": args.release_lead_ms,
            "lifecycle": "warm persistent actual NeuralRx/shared cuPHY/padded Qwen diagnostic",
        },
        "placement": {
            "home0": "GPU0", "home1": "GPU1",
            "shared_cuphy_and_qwen": "GPU2 under MPS 80/20",
            "persistent_actual_nrx": "GPU3 under MPS 80",
        },
        "seed_base": args.seed_base,
        "scenarios": [row.__dict__ for row in SCENARIOS],
        "accepted_physical_keys": [list(key) for key in accepted],
        "expected_global_reject": list(REQUEST_ORDER[4]),
        "samples_per_scenario": args.samples_per_scenario,
        "iterations": args.samples_per_scenario * len(SCENARIOS),
        "reverse_scenarios": args.reverse_scenarios,
        "minimum_effect_size": {
            "forced_guard_violation_fraction": 0.90,
            "softwall_guard_violation_fraction": 0.0,
            "padding_lower_accuracy_fraction": 0.95,
            "padding_upper_compliance_fraction": 0.95,
            "jointly_bound_valid_rounds_per_scenario": 18,
            "model_physical_classification_match_fraction": 1.0,
        },
        "stop_rule": (
            "Run exactly the frozen iterations. Preserve all mismatch, bound "
            "overshoot, process failure, and partial artifacts. No sample or "
            "node exclusion after outcome inspection."
        ),
        "mechanism_change_from_c168": (
            "C168 measured an unbudgeted 1.390-1.815 ms response tail. C169's "
            "fixed reserve still had 10/60 AI overshoots and 2/140 recovery "
            "overshoots on the development node. C170 removes worker-side "
            "guessing: after the response, the controller pads the full "
            "acceptance-to-observation transaction with a 0.50 ms margin. It "
            "prespecifies 95% component compliance and 18/20 jointly valid rounds "
            "per scenario; every mismatch remains reported. C168/C169 remain failures."
        ),
        "decision_timing_rule": (
            "E6a/E6b must round to their frozen 88/89 ms cells. E4 wake-up may "
            "fall in [45,50] ms because the model remains QSN throughout that "
            "region; later observation cannot create a false-safe E4 admission."
        ),
        "node_exclusion_rule": {
            "excluded_before_allocation": list(EXCLUDED_NODES),
            "assigned_node_must_not_be_excluded": True,
            "no_post_outcome_exclusion": True,
        },
        "two_node_rule": (
            "Development and holdout must use distinct nonexcluded A100 nodes; "
            "holdout reverses scenario order."
        ),
        "capacity_model": {
            "path": str(args.capacity_protocol.resolve()),
            "sha256": digest(args.capacity_protocol),
        },
        "source_sha256": source_hashes(args.scripts_root, args.task1_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
