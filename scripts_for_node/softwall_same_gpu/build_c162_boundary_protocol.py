#!/usr/bin/env python3.11
"""Freeze one C162 physical boundary campaign before execution."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

from c162_boundary_cases import CASES
from c162_feasibility_model_v1 import EnvelopePoint, predict
from integrated_shared_recovery_plan_v1 import REQUEST_ORDER, IntegratedScenarioConfig


SOURCES = (
    "c159_q2_classes.py",
    "c162_boundary_cases.py",
    "c162_boundary_plan.py",
    "c162_feasibility_model_v1.py",
    "c162_boundary_coordinator.py",
    "build_c162_boundary_protocol.py",
    "analyze_c162_boundary.py",
    "run_c162_boundary.sh",
    "test_c162_boundary_plan.py",
    "bounded_launch_revalidation.py",
    "actual_nrx_batch_recovery_plan_v1.py",
    "c159_prestaged_owner.py",
    "c159_persistent_nrx_worker.py",
    "c159_q2_qwen_worker.py",
    "actual_nrx_repeated_control.py",
    "actual_nrx_integrated_coordinator.py",
    "integrated_shared_recovery_launch_plan_v1.py",
    "integrated_shared_recovery_plan_v1.py",
    "shared_recovery_certificate_v1.py",
    "integrated_shared_recovery_worker.py",
    "shared_conventional_worker.py",
    "multigpu_p2p_nrx_worker.py",
    "multigpu_p2p_ipc_gate.py",
    "dual_receiver_phy.py",
)
TASK1_SOURCES = ("isca_v2/cuda_ipc_channel.py",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(scripts_root: Path, task1_root: Path) -> dict[str, str]:
    paths = {f"scripts/{name}": scripts_root / name for name in SOURCES}
    paths.update({f"task1/{name}": task1_root / name for name in TASK1_SOURCES})
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing C162 source: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peer-spec", type=Path, required=True)
    parser.add_argument("--peer-tsv", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--campaign", choices=("development", "holdout"), required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--excluded-nodes", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--period-ms", type=float, default=180.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--release-lead-ms", type=float, default=3000.0)
    parser.add_argument("--reverse-cases", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0 or args.iterations % len(CASES) or args.seed_base <= 0:
        parser.error("iterations must be a positive multiple of the six cases")
    grid = json.loads(args.grid.read_text())
    if grid.get("status") != "C162_MODEL_GRID_PASS":
        parser.error("C162 grid has not passed")
    config = IntegratedScenarioConfig(period_ms=round(args.period_ms))
    config.validate()
    accepted = REQUEST_ORDER[:4]
    peers = []
    for index, key in enumerate(accepted):
        home_index = int(key[0].removeprefix("home"))
        safe = f"{key[0]}_{key[1]}"
        peers.append({
            "key": list(key), "source_device": home_index,
            "receiver_seed": args.seed_base + index * 100_000 + 11,
            "channel_seed_base": args.seed_base + index * 100_000 + 10_001,
            "snr_db": 20.0,
            "nrx_tag": f"{args.label}_{safe}_nrx",
            "recovery_tag": f"{args.label}_{safe}_recovery",
            "ready_file": str(args.state_dir / f"{safe}_ready.json"),
            "decision_control": str(args.state_dir / f"{safe}_decision.ctrl"),
            "owner_output": str(args.raw_dir / f"{args.label}_{safe}_owner.json"),
        })
    atomic_json(args.peer_spec, {
        "schema": "softwall-actual-nrx-peer-spec-v1",
        "label": args.label, "peers": peers,
    })
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
    case_rows = []
    for case in CASES:
        decision = 88 if case.case_id.startswith("E6a") else (
            89 if case.case_id.startswith("E6b") else case.target_decision_ms
        )
        prediction = predict(EnvelopePoint(
            4, 4 - case.success_count, decision, case.context_length
        ))
        case_rows.append({
            **dataclasses.asdict(case),
            "conservative_prediction_time_ms": decision,
            "model_state": prediction.state,
            "model_ai_safe": prediction.ai_safe,
        })
    protocol = {
        "schema": "softwall-c162-boundary-protocol-v1",
        "status": "frozen-before-run",
        "label": args.label, "campaign": args.campaign,
        "iterations": args.iterations,
        "samples_per_case": args.iterations // len(CASES),
        "reverse_cases": args.reverse_cases,
        "seed_base": args.seed_base,
        "excluded_nodes": sorted({v for v in args.excluded_nodes.split(",") if v}),
        "mode": {
            "period_ms": args.period_ms, "expiry_ms": config.expiry_ms,
            "nrx_bound_ms": config.nrx_bound_ms,
            "conventional_bound_ms": config.conventional_bound_ms,
            "guard_ms": config.guard_ms, "launch_control_bound_ms": 5,
            "warmup": args.warmup, "release_lead_ms": args.release_lead_ms,
            "lifecycle": "warm persistent actual NeuralRx/shared cuPHY/Qwen",
        },
        "placement": {
            "home0": "GPU0", "home1": "GPU1",
            "shared_cuphy_and_qwen": "GPU2 under MPS 80/20",
            "persistent_actual_nrx": "GPU3 under MPS 80",
        },
        "accepted_physical_keys": [list(key) for key in accepted],
        "expected_global_reject": list(REQUEST_ORDER[4]),
        "physical_input": "All four accepted NeuralRx paths use +20 dB stable inputs and must physically succeed by NRx45.",
        "fault_injection": "After physical NRx completion, each frozen case converts a suffix of successful requests to failed outcomes before one atomic transition; every injected failure executes physical shared-cuPHY recovery.",
        "cases": case_rows,
        "gates": {
            "prediction": "observed conservative decision time maps to the frozen state and every lease decision matches it",
            "boundary": "E6a is observed no later than integer-ms 88 and admitted; E6b is observed at integer-ms 89 or later and rejected",
            "physical": "all actual NRx, injected recoveries, admitted Qwen units, commits, whole-path bounds and lifecycle gates pass",
            "admission": "the fifth all-fail debt is rejected without state mutation in every round",
            "sample": "every case completes the frozen samples_per_case",
        },
        "grid": {"path": str(args.grid), "sha256": sha256(args.grid)},
        "source_sha256": source_hashes(args.scripts_root, args.task1_root),
    }
    atomic_json(args.output, protocol)


if __name__ == "__main__":
    main()
