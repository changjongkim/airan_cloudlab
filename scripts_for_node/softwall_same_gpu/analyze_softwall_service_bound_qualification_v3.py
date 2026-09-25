#!/usr/bin/env python3
"""Audit the timeout/wall-bound correction and the V13 service class."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def qualification(root, v2_path, c138_path, c139_path, c140_path, v13_path):
    v2 = load(v2_path)
    c138 = load(c138_path)
    c139 = load(c139_path)
    c140 = load(c140_path)
    v13 = load(v13_path)

    old_bound = 5.0
    new_bound = float(c139["control_evidence"]["declared_per_rpc_admission_bound_ms"])
    socket_timeout = float(c139["control_evidence"]["socket_timeout_ms"])
    fault_max = float(c138["control_evidence"]["max_elapsed_ms"])
    corrected_max = float(c139["control_evidence"]["maximum_elapsed_ms"])
    cf = c140["model_counterfactual"]

    gates = {
        "historical_v2_was_finite_sample_pass": v2.get("all_pass") is True,
        "c138_falsifies_timeout_as_wall_bound": (
            c138.get("all_pass") is False
            and c138["gates"].get("system_safety") is True
            and c138["gates"].get("rpc_wall_bound") is False
            and fault_max > old_bound
        ),
        "socket_timeout_separated_from_admission_bound": (
            socket_timeout == old_bound and new_bound > socket_timeout
        ),
        "c139_corrected_fault_matrix_passes": (
            c139.get("all_pass") is True
            and c139["control_evidence"].get("admission_bound_exceeded") == 0
            and corrected_max <= new_bound
        ),
        "full_control_charge_is_21_ms": new_bound * 3 == 21.0,
        "c140_corrected_conditional_class_passes": (
            c140.get("all_pass") is True
            and cf.get("effective_transaction_bound_ms") == 58.0
            and cf.get("static_margin_ms") == -5.0
            and cf.get("conditional_margin_ms") == 0.0
            and c140["totals"].get("ai35_candidate_exchanges") == 8
        ),
        "v13_validation_chain_passes": v13.get("all_pass") is True,
    }
    all_pass = all(gates.values())

    inputs = {}
    for key, path in (
        ("historical_v2", v2_path),
        ("c138_falsification", c138_path),
        ("c139_corrected_fault_matrix", c139_path),
        ("c140_corrected_conditional_class", c140_path),
        ("v13_validation", v13_path),
    ):
        inputs[key] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256(path),
        }

    return {
        "schema": "softwall-service-bound-qualification-v3",
        "status": "FINITE_SAMPLE_CORRECTED_CONTROL_PASS" if all_pass else "UNQUALIFIED",
        "supersession": {
            "superseded_contract": "V12_AI40_CONTROL15_GUARD2",
            "reason": (
                "C138 observed controller-return wall time above the 5 ms socket timeout; "
                "the timeout therefore cannot be reused as the admission bound."
            ),
            "current_contract": "V13_AI35_CONTROL21_GUARD2",
        },
        "control_contract": {
            "socket_timeout_ms": socket_timeout,
            "per_rpc_admission_bound_ms": new_bound,
            "synchronous_rpc_count": 3,
            "control_transaction_bound_ms": 3 * new_bound,
            "ai_completion_guard_ms": 2.0,
            "raw_ai_bound_ms": 35.0,
            "effective_transaction_bound_ms": 58.0,
            "static_all_fail_slack_ms": 53.0,
            "conditional_decision_window_ms": 58.0,
        },
        "falsification": {
            "campaign": "C138",
            "node": "nid001144",
            "attempted_rpc_records": c138["control_evidence"]["records"],
            "faulting_rpc_max_ms": fault_max,
            "five_ms_exceedances": c138["control_evidence"]["wall_timeout_exceeded"],
            "system_safety_preserved": c138["gates"]["system_safety"],
        },
        "corrected_fault_qualification": {
            "campaign": "C139",
            "nodes": c139["nodes"],
            "arms": c139["totals"]["arms"],
            "radio_records": c139["totals"]["radio_records"],
            "radio_records_after_fault_detection": c139["totals"]["radio_records_after_fault_detection"],
            "attempted_rpc_records": c139["control_evidence"]["records"],
            "maximum_rpc_wall_ms": corrected_max,
            "admission_bound_exceedances": c139["control_evidence"]["admission_bound_exceeded"],
        },
        "corrected_conditional_class": {
            "campaign": "C140",
            "nodes": c140["nodes"],
            "arms": len(c140["arms"]),
            "radio_records": c140["totals"]["radio_records"],
            "candidate_branches": c140["totals"]["candidate_branches"],
            "ai35_candidate_exchanges": c140["totals"]["ai35_candidate_exchanges"],
            "minimum_physical_guarded_horizon_margin_ms": c140["totals"]["minimum_physical_guarded_horizon_margin_ms"],
            "static_margin_ms": cf["static_margin_ms"],
            "conditional_margin_ms": cf["conditional_margin_ms"],
        },
        "inputs": inputs,
        "gates": gates,
        "all_pass": all_pass,
        "hard_real_time_status": "UNPROVEN",
        "claim_boundary": (
            "C138--C140 establish a finite-sample correction chain on A100: a socket "
            "timeout is not a controller-return bound, a distinct 7 ms per-RPC admission "
            "charge passed six fault arms, and the resulting AI35+control21+guard2 class "
            "executed eight conditional exchanges. This is not WCET, production d_MAC, "
            "independent-node C140, cross-family, or throughput-superiority evidence."
        ),
    }


def main():
    root_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--v2", type=Path)
    parser.add_argument("--c138", type=Path)
    parser.add_argument("--c139", type=Path)
    parser.add_argument("--c140", type=Path)
    parser.add_argument("--v13", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result_dir = root / "results/softwall_multigpu"
    paths = {
        "v2_path": (args.v2 or result_dir / "softwall_service_bound_qualification_v2.json").resolve(),
        "c138_path": (args.c138 or result_dir / "confirm138_prepare1_job58831304_result.json").resolve(),
        "c139_path": (args.c139 or result_dir / "confirm139_qualified_control_bound_result.json").resolve(),
        "c140_path": (args.c140 or result_dir / "confirm140_v13_ai35_result.json").resolve(),
        "v13_path": (args.v13 or result_dir / "softwall_envelope_v13_validation_summary.json").resolve(),
    }
    output = (args.output or result_dir / "softwall_service_bound_qualification_v3.json").resolve()
    result = qualification(root, **paths)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": result["status"]}, indent=2))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
