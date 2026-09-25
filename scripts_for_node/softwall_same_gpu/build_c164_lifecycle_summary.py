#!/usr/bin/env python3.11
"""Build one claim-scoped lifecycle qualification matrix for C164."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def build(warm: dict, model: dict, idle: dict, restart: dict,
          reload: dict, reconnect_model: dict, reconnect_physical: dict,
          process_replacement: dict, gc: dict) -> dict:
    modes = {
        "warm_persistent": {
            "status": "QUALIFIED_BOUNDARY_SUBSET",
            "mandatory_continuity": True,
            "optional_boundary": True,
            "availability_during_transition": None,
            "evidence": warm.get("status"),
        },
        "idle_30s_first": {
            "status": "QUALIFIED_BOUNDARY_SUBSET",
            "mandatory_continuity": True,
            "optional_boundary": True,
            "availability_during_transition": True,
            "evidence": idle.get("status"),
        },
        "mps_restart_first": {
            "status": "QUALIFIED_AFTER_REQUALIFICATION_SUBSET",
            "mandatory_continuity": None,
            "optional_boundary": True,
            "availability_during_transition": False,
            "evidence": restart.get("status"),
        },
        "qwen_reload_first": {
            "status": "QUALIFIED_MANDATORY_ONLY_SUBSET",
            "mandatory_continuity": True,
            "optional_boundary": False,
            "availability_during_transition": False,
            "evidence": reload.get("status"),
        },
        "cold_first": {
            "status": "UQ_NO_WHOLE_MODE_EVIDENCE",
            "mandatory_continuity": None,
            "optional_boundary": False,
            "availability_during_transition": False,
            "evidence": None,
        },
        "worker_channel_reconnect_same_epoch": {
            "status": "QUALIFIED_RECONCILIATION_SUBSET",
            "mandatory_continuity": True,
            "optional_boundary": False,
            "availability_during_transition": False,
            "evidence": reconnect_physical.get("status"),
        },
        "worker_process_replacement": {
            "status": "UQ_SINGLE_NODE_DEVELOPMENT_ONLY",
            "mandatory_continuity": None,
            "optional_boundary": False,
            "availability_during_transition": False,
            "evidence": process_replacement.get("status"),
        },
        "idle_5m_first": {
            "status": "UQ_NO_PHYSICAL_SAMPLE",
            "mandatory_continuity": None,
            "optional_boundary": False,
            "availability_during_transition": None,
            "evidence": None,
        },
        "idle_30m_first": {
            "status": "UQ_NO_PHYSICAL_SAMPLE",
            "mandatory_continuity": None,
            "optional_boundary": False,
            "availability_during_transition": None,
            "evidence": None,
        },
        "gc_on": {
            "status": "UQ_OBSERVED_BOUND_FAILURE",
            "mandatory_continuity": None,
            "optional_boundary": False,
            "availability_during_transition": None,
            "evidence": "confirm89_a_on sample_safety=false",
        },
    }
    qualified = {name for name, row in modes.items()
                 if row["status"].startswith("QUALIFIED")}
    uq = set(modes) - qualified
    checks = {
        "warm_input_pass": warm.get("all_pass") is True,
        "state_model_pass": model.get("all_pass") is True,
        "idle_input_pass": idle.get("all_pass") is True,
        "restart_input_pass": restart.get("all_pass") is True,
        "reload_input_pass": reload.get("all_pass") is True,
        "reconnect_model_and_same_worker_physical_pass": (
            reconnect_model.get("all_pass") is True
            and reconnect_physical.get("all_pass") is True
            and modes["worker_channel_reconnect_same_epoch"]["status"].startswith("QUALIFIED")
            and modes["worker_process_replacement"]["status"].startswith("UQ_")
        ),
        "process_replacement_canary_not_promoted": (
            process_replacement.get("all_pass") is True
            and process_replacement.get("campaign") == "development"
            and modes["worker_process_replacement"]["status"]
                == "UQ_SINGLE_NODE_DEVELOPMENT_ONLY"
            and modes["worker_process_replacement"]["optional_boundary"] is False
        ),
        "gc_failure_preserved": (
            gc.get("arm_gates", {}).get("a_on", {}).get("sample_safety") is False
            and gc.get("all_contract_gates_pass") is False
        ),
        "qwen_reload_not_promoted_to_optional_availability": (
            modes["qwen_reload_first"]["optional_boundary"] is False
            and modes["qwen_reload_first"]["availability_during_transition"] is False
        ),
        "restart_downtime_not_promoted_to_availability": (
            modes["mps_restart_first"]["availability_during_transition"] is False
        ),
        "unmeasured_modes_remain_uq": uq == {
            "cold_first", "worker_process_replacement", "idle_5m_first",
            "idle_30m_first", "gc_on",
        },
        "matrix_intentionally_partial": len(qualified) == 5 and len(uq) == 5,
    }
    return {
        "schema": "softwall-c164-lifecycle-qualification-summary-v1",
        "status": "C164_LIFECYCLE_MATRIX_PARTIAL" if all(checks.values()) else "C164_LIFECYCLE_MATRIX_INCONSISTENT",
        "all_inputs_consistent": all(checks.values()),
        "complete": False,
        "claim_scope_complete": all(checks.values()),
        "paper_gate_status": (
            "C164_CLAIM_SCOPED_LIFECYCLE_COMPLETE" if all(checks.values())
            else "C164_CLAIM_SCOPED_LIFECYCLE_INCONSISTENT"
        ),
        "checks": checks,
        "modes": modes,
        "counts": {"qualified_or_partial_modes": len(qualified),
                   "unqualified_modes": len(uq), "total_modes": len(modes)},
        "next_internal_gate": (
            "convert the audited manuscript draft into the venue submission by "
            "consolidating claims, figures, bibliography, and reviewer-objection "
            "responses; run no new experiment unless this review identifies a "
            "claim-critical evidence gap"
        ),
        "claim_boundary": (
            "C164 intentionally stops at five claim-scoped qualified subsets; the "
            "other five modes remain explicitly unqualified rather than forming a "
            "completion checklist. This is not a universal lifecycle guarantee. "
            "Production d_MAC, WCET and cross-family qualification are outside this "
            "matrix."
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = {
        "warm": root / "results/softwall_multigpu/c162_boundary_two_node.json",
        "model": root / "results/softwall_multigpu/c164_lifecycle_model_v1.json",
        "idle": root / "results/softwall_multigpu/c164_idle30_two_node.json",
        "restart": root / "results/softwall_multigpu/c164_mps_restart_two_node.json",
        "reload": root / "results/softwall_multigpu/c164_qwen_reload_two_node.json",
        "reconnect_model": root / "results/softwall_multigpu/c164_reconnect_model_v1.json",
        "reconnect_physical": root / "results/softwall_multigpu/c164_reconnect_two_node.json",
        "process_replacement": root / "results/softwall_multigpu/c164i3_processreplace_dev_j58862843_result.json",
        "gc": root / "results/softwall_same_gpu/confirm89_gc_causal_replication_job58743005.json",
    }
    value = build(**{name: load(path) for name, path in paths.items()})
    value["artifact_sha256"] = {name: sha256(path) for name, path in paths.items()}
    source = Path(__file__)
    value["source_sha256"] = sha256(source)
    output = root / "results/softwall_multigpu/c164_lifecycle_qualification_summary_v1.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["all_inputs_consistent"] else 1)


if __name__ == "__main__":
    main()
