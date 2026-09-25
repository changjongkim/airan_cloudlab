#!/usr/bin/env python3.11
"""Audit the numerical and scope claims in the SoftWall manuscript draft."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/softwall_multigpu"
MANUSCRIPT = ROOT / "docs/current/SOFTWALL_MANUSCRIPT_DRAFT_EN.md"
NOVELTY = ROOT / "docs/current/SOFTWALL_NOVELTY_DEFENSE_MATRIX_KO.md"
CONTRACT_FIGURE = ROOT / "docs/current/figures/softwall_system_contract.svg"
ENVELOPE_FIGURE = ROOT / "docs/current/figures/softwall_c162_envelope_scalability.svg"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    q2 = load("confirm159_q2_variable_two_node.json")
    fault = load("c161_full_fault_qualification.json")
    grid = load("c162_feasibility_grid_v1.json")
    retro = load("c162_q2_retrospective_validation.json")
    boundary = load("c162_boundary_two_node.json")
    scale = load("c162_scheduler_scalability_v1.json")
    oracle = load("confirm159_q3_oracle_screen_v1.json")
    lifecycle = load("c164_lifecycle_qualification_summary_v1.json")
    process = load("c164i3_processreplace_dev_j58862843_result.json")
    necessity = load("softwall_necessity_witness_v2.json")
    production = load("softwall_production_exit_gate_v2.json")
    manuscript = MANUSCRIPT.read_text()
    normalized_manuscript = " ".join(manuscript.split())
    novelty = NOVELTY.read_text()

    q2s = q2["summary"]
    faults = fault["summary"]
    debt64 = scale["large_grid"]["by_debt"]["64"]
    oracles = oracle["summary"]

    checks = {
        "q2_authoritative_counts": (
            q2.get("all_pass") is True
            and q2s["rounds"] == 1200
            and q2s["actual_nrx_requests"] == 4800
            and q2s["physical_recoveries"] == 1312
            and q2s["qwen_units"] == 1077
            and q2s["radio_commits"] == 4800
            and q2s["deadline_misses"] == 0
            and q2s["recovery_contract_violations"] == 0
        ),
        "fault_authoritative_counts": (
            fault.get("all_pass") is True
            and faults["actual_nrx_requests"] == 2800
            and faults["physical_recoveries"] == 942
            and faults["radio_commits"] == 2800
            and faults["deadline_misses"] == 0
            and faults["post_terminal_radio_rounds"] == 260
        ),
        "model_exact_and_retrospective": (
            grid.get("all_pass") is True
            and grid["qualified_grid"]["count"] == 16023
            and grid["qualified_grid"]["analytic_exact_mismatch"] == 0
            and retro.get("all_pass") is True
            and retro["matches"] == 1200
            and retro["mismatches"] == 0
        ),
        "physical_boundary": (
            boundary.get("all_pass") is True
            and boundary["total_rounds"] == 180
            and boundary["counts"] == {
                "actual_nrx": 720,
                "deadline_misses": 0,
                "injected_recoveries": 330,
                "qwen_units": 90,
                "radio_commits": 720,
            }
            and set(boundary["case_counts"].values()) == {30}
        ),
        "certified_scheduler": (
            scale.get("all_pass") is True
            and scale["small_exact"]["states"] == 600
            and scale["small_exact"]["exact_accept"] == 504
            and scale["small_exact"]["certified_accept"] == 494
            and scale["small_exact"]["false_safe"] == 0
            and scale["small_exact"]["false_conservative"] == 10
            and scale["current_qualified_mode"]["acceptance_mismatch"] == 0
            and debt64["decision_us"]["p99"] == 1494.332
        ),
        "negative_performance_result": (
            oracle["status"] == "C159_Q3_ORACLE_SCREEN_STOP_PERFORMANCE"
            and oracle["open_confirmatory_holdout"] is False
            and oracles["minimum_effect_pct"] == 5.0
            and oracles["totals"]["event_empirical"]["timely_value_tokens"] == 385262
            and oracles["totals"]["softwall"]["timely_value_tokens"] == 385262
            and oracles["softwall_gain_over_event_empirical_pct"] == 0.0
        ),
        "claim_scoped_lifecycle": (
            lifecycle.get("claim_scope_complete") is True
            and lifecycle["paper_gate_status"] == "C164_CLAIM_SCOPED_LIFECYCLE_COMPLETE"
            and lifecycle["counts"] == {
                "qualified_or_partial_modes": 5,
                "total_modes": 10,
                "unqualified_modes": 5,
            }
            and lifecycle["modes"]["worker_process_replacement"]["status"]
                == "UQ_SINGLE_NODE_DEVELOPMENT_ONLY"
            and process.get("all_pass") is True
            and process.get("campaign") == "development"
        ),
        "manuscript_numbers_present": all(token in normalized_manuscript for token in (
            "4,800 actual TensorRT neural-receiver requests",
            "1,312 shared-cuPHY recoveries",
            "1,077 Qwen units",
            "16,023 qualified states",
            "1.494 ms p99 decision",
            "385,262 tokens",
            "no material throughput advantage",
        )),
        "necessity_witness": (
            necessity.get("all_pass") is True
            and necessity.get("status") == "SOFTWALL_CONTRACT_NECESSITY_WITNESS_PASS"
            and necessity["summary"]["prespecified_false_safe_cases"] == 2
            and necessity["summary"]["physical_softwall_reject_rounds"] == 60
            and necessity["summary"]["minimum_contract_excess_ms"] == 1
            and necessity["summary"]["maximum_contract_excess_ms"] == 12
            and necessity["contract_sensitivity"]["thresholds"]["19_ms_lt_B_conv_le_24_ms"]
                == "E6b disappears but E4 remains false-safe"
            and necessity["contract_sensitivity"]["observed_sample_maxima_are_not_contract_bounds"]
                ["c159_q2_authoritative_recovery_path_max_ms"] == 13.465105
            and "certificate-preserving recovery-first" in manuscript
            and "contract-level counterexamples" in manuscript
            and "not observed debt-blind misses" in manuscript
            and "E6b disappears at or below 24 ms" in manuscript
            and "E4 remains until the bound reaches 19 ms" in manuscript
        ),
        "manuscript_scope_present": all(token in normalized_manuscript for token in (
            "no WCET or production-HARQ claim",
            "we make no optimizer claim",
            "finite-sample qualification",
            "production `d_MAC`",
            "Process/model cold, 5 min and 30 min idle, GC-on, and worker process replacement remain UQ",
        )),
        "production_exit_scope_preserved": (
            production.get("all_pass") is False
            and production.get("status") == "FAIL_CURRENT_DESIGN_NOT_PRODUCTION_QUALIFIED"
            and production["current_timing_diagnosis"]["gate_pass"] is False
            and production["current_timing_diagnosis"]["cross_gpu_candidates"]
                ["raw_iq_full_remote_same_stream"]["deadline_counts"]
                ["parallel_pair_wall_le_deadline"] == 885
            and production["external_channel_diagnosis"]["gate_pass"] is True
            and production["external_channel_diagnosis"]["aerial_tdl_a"]["qualified"] is False
            and production["claim_decision"]["sionna_cdl_de_neuralrx_support"]
                == "FINITE_SAMPLE_PASS"
            and "885/1,000" in manuscript
            and "external aerial tdl-a remains unqualified"
                in normalized_manuscript.lower()
            and "31 neuralrx-only versus 12 conventional-only"
                in normalized_manuscript.lower()
            and "make no production harq guarantee"
                in normalized_manuscript.lower()
        ),
        "closest_work_covered": all(token in novelty for token in (
            "YinYangRAN",
            "CloudRIC",
            "Concordia",
            "DARIS",
            "ARCHES",
            "OCUDU",
            "Interplay/CAORA",
            "HAF",
            "SMEC",
        )),
        "paper_figures_present": (
            CONTRACT_FIGURE.stat().st_size > 1000
            and ENVELOPE_FIGURE.stat().st_size > 1000
            and "figures/softwall_system_contract.svg" in manuscript
            and "figures/softwall_c162_envelope_scalability.svg" in manuscript
        ),
        "forbidden_overclaim_absent": all(token not in manuscript.lower() for token in (
            "first ai-ran system",
            "mps guarantees isolation",
            "we prove wcet",
            "softwall guarantees zero deadline misses",
            "novel optimizer",
        )),
    }

    inputs = [
        "confirm159_q2_variable_two_node.json",
        "c161_full_fault_qualification.json",
        "c162_feasibility_grid_v1.json",
        "c162_q2_retrospective_validation.json",
        "c162_boundary_two_node.json",
        "c162_scheduler_scalability_v1.json",
        "confirm159_q3_oracle_screen_v1.json",
        "c164_lifecycle_qualification_summary_v1.json",
        "c164i3_processreplace_dev_j58862843_result.json",
        "softwall_necessity_witness_v2.json",
        "softwall_production_exit_gate_v2.json",
    ]
    output = {
        "schema": "softwall-manuscript-claim-audit-v1",
        "status": "SOFTWALL_MANUSCRIPT_CLAIMS_AUDITED_PASS" if all(checks.values())
                  else "SOFTWALL_MANUSCRIPT_CLAIMS_AUDITED_FAIL",
        "all_pass": all(checks.values()),
        "checks": checks,
        "evidence": {
            "q2": {key: q2s[key] for key in (
                "actual_nrx_requests", "physical_recoveries", "qwen_units",
                "radio_commits", "deadline_misses")},
            "fault": {key: faults[key] for key in (
                "actual_nrx_requests", "physical_recoveries", "radio_commits",
                "deadline_misses", "post_terminal_radio_rounds")},
            "model_states": grid["qualified_grid"]["count"],
            "boundary_rounds": boundary["total_rounds"],
            "debt64_decision_p99_ms": debt64["decision_us"]["p99"] / 1000.0,
            "oracle_timely_tokens": oracles["totals"]["softwall"]["timely_value_tokens"],
            "lifecycle_counts": lifecycle["counts"],
            "necessity": necessity["summary"],
            "production_exit": {
                "status": production["status"],
                "timing_gate_pass": production["current_timing_diagnosis"]["gate_pass"],
                "channel_gate_pass": production["external_channel_diagnosis"]["gate_pass"],
                "raw_iq_timely": production["current_timing_diagnosis"]
                    ["cross_gpu_candidates"]["raw_iq_full_remote_same_stream"]
                    ["deadline_counts"]["parallel_pair_wall_le_deadline"],
            },
        },
        "artifact_sha256": {
            **{name: sha256(RESULTS / name) for name in inputs},
            "docs/current/SOFTWALL_MANUSCRIPT_DRAFT_EN.md": sha256(MANUSCRIPT),
            "docs/current/SOFTWALL_NOVELTY_DEFENSE_MATRIX_KO.md": sha256(NOVELTY),
            "docs/current/figures/softwall_system_contract.svg": sha256(CONTRACT_FIGURE),
            "docs/current/figures/softwall_c162_envelope_scalability.svg": sha256(ENVELOPE_FIGURE),
            "scripts_for_node/softwall_same_gpu/audit_softwall_manuscript_claims.py": sha256(Path(__file__)),
        },
    }
    target = RESULTS / "softwall_manuscript_claim_audit_v1.json"
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    print(json.dumps(output, indent=2, sort_keys=True))
    raise SystemExit(0 if output["all_pass"] else 1)


if __name__ == "__main__":
    main()
