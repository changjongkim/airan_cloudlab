#!/usr/bin/env python3.11
"""Audit the numerical and scope claims in the SoftWall manuscript draft."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


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


def read_field(data: Any, field: str) -> Any:
    """Resolve a dotted JSON field, including numeric list indices."""
    if field.startswith("sum(") and field.endswith(")") and "[]" in field:
        expression = field[4:-1]
        prefix, suffix = expression.split("[]", 1)
        rows = read_field(data, prefix.rstrip("."))
        suffix = suffix.lstrip(".")
        return sum(read_field(row, suffix) for row in rows)
    value = data
    for part in field.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def audit_quantitative_provenance(manuscript: str) -> tuple[dict, list[dict]]:
    """Bind every empirical result block in the manuscript to JSON fields.

    Model constants, configuration literals, dates, equation examples, section
    numbers, and citation years are not empirical outcomes and are outside this
    check.  Each empirical block has an invisible marker in the Markdown and an
    equivalent marker in the LaTeX submission; the submission audit verifies
    that the same marker set survives typesetting.
    """
    groups = {
        "Q1_MPS_DIAGNOSTICS": [
            ("results/softwall_same_gpu/confirm26_physical_overrun.json", "by_cap.100.releases", 1500),
            ("results/softwall_same_gpu/confirm26_physical_overrun.json", "by_cap.100.misses", 2),
            ("results/softwall_same_gpu/confirm26_physical_overrun.json", "by_cap.20.releases", 1500),
            ("results/softwall_same_gpu/confirm26_physical_overrun.json", "by_cap.20.misses", 1),
            ("results/softwall_same_gpu/confirm46_retirement_cpu_control.json", "mps_only_misses", 12),
            ("results/softwall_same_gpu/confirm46_retirement_cpu_control.json", "cpu_only_misses", 0),
            ("results/softwall_same_gpu/confirm46_retirement_cpu_control_protocol.json", "radio.iterations_per_condition", 1000),
        ],
        "Q2_WARM_PATH": [
            ("results/softwall_multigpu/confirm159_q2_variable_two_node.json", "summary.rounds", 1200),
            ("results/softwall_multigpu/confirm159_q2_variable_two_node.json", "summary.actual_nrx_requests", 4800),
            ("results/softwall_multigpu/confirm159_q2_variable_two_node.json", "summary.actual_nrx_successes", 3488),
            ("results/softwall_multigpu/confirm159_q2_variable_two_node.json", "summary.physical_recoveries", 1312),
            ("results/softwall_multigpu/confirm159_q2_variable_two_node.json", "summary.qwen_units", 1077),
            ("results/softwall_multigpu/confirm159_q2_variable_two_node.json", "summary.radio_commits", 4800),
            ("results/softwall_multigpu/confirm159_q2_variable_two_node.json", "summary.deadline_misses", 0),
            ("results/softwall_multigpu/confirm159_q2_variable_two_node.json", "summary.nrx_release_to_complete_ms.max", 18.123007),
            ("results/softwall_multigpu/confirm159_q2_variable_two_node.json", "summary.recovery_path_ms.max", 13.465105),
            ("results/softwall_multigpu/confirm159_q2_variable_two_node.json", "summary.radio_release_to_commit_ms.max", 118.022282),
            ("results/softwall_multigpu/confirm159_q2_variable_two_node.json", "summary.qwen_certificate_rejected_by_class.256", 55),
            ("results/softwall_multigpu/confirm159_q2_variable_two_node.json", "summary.qwen_certificate_rejected_by_class.512", 68),
        ],
        "C160_FAULT": [
            ("results/softwall_multigpu/c160_fault_state_model_v1.json", "summary.state_transition_cases", 51),
            ("results/softwall_multigpu/c160_fault_state_model_v1.json", "summary.invariant_violations", 0),
            ("results/softwall_multigpu/c161_full_fault_qualification.json", "summary.physical_rounds", 700),
            ("results/softwall_multigpu/c161_full_fault_qualification.json", "summary.actual_nrx_requests", 2800),
            ("results/softwall_multigpu/c161_full_fault_qualification.json", "summary.physical_recoveries", 942),
            ("results/softwall_multigpu/c161_full_fault_qualification.json", "summary.radio_commits", 2800),
            ("results/softwall_multigpu/c161_full_fault_qualification.json", "summary.terminal_channel_faults", 4),
            ("results/softwall_multigpu/c161_full_fault_qualification.json", "summary.post_terminal_radio_rounds", 260),
        ],
        "C162_ENVELOPE": [
            ("results/softwall_multigpu/c162_feasibility_grid_v1.json", "qualified_grid.count", 16023),
            ("results/softwall_multigpu/c162_feasibility_grid_v1.json", "qualified_grid.analytic_exact_mismatch", 0),
            ("results/softwall_multigpu/c162_q2_retrospective_validation.json", "matches", 1200),
            ("results/softwall_multigpu/c162_q2_retrospective_validation.json", "mismatches", 0),
            ("results/softwall_multigpu/c162_boundary_two_node.json", "total_rounds", 180),
            ("results/softwall_multigpu/c162_boundary_two_node.json", "counts.actual_nrx", 720),
            ("results/softwall_multigpu/c162_boundary_two_node.json", "counts.injected_recoveries", 330),
            ("results/softwall_multigpu/c162_boundary_two_node.json", "counts.qwen_units", 90),
            ("results/softwall_multigpu/c162_boundary_two_node.json", "counts.radio_commits", 720),
        ],
        "C162_SCHEDULER": [
            ("results/softwall_multigpu/c162_scheduler_scalability_v1.json", "small_exact.states", 600),
            ("results/softwall_multigpu/c162_scheduler_scalability_v1.json", "small_exact.exact_accept", 504),
            ("results/softwall_multigpu/c162_scheduler_scalability_v1.json", "small_exact.certified_accept", 494),
            ("results/softwall_multigpu/c162_scheduler_scalability_v1.json", "small_exact.false_safe", 0),
            ("results/softwall_multigpu/c162_scheduler_scalability_v1.json", "small_exact.false_conservative", 10),
            ("results/softwall_multigpu/c162_scheduler_scalability_v1.json", "large_grid.states", 2800),
            ("results/softwall_multigpu/c162_scheduler_scalability_v1.json", "large_grid.by_debt.64.decision_us.p99", 1494.332),
            ("results/softwall_multigpu/c162_scheduler_scalability_v1.json", "large_grid.by_debt.64.decision_us.max", 2197.432),
            ("results/softwall_multigpu/c162_scheduler_scalability_v1.json", "large_grid.by_debt.64.verify_us.p99", 130.984),
        ],
        "C162_NECESSITY": [
            ("results/softwall_multigpu/softwall_necessity_witness_v2.json", "witnesses.0.decision_time_ms", 45),
            ("results/softwall_multigpu/softwall_necessity_witness_v2.json", "witnesses.0.bound_respecting_finish_ms", 165),
            ("results/softwall_multigpu/softwall_necessity_witness_v2.json", "witnesses.0.contract_excess_ms", 12),
            ("results/softwall_multigpu/softwall_necessity_witness_v2.json", "witnesses.0.radio_guard_boundary_ms", 153),
            ("results/softwall_multigpu/softwall_necessity_witness_v2.json", "witnesses.1.decision_time_ms", 89),
            ("results/softwall_multigpu/softwall_necessity_witness_v2.json", "witnesses.1.bound_respecting_finish_ms", 154),
            ("results/softwall_multigpu/softwall_necessity_witness_v2.json", "witnesses.1.contract_excess_ms", 1),
            ("results/softwall_multigpu/softwall_necessity_witness_v2.json", "summary.physical_softwall_reject_rounds", 60),
        ],
        "C159_BASELINE": [
            ("results/softwall_multigpu/confirm159_q3_oracle_screen_v1.json", "summary.offered_requests", 4290),
            ("results/softwall_multigpu/confirm159_q3_oracle_screen_v1.json", "summary.offered_value_tokens", 1129504),
            ("results/softwall_multigpu/confirm159_q3_oracle_screen_v1.json", "summary.totals.softwall.timely_requests", 931),
            ("results/softwall_multigpu/confirm159_q3_oracle_screen_v1.json", "summary.totals.softwall.timely_value_tokens", 385262),
            ("results/softwall_multigpu/confirm159_q3_oracle_screen_v1.json", "summary.minimum_effect_pct", 5.0),
        ],
        "C164_LIFECYCLE": [
            ("results/softwall_multigpu/c164_lifecycle_qualification_summary_v1.json", "counts.qualified_or_partial_modes", 5),
            ("results/softwall_multigpu/c164_lifecycle_qualification_summary_v1.json", "counts.unqualified_modes", 5),
            ("results/softwall_multigpu/c164_lifecycle_qualification_summary_v1.json", "counts.total_modes", 10),
        ],
        "P2_FAST_PATH_AND_STAGE": [
            ("results/softwall_multigpu/softwall_production_exit_gate_v2.json", "timing_contract.ul_ind_deadline_ns", 4500000),
            ("results/softwall_multigpu/softwall_production_exit_gate_v2.json", "current_timing_diagnosis.cross_gpu_candidates.raw_iq_full_remote.deadline_counts.parallel_pair_wall_le_deadline", 852),
            ("results/softwall_multigpu/softwall_production_exit_gate_v2.json", "current_timing_diagnosis.cross_gpu_candidates.raw_iq_full_remote_same_stream.deadline_counts.parallel_pair_wall_le_deadline", 885),
            ("results/softwall_multigpu/softwall_production_exit_gate_v2.json", "current_timing_diagnosis.cross_gpu_candidates.raw_iq_full_remote_same_stream.deadline_counts.parallel_pair_wall_gt_deadline", 115),
            ("results/softwall_multigpu/c163_raw_p2p_v5_stage_profile_analysis_job58868184.json", "stages.channel_estimation.gpu_ms.p50", 0.8664640188217163),
            ("results/softwall_multigpu/c163_raw_p2p_v5_stage_profile_analysis_job58868184.json", "stages.channel_estimation.gpu_ms.p99", 4.873409118652329),
            ("results/softwall_multigpu/c163_raw_p2p_v5_stage_profile_analysis_job58868184.json", "stages.channel_estimation.gpu_ms.max", 9.592415809631348),
            ("results/softwall_multigpu/c163_raw_p2p_v5_stage_profile_analysis_job58868184.json", "stages.channel_estimation.pearson_with_pair_wall", 0.9602820841334867),
            ("results/softwall_multigpu/c163_raw_p2p_v5_stage_profile_analysis_job58868184.json", "stages.tensorrt_graph.gpu_ms.p99", 0.8863686168193817),
            ("results/softwall_multigpu/c163_raw_p2p_v5_stage_profile_analysis_job58868184.json", "host_enqueue_us.tensorrt_graph.mean", 7.442636666666664),
            ("results/softwall_multigpu/c163_raw_p2p_v5_stage_profile_analysis_job58868184.json", "copy.forward_gpu_us.mean", 36.26122652242581),
            ("results/softwall_multigpu/c163_raw_p2p_v5_stage_profile_analysis_job58868184.json", "copy.backward_gpu_us.mean", 17.150400035704177),
            ("results/softwall_multigpu/c163_raw_p2p_v5_stage_profile_analysis_job58868184.json", "copy.forward_gpu_us.p99", 69.26303833723067),
            ("results/softwall_multigpu/c163_raw_p2p_v5_stage_profile_analysis_job58868184.json", "host_enqueue_us.derate_match.mean", 1060.1050966666674),
            ("results/softwall_multigpu/c163_raw_p2p_v5_stage_profile_analysis_job58868184.json", "host_enqueue_us.derate_match.p99", 1105.38023),
            ("results/softwall_multigpu/c163_raw_p2p_v5_stage_profile_analysis_job58868184.json", "stages.derate_match.gpu_ms.p99", 0.1675606334209441),
        ],
        "P3_CHANNEL_HOLDOUT": [
            ("results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json", "models.D.strata.4.trials", 50),
            ("results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json", "models.D.strata.4.conventional_correct", 50),
            ("results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json", "models.D.strata.4.neural_correct", 50),
            ("results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json", "sum(models.D.strata[].conventional_correct)", 153),
            ("results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json", "sum(models.D.strata[].neural_correct)", 164),
            ("results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json", "sum(models.E.strata[].conventional_correct)", 155),
            ("results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json", "sum(models.E.strata[].neural_correct)", 163),
            ("results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json", "models.D.low_snr_neural_only", 16),
            ("results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json", "models.D.low_snr_conventional_only", 5),
            ("results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json", "models.E.low_snr_neural_only", 15),
            ("results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json", "models.E.low_snr_conventional_only", 7),
            ("results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json", "aggregate_low_snr_paired.neural_only", 31),
            ("results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json", "aggregate_low_snr_paired.conventional_only", 12),
            ("results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json", "aggregate_low_snr_paired.exact_pvalue", 0.005401572654591291),
        ],
        "C158_LONG_TAIL": [
            ("results/softwall_multigpu/confirm158a_repeated_attempt4_job58855711_result.json", "summary.actual_nrx_requests", 400),
            ("results/softwall_multigpu/confirm158a_repeated_attempt4_job58855711_result.json", "summary.deadline_misses", 4),
            ("results/softwall_multigpu/confirm158a_repeated_attempt4_job58855711_result.json", "summary.nrx_release_to_complete_ms.max", 350.94815),
            ("results/softwall_multigpu/confirm158a_repeated_attempt4_job58855711_result.json", "summary.nrx_release_to_complete_ms.p99", 8.111042),
            ("results/softwall_multigpu/c164_idle30_two_node.json", "maxima_ms.nrx_release_to_complete", 25.07175),
            ("results/softwall_multigpu/c164_mps_restart_two_node.json", "maxima_ms.nrx_release_to_complete", 22.712971),
        ],
    }
    cache: dict[str, dict] = {}
    evidence: list[dict] = []
    checks: dict[str, bool] = {}
    for group, claims in groups.items():
        marker = f"<!-- provenance: {group} -->"
        checks[f"marker_{group.lower()}"] = marker in manuscript
        for artifact, field, expected in claims:
            if artifact not in cache:
                cache[artifact] = json.loads((ROOT / artifact).read_text())
            actual = read_field(cache[artifact], field)
            ok = actual == expected
            checks[f"field_{group.lower()}_{len(evidence):03d}"] = ok
            evidence.append({
                "group": group,
                "artifact": artifact,
                "field": field,
                "expected": expected,
                "actual": actual,
                "match": ok,
            })
    return checks, evidence


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
    provenance_checks, quantitative_provenance = audit_quantitative_provenance(manuscript)

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
    checks["quantitative_provenance_complete"] = all(provenance_checks.values())
    checks.update(provenance_checks)

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
            "quantitative_provenance_scope": (
                "All empirical quantitative result blocks in the Markdown manuscript; "
                "model constants, configuration literals, dates, equation examples, "
                "section numbers, and citation years are excluded."
            ),
            "quantitative_provenance": quantitative_provenance,
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
