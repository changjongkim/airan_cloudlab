#!/usr/bin/env python3
"""Build the evidence-backed production-readiness exit gate for SoftWall."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTMAC = ROOT / "third_party/aerial-cuda-accelerated-ran/cuPHY-CP/testMAC"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yaml_integer(text: str, key: str) -> int:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*(\d+)\s*$", text)
    if match is None:
        raise RuntimeError(f"missing {key}")
    return int(match.group(1))


def require(text: str, pattern: str, label: str) -> None:
    if re.search(pattern, text, flags=re.MULTILINE) is None:
        raise RuntimeError(f"source contract missing: {label}")


def counts_from_profiles(data: dict) -> list[dict]:
    key = "profiles" if "profiles" in data else "arms"
    return [
        {
            "name": item.get("name"),
            "conventional_correct": item["conventional_correct"],
            "neural_correct": item["neural_correct"],
        }
        for item in data[key]
    ]


def main() -> None:
    readme_path = TESTMAC / "README.md"
    config_path = TESTMAC / "testMAC/test_mac_config.yaml"
    loader_path = TESTMAC / "testMAC/test_mac_configs.cpp"
    handler_path = TESTMAC / "testMAC/scf_fapi_handler.cpp"
    readme = readme_path.read_text(encoding="utf-8")
    config = config_path.read_text(encoding="utf-8")
    loader = loader_path.read_text(encoding="utf-8")
    handler = handler_path.read_text(encoding="utf-8")
    require(readme, r"tool used by developers.*controlled environment", "testMAC role")
    require(loader, r'config_node\["ul_ind_deadline_ns"\]', "deadline loader")
    require(handler, r"void scf_fapi_handler::validate_indication_timing", "validator")
    require(handler, r"handle_start_time-t0_slot > deadline_ns", "late comparison")
    require(
        handler,
        r"configs->ul_ind_deadline_ns.*cell_summary\[cell_id\]\.ul_ind",
        "CRC/UL indication validation call",
    )

    local = load("results/softwall_multigpu/raw/c163_local_4p5ms_job58866163.json")
    parallel = load(
        "results/softwall_multigpu/raw/c163_speculative_parallel_job58866163.json"
    )
    p2p_split = load(
        "results/softwall_multigpu/raw/c163_p2p_speculative_job58866163_controller.json"
    )
    raw_p2p = load(
        "results/softwall_multigpu/raw/c163_raw_p2p_job58866163_controller.json"
    )
    raw_p2p_gc_off = load(
        "results/softwall_multigpu/raw/c163_raw_p2p_v2_gc_off_job58866163_controller.json"
    )
    raw_p2p_single_stream_worker = load(
        "results/softwall_multigpu/raw/c163_raw_p2p_v3_single_stream_job58866163_worker.json"
    )
    raw_p2p_same_stream = load(
        "results/softwall_multigpu/raw/c163_raw_p2p_v4_same_stream_job58868184_controller.json"
    )
    stage_profile = load(
        "results/softwall_multigpu/c163_raw_p2p_v5_stage_profile_analysis_job58868184.json"
    )
    q2 = load("results/softwall_multigpu/confirm159_q2_variable_two_node.json")
    tdl_original = load(
        "results/softwall_same_gpu/raw/aerial_tdl_1x4_compatibility_job58694384.json"
    )
    normalization = load(
        "results/softwall_same_gpu/raw/aerial_tdl_normalization_development_job58865869.json"
    )
    profile = load(
        "results/softwall_same_gpu/raw/aerial_tdl_reference_profile_development_job58865869.json"
    )
    tx_contract = load(
        "results/softwall_same_gpu/raw/aerial_tdl_tx_contract_development_job58865869.json"
    )
    fp32 = load(
        "results/softwall_same_gpu/raw/aerial_tdl_fp32_reference_development_v3_job58866163.json"
    )
    clean_reference = load(
        "results/softwall_same_gpu/raw/nrx_reference_clean_reference_mcs7_stream1_tx1_wrapper_attempt3_job58868184.json"
    )
    sionna_rayleigh = load(
        "results/softwall_same_gpu/raw/sionna_rayleigh_reference_development_job58868184.json"
    )
    sionna_cdl_family = [
        load("results/softwall_same_gpu/raw/sionna_cdl_A_delay1ns_development_job58868184.json"),
        load("results/softwall_same_gpu/raw/sionna_cdl_A_delay10ns_development_job58868184.json"),
        load("results/softwall_same_gpu/raw/sionna_cdl_A_delay30ns_development_job58868184.json"),
        load("results/softwall_same_gpu/raw/sionna_cdl_A_delay100ns_development_job58868184.json"),
        load("results/softwall_same_gpu/raw/sionna_cdl_B_delay100ns_development_job58868184.json"),
        load("results/softwall_same_gpu/raw/sionna_cdl_C_delay100ns_development_job58868184.json"),
        load("results/softwall_same_gpu/raw/sionna_cdl_D_delay100ns_development_job58868184.json"),
        load("results/softwall_same_gpu/raw/sionna_cdl_E_delay100ns_development_job58868184.json"),
    ]
    channel_holdout = load(
        "results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json"
    )
    staged_production_gates = load(
        "results/softwall_multigpu/softwall_production_gates_current_v2.json"
    )

    deadline_ns = yaml_integer(config, "ul_ind_deadline_ns")
    wait_then_recover_diagnostic_pass = (
        local["deadline_counts"]["wait_then_recover_wall_le_deadline"]
        == local["iterations"]
    )
    raw_p2p_diagnostic_pass = (
        raw_p2p_same_stream["deadline_counts"]["parallel_pair_wall_le_deadline"]
        == raw_p2p_same_stream["parallel_pair_wall_ms"]["n"]
    )
    # A testMAC threshold is an integration target, not a production d_MAC.
    # Keep this explicit so a clean microbenchmark can never silently promote
    # itself into a production timing qualification.
    live_du_trace_available = False
    timing_pass = live_du_trace_available and raw_p2p_diagnostic_pass
    channel_arms = [
        {
            "campaign": "prior_corrected_1x4",
            "arms": [{
                "name": "raw",
                "conventional_correct": tdl_original["conventional_correct"],
                "neural_correct": tdl_original["neural_correct"],
            }],
        },
        {
            "campaign": "normalization",
            "arms": [
                {"name": name, **values}
                for name, values in normalization["totals"].items()
            ],
        },
        {"campaign": "radio_profile", "arms": counts_from_profiles(profile)},
        {"campaign": "tx_and_channel_mode", "arms": counts_from_profiles(tx_contract)},
        {"campaign": "official_fp32", "arms": counts_from_profiles(fp32)},
    ]
    channel_diagnostic_has_any_success = any(
        arm["neural_correct"] > 0
        for campaign in channel_arms
        for arm in campaign["arms"]
    )
    # Development arms define the candidate domain; only the separately
    # frozen, disjoint-seed D/E holdout can close P3.
    supported_channel_holdout_available = channel_holdout.get("all_pass") is True
    channel_pass = supported_channel_holdout_available

    all_pass = timing_pass and channel_pass

    result = {
        "schema": "softwall-production-exit-gate-v2",
        "all_pass": all_pass,
        "status": (
            "PASS_PRODUCTION_QUALIFIED"
            if all_pass
            else "FAIL_CURRENT_DESIGN_NOT_PRODUCTION_QUALIFIED"
        ),
        "timing_contract": {
            "status": "VENDOR_TESTMAC_THRESHOLD_IDENTIFIED_NOT_PRODUCTION_D_MAC",
            "source_role": (
                "NVIDIA describes testMAC as a developer tool acting as L2 in a "
                "controlled environment; it is a stronger integration target than "
                "the synthetic P180/D155 contract but not a field DU trace."
            ),
            "early_harq_deadline_ns": yaml_integer(config, "early_harq_deadline_ns"),
            "ul_ind_deadline_ns": deadline_ns,
            "prach_ind_deadline_ns": yaml_integer(config, "prach_ind_deadline_ns"),
            "uci_ind_deadline_ns": yaml_integer(config, "uci_ind_deadline_ns"),
            "validator_semantics": (
                "testMAC reconstructs slot T0 with sfn_to_tai and marks the indication "
                "late when handler-entry time minus T0 exceeds the configured threshold."
            ),
            "source_sha256": {
                str(readme_path.relative_to(ROOT)): digest(readme_path),
                str(config_path.relative_to(ROOT)): digest(config_path),
                str(loader_path.relative_to(ROOT)): digest(loader_path),
                str(handler_path.relative_to(ROOT)): digest(handler_path),
            },
        },
        "current_timing_diagnosis": {
            "gate_pass": timing_pass,
            "live_du_trace_available": live_du_trace_available,
            "wait_then_recover_diagnostic_pass": wait_then_recover_diagnostic_pass,
            "raw_iq_full_remote_diagnostic_pass": raw_p2p_diagnostic_pass,
            "qualification_rule": (
                "Production timing requires both a target-DU trace-derived d_MAC and a "
                "physically requalified implementation under that contract. Passing the "
                "testMAC microbenchmark alone is insufficient."
            ),
            "local_isolated_clean": {
                "neural_wall_ms": local["neural_wall_ms"],
                "conventional_wall_ms": local["conventional_wall_ms"],
                "wait_then_recover_wall_ms": local["wait_then_recover_wall_ms"],
                "deadline_counts": local["deadline_counts"],
            },
            "speculative_same_gpu_parallel": {
                "parallel_pair_wall_ms": parallel["parallel_pair_wall_ms"],
                "deadline_counts": parallel["deadline_counts"],
            },
            "cross_gpu_candidates": {
                "precomputed_ls_split": {
                    "parallel_pair_wall_ms": p2p_split["parallel_pair_wall_ms"],
                    "deadline_counts": p2p_split["deadline_counts"],
                },
                "raw_iq_full_remote": {
                    "parallel_pair_wall_ms": raw_p2p["parallel_pair_wall_ms"],
                    "deadline_counts": raw_p2p["deadline_counts"],
                },
                "raw_iq_full_remote_gc_off": {
                    "parallel_pair_wall_ms": raw_p2p_gc_off["parallel_pair_wall_ms"],
                    "deadline_counts": raw_p2p_gc_off["deadline_counts"],
                },
                "raw_iq_full_remote_same_stream": {
                    "parallel_pair_wall_ms": raw_p2p_same_stream["parallel_pair_wall_ms"],
                    "deadline_counts": raw_p2p_same_stream["deadline_counts"],
                    "correctness": raw_p2p_same_stream["correctness"],
                },
                "raw_iq_single_stream_wrapper": {
                    "controller_outcome": "endpoint timeout on first timed request",
                    "completed_worker_units": raw_p2p_single_stream_worker["completed_units"],
                    "neural_gpu_ms": raw_p2p_single_stream_worker["neural_gpu_ms"],
                },
            },
            "same_stream_stage_diagnosis": {
                "analysis_role": stage_profile["analysis_role"],
                "timely_units": stage_profile["timely_units"],
                "late_units": stage_profile["late_units"],
                "channel_estimation": stage_profile["stages"]["channel_estimation"],
                "tensorrt_graph": stage_profile["stages"]["tensorrt_graph"],
                "forward_copy_gpu_us": stage_profile["copy"]["forward_gpu_us"],
                "interpretation": (
                    "Profiling events alter timing, so this is mechanism evidence only. "
                    "Channel estimation carried the dominant tail and correlated 0.9603 "
                    "with pair wall time; TensorRT and P2P remained comparatively stable."
                ),
            },
            "qualified_q2_context": {
                "nrx_release_to_complete_ms": q2["summary"]["nrx_release_to_complete_ms"],
                "recovery_path_ms": q2["summary"]["recovery_path_ms"],
            },
            "interpretation": (
                "The current wait-then-recover path missed the 4.5 ms diagnostic in "
                "1000/1000 clean isolated trials. Same-GPU speculative dual execution and "
                "the precomputed-LS cross-GPU split also missed 1000/1000. Moving the full "
                "NeuralRx path behind raw-IQ P2P improved completion to 852/1000, while GC "
                "OFF produced 838/1000. Caller-owned same-stream execution improved this to "
                "885/1000 but retained 115 late completions. Stage diagnosis localized the "
                "tail to cuPHY channel estimation rather than TensorRT or P2P. These are "
                "sample diagnostics and do not qualify the path at 4.5 ms."
            ),
        },
        "external_channel_diagnosis": {
            "gate_pass": channel_pass,
            "supported_channel_holdout_available": supported_channel_holdout_available,
            "aerial_tdl_development_has_any_neural_success": channel_diagnostic_has_any_success,
            "qualification_rule": (
                "External-channel support requires a model with an explicit supported-channel "
                "contract and a frozen disjoint-seed holdout; development-screen success alone "
                "would not qualify the mode."
            ),
            "aerial_tdl_a": {
                "channel": "Aerial TDL-A, 30 ns, 10 Hz, 1x4 UL, no AWGN",
                "qualified": False,
                "campaigns": channel_arms,
            },
            "interface_controls": {
                "clean_reference": {
                    "conventional_correct": clean_reference["profiles"][0]["conventional_correct"],
                    "neural_correct": clean_reference["profiles"][0]["neural_correct"],
                },
                "public_default_sionna_rayleigh": {
                    "conventional_correct": sionna_rayleigh["conventional_correct"],
                    "neural_correct": sionna_rayleigh["neural_correct"],
                },
            },
            "sionna_cdl_development_boundary": [
                {
                    "model": item["cdl_model"],
                    "delay_spread_ns": item["delay_spread_ns"],
                    "conventional_correct": item["conventional_correct"],
                    "neural_correct": item["neural_correct"],
                }
                for item in sionna_cdl_family
            ],
            "sionna_cdl_de_holdout": channel_holdout,
            "interpretation": (
                "Aerial TDL-A remains unqualified: conventional succeeded in every reported "
                "arm and NeuralRx in none. The clean MCS7 and public-default Sionna Rayleigh "
                "controls both passed 20/20, localizing the gap to the selective-channel "
                "domain. A fixed development screen found support for CDL-D/E at 100 ns; "
                "the disjoint-seed paired holdout then passed its preregistered P3 checks. "
                "Across low-SNR D/E pairs NeuralRx-only correctness was 31 versus 12 "
                "conventional-only (two-sided exact p=0.00540). This qualifies only the "
                "declared Sionna CDL-D/E finite-sample mode, not TDL-A or field IQ."
            ),
        },
        "claim_decision": {
            "production_harq_guarantee": "REJECT",
            "external_tdl_neuralrx_support": "REJECT",
            "sionna_cdl_de_neuralrx_support": "FINITE_SAMPLE_PASS",
            "synthetic_qualified_substrate": "RETAIN",
            "reason": (
                "P3 is now closed for the narrowly declared Sionna CDL-D/E mode. P1 has no "
                "target-DU clock contract and P2 retains 4.5 ms tail violations, so the "
                "current implementation still cannot be presented as production PUSCH/HARQ capable."
            ),
        },
        "required_exit_gates": [
            {
                "id": "P1_LIVE_DU_CLOCK",
                "requirement": (
                    "Capture IQ-ready, PHY submit, CRC/FAPI publish, and MAC consume/expiry "
                    "on one clock from the target DU; derive d_MAC from that trace."
                ),
            },
            {
                "id": "P2_FAST_PATH",
                "requirement": (
                    "Implement a path whose qualified all-fail completion fits the derived "
                    "deadline. Same-stream raw-IQ full-remote P2P is the best measured "
                    "candidate but still exceeded 4.5 ms in 115/1000 trials; the measured "
                    "tail is dominated by remote cuPHY channel estimation."
                ),
            },
            {
                "id": "P3_CHANNEL_COMPATIBILITY",
                "status": "PASS_FOR_SIONNA_CDL_D_E_ONLY",
                "requirement": (
                    "Completed for the declared Sionna CDL-D/E 100 ns mode. Field-IQ or "
                    "Aerial TDL-A use requires a new model/domain contract and holdout."
                ),
            },
            {
                "id": "P4_INTEGRATED_REQUALIFICATION",
                "requirement": (
                    "Re-run MPS plus bounded Qwen plus correlated all-fail recovery under the "
                    "P1 timing contract and P3 channel workload on independent nodes."
                ),
            },
        ],
        "staged_gate_status": staged_production_gates,
    }
    output = ROOT / "results/softwall_multigpu/softwall_production_exit_gate_v2.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "all_pass": result["all_pass"],
        "status": result["status"],
        "timing_gate_pass": timing_pass,
        "channel_gate_pass": channel_pass,
    }, indent=2))


if __name__ == "__main__":
    main()
