#!/usr/bin/env python3
"""Build the contract and physical necessity witness from C162 and C172.

The C162 model establishes the false-safe states.  The prespecified C172
diagnostic physically realizes the declared service vector and launches the
debt-blind/shadow arms on two independent nodes.
"""

from __future__ import print_function

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "results" / "softwall_multigpu" / "c162_feasibility_grid_v1.json"
BOUNDARY = ROOT / "results" / "softwall_multigpu" / "c162_boundary_two_node.json"
ORACLE = ROOT / "results" / "softwall_multigpu" / "c173_deadline_correction_result_v1.json"
C172 = ROOT / "results" / "softwall_multigpu" / "c172_debt_blind_two_node_v1.json"
Q2 = ROOT / "results" / "softwall_multigpu" / "confirm159_q2_variable_two_node.json"
OUT = ROOT / "results" / "softwall_multigpu" / "softwall_necessity_witness_v3.json"


def load(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    grid = load(GRID)
    boundary = load(BOUNDARY)
    oracle = load(ORACLE)
    c172 = load(C172)
    q2 = load(Q2)
    names = ["E4_two_unresolved_context256", "E6b_first_unsafe_context64"]
    witnesses = []
    for name in names:
        row = grid["prespecified_points"][name]
        point = row["point"]
        prediction = row["prediction"]
        radio_guard_boundary = point["expiry_ms"] - point["guard_ms"]
        excess = prediction["analytic_finish_ms"] - radio_guard_boundary
        recovery_waves = int(math.ceil(float(point["unresolved_debts"]) / point["capacity"]))
        non_recovery_charge = (
            prediction["analytic_finish_ms"]
            - recovery_waves * point["recovery_bound_ms"]
        )
        maximum_safe_recovery_bound = (
            float(radio_guard_boundary - non_recovery_charge) / recovery_waves
        )
        samples = boundary["case_counts"][name]
        witnesses.append(
            {
                "case": name,
                "decision_time_ms": point["decision_time_ms"],
                "unresolved_debts": point["unresolved_debts"],
                "context_length": point["context_length"],
                "mandatory_all_fail_safe": prediction["mandatory_all_fail_safe"],
                "softwall_state": prediction["state"],
                "softwall_ai_safe": prediction["ai_safe"],
                "debt_blind_idle_only_decision": "ACCEPT",
                "bound_respecting_finish_ms": prediction["analytic_finish_ms"],
                "radio_guard_boundary_ms": radio_guard_boundary,
                "contract_excess_ms": excess,
                "sensitivity": {
                    "recovery_waves": recovery_waves,
                    "non_recovery_charge_ms": non_recovery_charge,
                    "maximum_safe_recovery_bound_ms": maximum_safe_recovery_bound,
                    "unsafe_iff": "B_conv > %.3f ms" % maximum_safe_recovery_bound,
                },
                "physical_softwall_reject_rounds": samples,
                "c172_physical_diagnostic": (
                    c172["per_scenario"]["E4_debt_blind_launch"]
                    if name == "E4_two_unresolved_context256"
                    else c172["per_scenario"]["E6b_shadow_launch"]
                ),
            }
        )

    physical_reject_rounds = sum(x["physical_softwall_reject_rounds"] for x in witnesses)
    sensitivity_by_case = {x["case"]: x["sensitivity"] for x in witnesses}
    q2_recovery = q2["summary"]["recovery_path_ms"]
    boundary_recovery_max = boundary["maxima_ms"]["recovery_path"]
    qualified_recovery_bound = grid["prespecified_points"][names[0]]["point"]["recovery_bound_ms"]
    all_pass = (
        grid.get("all_pass") is True
        and boundary.get("all_pass") is True
        and q2.get("all_pass") is True
        and oracle.get("all_pass") is True
        and c172.get("all_pass") is True
        and all(x["mandatory_all_fail_safe"] for x in witnesses)
        and all(x["softwall_state"] == "QSN" for x in witnesses)
        and all(x["contract_excess_ms"] > 0 for x in witnesses)
        and physical_reject_rounds == 60
        and sensitivity_by_case["E4_two_unresolved_context256"]["maximum_safe_recovery_bound_ms"] == 19.0
        and sensitivity_by_case["E6b_first_unsafe_context64"]["maximum_safe_recovery_bound_ms"] == 24.0
        and q2_recovery["max"] == 13.465105
        and boundary_recovery_max == 8.950475
        and c172["per_scenario"]["E4_debt_blind_launch"]["bound_valid_deadline_violations"] == 39
        and c172["per_scenario"]["E6b_shadow_launch"]["bound_valid_guard_violations"] == 39
        and c172["safe_policy_summary"]["bound_valid_guard_violations"] == 0
    )

    result = {
        "schema": "softwall-necessity-witness-v3",
        "status": "SOFTWALL_CONTRACT_NECESSITY_WITNESS_PASS" if all_pass else "SOFTWALL_CONTRACT_NECESSITY_WITNESS_FAIL",
        "analysis_role": "Posthoc contract interpretation of prespecified C162 points; no new outcome was opened.",
        "all_pass": all_pass,
        "claim": (
            "Current GPU idleness is insufficient for early external-AI admission when unresolved "
            "same-TB recovery debt exists. For each witness, mandatory-only recovery is feasible, "
            "but a debt-blind AI admission has a duration realization within the qualified bounds "
            "that crosses the radio guard. C172 physically realized the declared service vector: "
            "E4 crossed guard and deadline, while the E6b shadow crossed the guard. SoftWall "
            "rejected the unsafe states."
        ),
        "non_claims": [
            "C172 is a bound-padded diagnostic, not production-rate, WCET, or production-HARQ qualification.",
            "The witness does not claim a throughput benefit over certificate-preserving recovery-first.",
            "Failure-correlation probabilities are not estimated; all-fail safety is distribution-free within the fault model.",
        ],
        "baseline_semantics": {
            "c159_recovery_first": "certificate-preserving conservative policy that drains unresolved recovery before AI",
            "softwall": "certificate-preserving policy that may place AI before recovery when a witness schedule remains",
            "debt_blind_diagnostic": "unsafe policy that treats current GPU idleness as sufficient and ignores unresolved recovery",
            "tie_interpretation": "The corrected 349387-token tie rejects material AI-first throughput benefit for the trace; it does not reject the all-fail certificate.",
        },
        "contract_sensitivity": {
            "qualified_recovery_bound_ms": qualified_recovery_bound,
            "premise": (
                "The guarantee is relative to the declared qualified service bound. A lower bound "
                "defines a different, unqualified mode until its complete calendar, workload, "
                "placement, and lifecycle are requalified."
            ),
            "thresholds": {
                "B_conv_gt_24_ms": "E4 and E6b are both false-safe for debt-blind admission",
                "19_ms_lt_B_conv_le_24_ms": "E6b disappears but E4 remains false-safe",
                "B_conv_le_19_ms": "both selected witnesses disappear",
            },
            "observed_sample_maxima_are_not_contract_bounds": {
                "c159_q2_authoritative_recovery_path_max_ms": q2_recovery["max"],
                "c159_q2_recovery_path_median_ms": q2_recovery["p50"],
                "c162_boundary_recovery_path_max_ms": boundary_recovery_max,
                "qualified_to_q2_sample_max_ratio": qualified_recovery_bound / q2_recovery["max"],
                "qualified_to_c162_sample_max_ratio": qualified_recovery_bound / boundary_recovery_max,
            },
            "interpretation": (
                "E6b is a one-millisecond boundary witness and vanishes at B_conv <= 24 ms. "
                "E4 remains a witness until B_conv <= 19 ms. The strongest current Q2 sample "
                "maximum is 13.465105 ms rather than the earlier approximately 3.1 ms value, "
                "so hypothetically promoting it to a bound would remove both selected witnesses. "
                "A sample maximum cannot replace the qualified bound without requalification. "
                "For every positive recovery charge, Proposition 2 still yields a later "
                "decision-time false-safe interval; tightening the bound moves the boundary rather "
                "than making unresolved debt irrelevant."
            ),
        },
        "witnesses": witnesses,
        "summary": {
            "prespecified_false_safe_cases": len(witnesses),
            "physical_softwall_reject_rounds": physical_reject_rounds,
            "minimum_contract_excess_ms": min(x["contract_excess_ms"] for x in witnesses),
            "maximum_contract_excess_ms": max(x["contract_excess_ms"] for x in witnesses),
            "recovery_first_timely_tokens": oracle["summary"]["totals"]["recovery_first_empirical"]["timely_value_tokens"],
            "softwall_timely_tokens": oracle["summary"]["totals"]["softwall"]["timely_value_tokens"],
            "physical_debt_blind_E4_selected_deadline_violations": c172["per_scenario"]["E4_debt_blind_launch"]["deadline_violations"],
            "physical_E6b_shadow_selected_guard_violations": c172["per_scenario"]["E6b_shadow_launch"]["guard_violations"],
            "physical_softwall_selected_guard_violations": c172["safe_policy_summary"]["guard_violations"],
            "physical_softwall_selected_rounds": c172["safe_policy_summary"]["selected_rounds"],
            "physical_softwall_zero_95pct_upper": c172["safe_policy_summary"]["zero_violation_95pct_rule_of_three_upper"],
        },
        "artifact_sha256": {
            str(GRID.relative_to(ROOT)): sha256(GRID),
            str(BOUNDARY.relative_to(ROOT)): sha256(BOUNDARY),
            str(ORACLE.relative_to(ROOT)): sha256(ORACLE),
            str(C172.relative_to(ROOT)): sha256(C172),
            str(Q2.relative_to(ROOT)): sha256(Q2),
        },
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
