#!/usr/bin/env python3
"""Build the contract-level necessity witness from frozen C162 points.

The witness does not claim an observed deadline miss.  It establishes that an
idle-only, debt-blind AI admission cannot guarantee the qualified radio
contract: there is a service-time realization within the declared bounds that
misses the radio guard.  The physical C162 runs show that SoftWall reached the
same prespecified decision states and rejected them before GPU launch.
"""

from __future__ import print_function

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "results" / "softwall_multigpu" / "c162_feasibility_grid_v1.json"
BOUNDARY = ROOT / "results" / "softwall_multigpu" / "c162_boundary_two_node.json"
ORACLE = ROOT / "results" / "softwall_multigpu" / "confirm159_q3_oracle_screen_v1.json"
Q2 = ROOT / "results" / "softwall_multigpu" / "confirm159_q2_variable_two_node.json"
OUT = ROOT / "results" / "softwall_multigpu" / "softwall_necessity_witness_v2.json"


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
                "observed_deadline_miss_claimed": False,
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
        and oracle.get("open_confirmatory_holdout") is False
        and oracle["model"]["event_order"] == "physically drain unresolved recoveries, then admit AI"
        and all(x["mandatory_all_fail_safe"] for x in witnesses)
        and all(x["softwall_state"] == "QSN" for x in witnesses)
        and all(x["contract_excess_ms"] > 0 for x in witnesses)
        and physical_reject_rounds == 60
        and sensitivity_by_case["E4_two_unresolved_context256"]["maximum_safe_recovery_bound_ms"] == 19.0
        and sensitivity_by_case["E6b_first_unsafe_context64"]["maximum_safe_recovery_bound_ms"] == 24.0
        and q2_recovery["max"] == 13.465105
        and boundary_recovery_max == 8.950475
    )

    result = {
        "schema": "softwall-necessity-witness-v2",
        "status": "SOFTWALL_CONTRACT_NECESSITY_WITNESS_PASS" if all_pass else "SOFTWALL_CONTRACT_NECESSITY_WITNESS_FAIL",
        "analysis_role": "Posthoc contract interpretation of prespecified C162 points; no new outcome was opened.",
        "all_pass": all_pass,
        "claim": (
            "Current GPU idleness is insufficient for early external-AI admission when unresolved "
            "same-TB recovery debt exists. For each witness, mandatory-only recovery is feasible, "
            "but a debt-blind AI admission has a duration realization within the qualified bounds "
            "that crosses the radio guard. SoftWall physically reached and rejected these frozen "
            "states before GPU launch."
        ),
        "non_claims": [
            "No observed debt-blind deadline miss is claimed from C162 because the rejected AI kernel was not launched.",
            "The witness does not claim a throughput benefit over certificate-preserving recovery-first.",
            "Failure-correlation probabilities are not estimated; all-fail safety is distribution-free within the fault model.",
        ],
        "baseline_semantics": {
            "c159_recovery_first": "certificate-preserving conservative policy that drains unresolved recovery before AI",
            "softwall": "certificate-preserving policy that may place AI before recovery when a witness schedule remains",
            "debt_blind_diagnostic": "unsafe policy that treats current GPU idleness as sufficient and ignores unresolved recovery",
            "tie_interpretation": "The 385262-token tie rejects material AI-first throughput benefit for the trace; it does not reject the all-fail certificate.",
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
            "recovery_first_timely_tokens": oracle["summary"]["totals"]["event_empirical"]["timely_value_tokens"],
            "softwall_timely_tokens": oracle["summary"]["totals"]["softwall"]["timely_value_tokens"],
        },
        "artifact_sha256": {
            str(GRID.relative_to(ROOT)): sha256(GRID),
            str(BOUNDARY.relative_to(ROOT)): sha256(BOUNDARY),
            str(ORACLE.relative_to(ROOT)): sha256(ORACLE),
            str(Q2.relative_to(ROOT)): sha256(Q2),
        },
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
