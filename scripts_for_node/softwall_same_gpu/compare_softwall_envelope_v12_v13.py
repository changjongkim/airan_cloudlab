#!/usr/bin/env python3
"""Audit the V12-to-V13 control-bound and conditional-class correction."""

import argparse
import json
from pathlib import Path

from build_softwall_envelope_v13 import NEW_ID, OLD_ID


def by_id(value):
    return {row["id"]: row for row in value["results"]}


def ai_class(row, bound):
    matches = [item for item in row["ai_classes"]
               if float(item["bound_ms"]) == float(bound)]
    if len(matches) != 1:
        raise ValueError("expected exactly one AI class")
    return matches[0]


def compare(old, new):
    old_rows = by_id(old)
    new_rows = by_id(new)
    former = old_rows[OLD_ID]
    invalidated = new_rows[OLD_ID]
    corrected = new_rows[NEW_ID]
    ai35 = ai_class(corrected, 35)
    ai40 = ai_class(corrected, 40)
    gates = {
        "v12_former_mode_was_qsu": former["status"] == "QSU",
        "v13_invalidates_former_5ms_mode": invalidated["status"] == "UQ",
        "v13_corrected_mode_is_qsu": corrected["status"] == "QSU",
        "corrected_control_budget_is_21ms": (
            corrected["control_rpc_bound_ms"] == 7
            and corrected["required_control_transaction_budget_ms"] == 21
            and corrected["declared_control_transaction_budget_ms"] == 21
            and corrected["control_budget_qualified"]
        ),
        "ai35_is_exchange_only_at_58ms": (
            ai35["effective_transaction_bound_ms"] == 58
            and not ai35["static_admissible"]
            and ai35["exchange_only_candidate"]
            and ai35["exchange_only_candidate_homes"] == [0, 1]
        ),
        "ai40_no_longer_fits_decision_window": (
            ai40["effective_transaction_bound_ms"] == 63
            and not ai40["static_admissible"]
            and not ai40["exchange_only_candidate"]
            and ai40["too_large_even_after_exchange"]
        ),
    }
    return {
        "schema": "softwall-envelope-v12-v13-control-bound-regression-v1",
        "former_mode": OLD_ID,
        "corrected_mode": NEW_ID,
        "v12_status": former["status"],
        "v13_former_status": invalidated["status"],
        "v13_corrected_status": corrected["status"],
        "v13_corrected_ai35": ai35,
        "v13_corrected_ai40": ai40,
        "gates": gates,
        "all_pass": all(gates.values()),
        "interpretation": (
            "C138 invalidates the former 5 ms wall-bound assumption. V13 charges "
            "three 7 ms RPC bounds, removes AI40 from the conditional decision "
            "window, and retains an exchange-only AI35 class at the exact 58 ms "
            "boundary validated physically by C140."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(json.loads(Path(args.old).read_text()),
                     json.loads(Path(args.new).read_text()))
    output = Path(args.output)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    if not result["all_pass"]:
        raise SystemExit("V12-to-V13 regression failed")


if __name__ == "__main__":
    main()
