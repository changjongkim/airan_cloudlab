#!/usr/bin/env python3
"""Audit SoftWall service bounds without treating correlated TBs as IID.

C135 is retained as the first prospective mechanism realization.  C136 is a
new-node holdout requalification.  The analyzer reports event, arm/process,
and physical-node evidence separately and never labels a sample maximum WCET.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
from pathlib import Path


MODE_FIELDS = (
    "system",
    "cells",
    "period_ms",
    "deadline_ms",
    "nrx_bound_ms",
    "conv_bound_ms",
    "commit_guard_ms",
    "nrx_endpoints",
    "gc_mode",
    "global_broker_rpc_timeout_ms",
    "global_broker_transaction_budget_ms",
    "admission_ai_guard_ms",
    "routing_policy",
    "fault_pattern",
    "early_mandatory",
    "ai_during_nrx",
    "trace_sha256",
)


def zero_failure_upper(n, confidence=0.95):
    if n <= 0:
        return None
    return 1.0 - (1.0 - confidence) ** (1.0 / n)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with path.open("r") as stream:
        return json.load(stream)


def resolve_recorded_path(root, recorded):
    candidate = Path(recorded)
    if candidate.exists():
        resolved = candidate.resolve()
    else:
        marker = "/airan_cloudlab/"
        text = str(recorded)
        if marker not in text:
            raise ValueError("cannot relocate recorded path: %s" % recorded)
        resolved = (root / text.split(marker, 1)[1]).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise ValueError("recorded path is outside workspace: %s" % resolved)
    return resolved


def mode_signature(controller):
    signature = {key: controller.get(key) for key in MODE_FIELDS}
    ai_bounds = controller.get("ai_bound_ms", {})
    signature["ai128_bound_ms"] = ai_bounds.get("128", ai_bounds.get(128))
    return signature


def campaign_arm_results(root, campaign_path):
    campaign = read_json(campaign_path)
    arms = []
    for index, arm in enumerate(campaign.get("arms", [])):
        recorded = arm.get("result")
        if not recorded:
            raise ValueError("campaign arm %d has no result path" % index)
        arms.append(resolve_recorded_path(root, recorded))
    return campaign, arms


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def collect_arm(root, campaign_name, arm_path):
    arm = read_json(arm_path)
    component_values = {"nrx_response_ms": [], "conv_host_path_ms": [], "ai_execution_ms": []}
    signatures = []
    nodes = set()
    controllers = []
    declared_violation_count = 0

    for home in arm.get("homes", []):
        controller_path = resolve_recorded_path(root, home["controller"])
        controller = read_json(controller_path)
        controllers.append(str(controller_path.relative_to(root)))
        nodes.add(controller.get("host"))
        signatures.append(mode_signature(controller))
        declared_violation_count += sum(int(controller.get(key, 0) or 0) for key in (
            "deadline_misses",
            "nrx_bound_violations",
            "conv_bound_violations",
            "conv_path_bound_violations",
            "background_budget_violations",
            "background_horizon_violations",
        ))
        for record in controller.get("records", []):
            value = record.get("nrx_response_ms")
            if numeric(value):
                component_values["nrx_response_ms"].append(float(value))
            value = record.get("conventional_host_path_ms")
            if numeric(value):
                component_values["conv_host_path_ms"].append(float(value))
        for record in controller.get("background_records", []):
            value = record.get("execution_ms")
            if numeric(value):
                component_values["ai_execution_ms"].append(float(value))

    if not signatures:
        raise ValueError("arm has no controller records: %s" % arm_path)
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("home mode mismatch within arm: %s" % arm_path)
    if len(nodes) != 1 or None in nodes:
        raise ValueError("arm does not map to one named physical node: %s" % arm_path)
    maxima = {key: (max(values) if values else None) for key, values in component_values.items()}
    return {
        "campaign": campaign_name,
        "arm_result": str(arm_path.relative_to(root)),
        "arm_result_sha256": sha256(arm_path),
        "node": next(iter(nodes)),
        "controllers": controllers,
        "mode_signature": signatures[0],
        "declared_violation_count": declared_violation_count,
        "values": component_values,
        "maxima_ms": maxima,
        "arm_all_pass": bool(arm.get("all_pass")),
    }


def summarize_component(arms, key, bound):
    values = [value for arm in arms for value in arm["values"][key]]
    arm_maxima = [arm["maxima_ms"][key] for arm in arms if arm["maxima_ms"][key] is not None]
    by_node = {}
    for arm in arms:
        by_node.setdefault(arm["node"], []).extend(arm["values"][key])
    node_maxima = {node: max(samples) for node, samples in by_node.items() if samples}
    exceedances = [value for value in values if value > bound]
    arm_exceedances = [value for value in arm_maxima if value > bound]
    node_exceedances = [value for value in node_maxima.values() if value > bound]
    observed_max = max(values) if values else None
    return {
        "declared_bound_ms": bound,
        "samples": len(values),
        "observed_max_ms": observed_max,
        "observed_headroom_ms": (bound - observed_max) if observed_max is not None else None,
        "sample_exceedances": len(exceedances),
        "arm_process_units": len(arm_maxima),
        "arm_process_exceedances": len(arm_exceedances),
        "physical_node_units": len(node_maxima),
        "physical_node_exceedances": len(node_exceedances),
        "per_arm_max_ms": arm_maxima,
        "per_node_max_ms": node_maxima,
        "zero_failure_95_upper_if_iid": {
            "sample": zero_failure_upper(len(values)) if not exceedances else None,
            "arm_process": zero_failure_upper(len(arm_maxima)) if not arm_exceedances else None,
            "physical_node": zero_failure_upper(len(node_maxima)) if not node_exceedances else None,
        },
    }


def analyze(root, c135_path, c136_path):
    c135, c135_arms = campaign_arm_results(root, c135_path)
    c136, c136_arms = campaign_arm_results(root, c136_path)
    arms = ([collect_arm(root, "C135", path) for path in c135_arms] +
            [collect_arm(root, "C136_HOLDOUT", path) for path in c136_arms])
    signatures = [arm["mode_signature"] for arm in arms]
    mode_consistent = bool(signatures) and all(item == signatures[0] for item in signatures[1:])
    signature = signatures[0] if signatures else {}
    bounds = {
        "nrx_response_ms": signature.get("nrx_bound_ms"),
        "conv_host_path_ms": signature.get("conv_bound_ms"),
        "ai_execution_ms": signature.get("ai128_bound_ms"),
    }
    components = {}
    c135_components = {}
    holdout_components = {}
    c135_only_arms = [arm for arm in arms if arm["campaign"] == "C135"]
    holdout_arms = [arm for arm in arms if arm["campaign"] == "C136_HOLDOUT"]
    if mode_consistent and all(numeric(value) for value in bounds.values()):
        components = {key: summarize_component(arms, key, float(bound))
                      for key, bound in bounds.items()}
        c135_components = {key: summarize_component(c135_only_arms, key, float(bound))
                           for key, bound in bounds.items()}
        holdout_components = {key: summarize_component(holdout_arms, key, float(bound))
                              for key, bound in bounds.items()}
    no_component_exceedance = bool(components) and all(
        item["sample_exceedances"] == 0 for item in components.values())
    every_arm_sampled = bool(components) and all(
        len(arm["values"][key]) > 0 for arm in arms for key in components)
    declared_gates_pass = all(arm["arm_all_pass"] and arm["declared_violation_count"] == 0
                              for arm in arms)
    nodes = sorted(set(arm["node"] for arm in arms))
    holdout_nodes = sorted(set(arm["node"] for arm in holdout_arms))
    gates = {
        "mode_signature_consistent": mode_consistent,
        "all_components_sampled_in_every_arm": every_arm_sampled,
        "no_declared_component_bound_exceedance": no_component_exceedance,
        "all_embedded_safety_gates_pass": declared_gates_pass,
        "c136_is_six_arm_new_node_holdout": (
            len(holdout_arms) == 6 and len(holdout_nodes) == 1 and
            holdout_nodes[0] not in sorted(set(arm["node"] for arm in arms
                                             if arm["campaign"] == "C135"))
        ),
    }
    finite_sample_pass = all(gates.values())
    result = {
        "schema": "softwall-clustered-service-bound-qualification-v1",
        "status": "FINITE_SAMPLE_PASS" if finite_sample_pass else "UNQUALIFIED",
        "inputs": {
            "c135": str(c135_path.relative_to(root)),
            "c135_sha256": sha256(c135_path),
            "c136": str(c136_path.relative_to(root)),
            "c136_sha256": sha256(c136_path),
        },
        "roles": {
            "C135": "first prospective realization of the model-predicted class",
            "C136_HOLDOUT": "frozen-contract new-node holdout requalification",
        },
        "mode_signature": signature if mode_consistent else None,
        "arms": [{key: value for key, value in arm.items() if key != "values"}
                 for arm in arms],
        "physical_nodes": nodes,
        "components": components,
        "c135_first_realization_components": c135_components,
        "c136_new_node_holdout_components": holdout_components,
        "control_budget": {
            "rpc_timeout_each_ms": signature.get("global_broker_rpc_timeout_ms"),
            "rpc_count": 3,
            "transaction_budget_ms": signature.get("global_broker_transaction_budget_ms"),
            "physical_ai_completion_guard_ms": (
                signature.get("admission_ai_guard_ms", 0) -
                signature.get("global_broker_transaction_budget_ms", 0)
                if mode_consistent else None
            ),
            "evidence": "timeout-enforced path with fail-stop injection; per-RPC latency telemetry is absent",
            "status": "ENFORCED_BUDGET_PATH_TESTED_NOT_WCET",
        },
        "gates": gates,
        "finite_sample_pass": finite_sample_pass,
        "hard_real_time_status": "UNPROVEN",
        "reporting_rule": (
            "Report sample, arm/process, and physical-node units separately. "
            "Use the coarsest defensible unit for the target population. "
            "Observed maxima and zero-exceedance confidence sensitivities are not WCET."
        ),
        "claim_boundary": (
            "The declared warm GC-off A100/MPS component bounds survived a frozen six-arm "
            "new-node holdout and the combined eight-arm audit. This is clustered finite-sample "
            "qualification, not a deterministic WCET, production deadline, or cross-family guarantee."
        ),
    }
    return result


def main():
    root_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--c135", type=Path)
    parser.add_argument("--c136", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    c135 = (args.c135 or root / "results/softwall_multigpu/confirm135_v11_ai40_campaign_result.json").resolve()
    c136 = (args.c136 or root / "results/softwall_multigpu/confirm136_v12_ai40_requalification_result.json").resolve()
    output = (args.output or root / "results/softwall_multigpu/softwall_v12_clustered_bound_qualification_v1.json").resolve()
    result = analyze(root, c135, c136)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": result["status"]}, indent=2))
    if not result["finite_sample_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
