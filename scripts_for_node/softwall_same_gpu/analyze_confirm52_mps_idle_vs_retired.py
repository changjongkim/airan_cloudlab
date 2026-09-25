#!/usr/bin/env python3
"""Frozen three-condition causal control for MPS client lifecycle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def exact_one_sided_sign_p(positive: int, negative: int) -> float:
    discordant = positive + negative
    if not discordant:
        return 1.0
    return sum(math.comb(discordant, k) for k in range(positive, discordant + 1)) / (2 ** discordant)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-rounds", type=int)
    args = parser.parse_args()
    protocol = read(args.protocol)
    planned_rounds = protocol["execution"]["rounds"]
    completed_rounds = args.completed_rounds or planned_rounds
    if not 1 <= completed_rounds <= planned_rounds:
        parser.error("completed rounds must be within the frozen campaign")
    errors = []
    rows = []
    triplets = []
    hosts = set()
    conditions = ("mps_idle", "mps_ack_only", "mps_retired")
    all_miss_indices = {condition: [] for condition in conditions}
    radio = protocol["radio"]
    for round_no in range(1, completed_rounds + 1):
        seed = 20322000 + round_no
        run = {}
        for condition in conditions:
            prefix = f"confirm52_r{round_no}_{condition}_job{args.job}"
            ran = read(args.raw / f"{prefix}_ran.json")
            worker = read(args.raw / f"{prefix}_worker.json")
            hosts.add(ran["host"])
            hosts.add(worker["host"])
            if ran["host"] != worker["host"]:
                errors.append(f"r{round_no} {condition}: worker/RAN host mismatch")
            for data, name in ((ran, "RAN"), (worker, "worker")):
                if data.get("slurm_job_id") != args.job:
                    errors.append(f"r{round_no} {condition}: {name} job mismatch")
            if (
                ran["seed"] != seed
                or ran["period_ms"] != radio["period_ms"]
                or ran["deadline_ms"] != radio["deadline_ms"]
                or ran["iterations"] != radio["iterations_per_condition"]
                or ran["fault_every"] != radio["fault_every"]
                or len(ran["records"]) != radio["iterations_per_condition"]
            ):
                errors.append(f"r{round_no} {condition}: frozen input mismatch")
            if ran["deadline_misses"] != sum(item["deadline_miss"] for item in ran["records"]):
                errors.append(f"r{round_no} {condition}: miss summary mismatch")
            if ran["correct_releases"] != sum(item["correct"] for item in ran["records"]):
                errors.append(f"r{round_no} {condition}: correct summary mismatch")
            if ran["correct_releases"] != radio["iterations_per_condition"]:
                errors.append(f"r{round_no} {condition}: RAN correctness failed")
            if (ran["injected_overruns"] or ran["worker_active_releases"] or ran["outstanding_at_end"]
                or any(item["injected_overrun"] or item["worker_active_during_conventional"] for item in ran["records"])):
                errors.append(f"r{round_no} {condition}: timed optional work present")
            if (
                worker["completed_units"]
                or worker["kind"] != "nrx"
                or worker["mode"] != "rpc"
                or worker["pre_ready_gpu_warmup_units"] != protocol["worker"]["pre_ready_gpu_warmup_units"]
            ):
                errors.append(f"r{round_no} {condition}: worker workload mismatch")
            if (
                worker["mps_active_thread_percentage"] != str(protocol["worker"]["mps_cap"])
                or worker["mps_client_priority"] != str(protocol["worker"]["mps_priority"])
                or worker["repeats"] != 40
            ):
                errors.append(f"r{round_no} {condition}: MPS worker configuration mismatch")
            if not ran["stop_response"] or not ran["stop_response"].get("stopped"):
                errors.append(f"r{round_no} {condition}: stop acknowledgement missing")
            handshake = ran.get("lifecycle_handshake")
            first_release_ns = ran["first_release_ns"]
            if ran["records"][0]["release_ns"] != first_release_ns:
                errors.append(f"r{round_no} {condition}: first release timestamp mismatch")
            if condition == "mps_idle":
                if not (
                    ran["quiesce_before_first_release"]
                    and ran["quiesce_response"]
                    and ran["quiesce_response"].get("quiesced")
                    and ran["gpu_client_live_during_epoch"]
                    and not ran["endpoint_retired"]
                    and ran["endpoint_retired_at_release"] == radio["iterations_per_condition"]
                    and handshake and handshake.get("worker_state") == "alive"
                    and 0 < handshake.get("confirmed_ns", 0) < first_release_ns
                    and ran.get("pre_epoch_stop_ack_ns") is None
                ):
                    errors.append(f"r{round_no} idle: quiescent-client lifecycle gate failed")
            elif condition == "mps_retired":
                if not (
                    ran["retire_before_first_release"]
                    and not ran["gpu_client_live_during_epoch"]
                    and ran["endpoint_retired"]
                    and ran["endpoint_retired_at_release"] == -1
                    and handshake and handshake.get("worker_state") == "exited"
                    and 0 < ran.get("pre_epoch_stop_ack_ns", 0) <= handshake.get("confirmed_ns", 0) < first_release_ns
                ):
                    errors.append(f"r{round_no} retired: pre-epoch retirement gate failed")
            else:
                launcher = read(args.raw / f"{prefix}_launcher.json")
                if not (
                    ran["retire_before_first_release"]
                    and not ran["gpu_client_live_during_epoch"]
                    and ran["endpoint_retired"]
                    and ran["endpoint_retired_at_release"] == -1
                    and handshake is None
                    and 0 < ran.get("pre_epoch_stop_ack_ns", 0) < first_release_ns
                    and launcher.get("worker_state") == "exited"
                    and launcher.get("confirmed_ns", 0) >= ran["pre_epoch_stop_ack_ns"]
                ):
                    errors.append(f"r{round_no} ack-only: stop-ack lifecycle gate failed")
            all_miss_indices[condition].extend(
                (round_no, item["index"])
                for item in ran["records"] if item["deadline_miss"]
            )
            run[condition] = ran
        idle = run["mps_idle"]
        ack = run["mps_ack_only"]
        retired = run["mps_retired"]
        triplets.append((idle["deadline_misses"], ack["deadline_misses"], retired["deadline_misses"]))
        rows.append(
            f"| {round_no} | {'idle→ack→exited' if round_no % 2 else 'exited→ack→idle'} | "
            f"{idle['deadline_misses']} | {ack['deadline_misses']} | {retired['deadline_misses']} | "
            f"{idle['response_ms']['p99']:.3f} | {ack['response_ms']['p99']:.3f} | {retired['response_ms']['p99']:.3f} |"
        )
    if len(hosts) != 1:
        errors.append(f"conditions span multiple hosts: {sorted(hosts)}")
    positive = sum(ack > retired for _, ack, retired in triplets)
    negative = sum(ack < retired for _, ack, retired in triplets)
    ties = len(triplets) - positive - negative
    p_value = exact_one_sided_sign_p(positive, negative)
    remaining_rounds = planned_rounds - completed_rounds
    best_case_p = exact_one_sided_sign_p(positive + remaining_rounds, negative)
    retired_idle_positive = sum(retired > idle for idle, _, retired in triplets)
    retired_idle_negative = sum(retired < idle for idle, _, retired in triplets)
    retired_idle_ties = len(triplets) - retired_idle_positive - retired_idle_negative
    retired_idle_p = exact_one_sided_sign_p(retired_idle_positive, retired_idle_negative)
    ack_excess = not errors and positive > negative and p_value < protocol["primary_gate"]["one_sided_exact_paired_sign_p_below"]
    idle_total = sum(idle for idle, _, _ in triplets)
    ack_total = sum(ack for _, ack, _ in triplets)
    retired_total = sum(retired for _, _, retired in triplets)
    if errors:
        decision = "INVALID: control gate failed; do not infer lifecycle causality."
        grade = "invalid"
    elif remaining_rounds and best_case_p >= protocol["primary_gate"]["one_sided_exact_paired_sign_p_below"]:
        decision = "EARLY STOP: even if every remaining pair favored ack-only excess, the frozen primary p<0.05 gate could not pass. No exit-barrier or novelty claim."
        grade = "C early-stop diagnostic"
    elif remaining_rounds:
        decision = "INCOMPLETE: primary decision remains possible; do not adjudicate."
        grade = "incomplete"
    elif ack_excess:
        decision = "Early start after stop acknowledgement has excess misses versus later start after confirmed exit. The intervention also changes wait time; a time-matched control is required before attributing the benefit to the exit barrier."
        grade = "A for the local timing/lifecycle contrast only"
    elif retired_total > idle_total and retired_idle_positive > retired_idle_negative:
        decision = "Confirmed exit trends worse than idle: residual teardown-state hypothesis remains diagnostic; isolate CUDA reset versus MPS detach before claiming a mechanism."
        grade = "B diagnostic"
    elif idle_total == 0 and ack_total == 0 and retired_total == 0:
        decision = "Prior lifecycle signal did not reproduce; do not claim phase attribution."
        grade = "B diagnostic"
    else:
        decision = "No primary ack-only excess; do not claim a process-exit barrier benefit. Inspect the controls before proposing a new protocol."
        grade = "B diagnostic"
    lines = [
        "# Confirm52 MPS client lifecycle boundary",
        "",
        "Same-node, same-seed, reversed-order triplets; each condition has 1,000 P60/D35 clean-PUSCH releases and zero intended optional GPU work during the timed epoch.",
        "",
        "| triplet | order | idle misses | ack-only misses | exited misses | idle p99 ms | ack-only p99 ms | exited p99 ms |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"Misses idle / ack-only / confirmed-exit: {idle_total} / {ack_total} / {retired_total} of {len(triplets) * radio['iterations_per_condition']} each.",
        f"Primary paired signs ack-only>exited / ack-only<exited / ties: {positive}/{negative}/{ties}; exact one-sided p={p_value:.8f}.",
        f"Completed/planned triplets: {completed_rounds}/{planned_rounds}; best-case final one-sided p={best_case_p:.8f} if all remaining pairs favor ack-only excess.",
        f"Secondary diagnostic signs exited>idle / exited<idle / ties: {retired_idle_positive}/{retired_idle_negative}/{retired_idle_ties}; descriptive p={retired_idle_p:.8f}.",
        f"Control errors: {len(errors)}. Evidence grade: {grade}.",
        f"Decision: {decision}",
        "",
        *(f"{condition} miss indices (round, zero-based release): {all_miss_indices[condition]}." for condition in conditions),
        "",
        "A launcher exit-confirmation timestamp after first release does not prove actual overlap. Waiting for actual exit also delays first release, so this contrast cannot separate an exit-barrier effect from a longer quiet period. It does not identify an internal CUDA/MPS operation, prove AI throughput gain, or establish a universal hard deadline bound.",
    ]
    if errors:
        lines.extend(["", "Errors:", *(f"- {error}" for error in errors)])
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"confirm52 idle={idle_total} ack={ack_total} exited={retired_total} p={p_value:.8f} "
        f"control_errors={len(errors)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
