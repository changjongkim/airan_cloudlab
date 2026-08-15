#!/usr/bin/env python3
"""Deterministic correctness and fault tests for the DART transaction core."""

from __future__ import annotations

import argparse
import json
import random
import threading
import unittest
from pathlib import Path

from dart_runtime import (
    BackgroundUnit,
    CommitState,
    DartRequest,
    DartRuntime,
    EndpointState,
    FallbackCalendar,
    ProfileTable,
    ServiceProfile,
)


NS_PER_MS = 1_000_000


def make_request(
    slot: int,
    epoch: int = 1,
    deadline_ms: float = 10.0,
    release_ns: int | None = None,
):
    release = slot * 20 * NS_PER_MS if release_ns is None else release_ns
    deadline = release + round(deadline_ms * NS_PER_MS)
    return DartRequest(
        slot_id=slot,
        epoch=epoch,
        graph_id=1,
        tensor_class=1,
        release_ns=release,
        deadline_ns=deadline,
        fallback_latest_start_ns=deadline - NS_PER_MS - 50_000,
        payload_checksum=slot & 0xFFFFFFFF,
    )


def make_runtime(
    ring_depth=4,
    service_ns=2 * NS_PER_MS,
    fallback_capacity=4,
):
    endpoints = [EndpointState("e0", ring_depth), EndpointState("e1", ring_depth)]
    profiles = {}
    for endpoint in endpoints:
        profiles[(endpoint.endpoint_id, 1, 1, "isolated")] = ServiceProfile(
            forward_ns=50_000,
            service_ns=service_ns,
            backward_ns=25_000,
            control_ns=10_000,
            positive_error_ns=15_000,
        )
    fallback = FallbackCalendar(
        capacity=fallback_capacity,
        service_ns=NS_PER_MS,
        commit_guard_ns=50_000,
    )
    return DartRuntime(
        endpoints,
        ProfileTable(profiles),
        commit_guard_ns=50_000,
        fallback_calendar=fallback,
    )


class DartRuntimeTests(unittest.TestCase):
    def test_descriptor_round_trip_is_exactly_64_bytes(self):
        request = make_request(7, epoch=19)
        encoded = request.pack()
        self.assertEqual(len(encoded), 64)
        self.assertEqual(DartRequest.unpack(encoded), request)

    def test_predicted_finish_routes_to_available_endpoint(self):
        runtime = make_runtime(ring_depth=2)
        runtime.endpoints["e0"].blocking_until_ns = 100 * NS_PER_MS
        request = make_request(0)
        transaction = runtime.submit(request, request.release_ns)
        self.assertIsNotNone(transaction.reservation)
        self.assertEqual(transaction.reservation.endpoint_id, "e1")

    def test_shortest_queue_uses_outstanding_count_not_predicted_tail(self):
        runtime = make_runtime(ring_depth=2)
        first = make_request(0, deadline_ms=1000)
        first_transaction = runtime.submit(first, 0, policy="static")
        self.assertEqual(first_transaction.reservation.endpoint_id, "e0")
        runtime.endpoints["e1"].blocking_until_ns = 100 * NS_PER_MS
        second = make_request(1, deadline_ms=1000)
        second_transaction = runtime.submit(
            second, second.release_ns, policy="shortest_queue"
        )
        self.assertIsNotNone(second_transaction.reservation)
        self.assertEqual(second_transaction.reservation.endpoint_id, "e1")

    def test_atomic_reservation_never_overdraws_credit(self):
        endpoints = [EndpointState("e0", ring_depth=2)]
        profiles = ProfileTable({
            ("e0", 1, 1, "isolated"): ServiceProfile(0, NS_PER_MS, 0),
        })
        runtime = DartRuntime(endpoints, profiles, commit_guard_ns=0)
        barrier = threading.Barrier(9)
        transactions = []
        lock = threading.Lock()

        def submit(slot):
            request = make_request(slot, deadline_ms=100)
            barrier.wait()
            transaction = runtime.submit(request, 0)
            with lock:
                transactions.append(transaction)

        threads = [threading.Thread(target=submit, args=(slot,)) for slot in range(8)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        admitted = [item for item in transactions if item.reservation is not None]
        self.assertEqual(len(admitted), 2)
        snapshot = endpoints[0].snapshot()
        self.assertEqual(snapshot["free_input_slots"], 0)
        self.assertEqual(snapshot["free_output_slots"], 0)
        self.assertEqual(snapshot["outstanding"], 2)

    def test_jit_fallback_and_single_winner(self):
        runtime = make_runtime()
        request = make_request(0)
        transaction = runtime.submit(request, 0)
        self.assertFalse(runtime.start_fallback(transaction, request.fallback_latest_start_ns - 1))
        self.assertTrue(runtime.start_fallback(transaction, request.fallback_latest_start_ns))
        self.assertTrue(runtime.complete_nrx(transaction, 0, 1, request.deadline_ns - 10))
        self.assertFalse(runtime.complete_conventional(transaction, request.deadline_ns - 5))
        self.assertEqual(transaction.commit_state, CommitState.NRX_WON)

    def test_dual_reservation_rejects_remote_when_fallback_is_full(self):
        runtime = make_runtime(ring_depth=4, fallback_capacity=1)
        first = runtime.submit(make_request(0, release_ns=0), 0)
        second = runtime.submit(make_request(1, release_ns=0), 0)
        self.assertIsNotNone(first.reservation)
        self.assertIsNotNone(first.fallback_reservation)
        self.assertIsNone(second.reservation)
        self.assertEqual(runtime.fallback_calendar.snapshot()["outstanding"], 1)
        self.assertEqual(
            sum(endpoint.snapshot()["outstanding"] for endpoint in runtime.endpoints.values()),
            1,
        )

    def test_nrx_win_releases_unstarted_fallback_credit(self):
        runtime = make_runtime(ring_depth=4, fallback_capacity=1)
        first_request = make_request(0, release_ns=0)
        first = runtime.submit(first_request, 0)
        self.assertTrue(
            runtime.complete_nrx(
                first,
                first_request.slot_id,
                first_request.epoch,
                3 * NS_PER_MS,
            )
        )
        self.assertEqual(runtime.fallback_calendar.snapshot()["outstanding"], 0)
        second = runtime.submit(make_request(1, release_ns=0), 0)
        self.assertIsNotNone(second.reservation)

    def test_running_fallback_holds_credit_until_completion(self):
        runtime = make_runtime(ring_depth=4, fallback_capacity=1)
        first_request = make_request(0, release_ns=0)
        first = runtime.submit(first_request, 0)
        self.assertTrue(
            runtime.start_fallback(first, first_request.fallback_latest_start_ns)
        )
        self.assertTrue(
            runtime.complete_nrx(
                first,
                first_request.slot_id,
                first_request.epoch,
                first_request.deadline_ns - 25_000,
            )
        )
        self.assertEqual(runtime.fallback_calendar.snapshot()["outstanding"], 1)
        blocked = runtime.submit(make_request(1, release_ns=0), 0)
        self.assertIsNone(blocked.reservation)
        self.assertFalse(
            runtime.complete_conventional(first, first_request.deadline_ns - 10_000)
        )
        self.assertEqual(runtime.fallback_calendar.snapshot()["outstanding"], 0)
        admitted = runtime.submit(make_request(2, release_ns=0), 0)
        self.assertIsNotNone(admitted.reservation)

    def test_invisible_payload_cannot_commit(self):
        runtime = make_runtime()
        request = make_request(0)
        transaction = runtime.submit(request, request.release_ns)
        self.assertFalse(
            runtime.complete_nrx(
                transaction,
                request.slot_id,
                request.epoch,
                request.deadline_ns - NS_PER_MS,
                payload_visible=False,
            )
        )
        self.assertTrue(
            runtime.start_fallback(transaction, request.fallback_latest_start_ns)
        )
        self.assertTrue(
            runtime.complete_conventional(transaction, request.deadline_ns - 10_000)
        )

    def test_restart_invalidates_old_endpoint_completion(self):
        runtime = make_runtime()
        request = make_request(0)
        transaction = runtime.submit(request, request.release_ns)
        endpoint = runtime.endpoints[transaction.reservation.endpoint_id]
        endpoint.restart(request.release_ns + NS_PER_MS)
        self.assertFalse(
            runtime.complete_nrx(
                transaction,
                request.slot_id,
                request.epoch,
                request.deadline_ns - NS_PER_MS,
            )
        )
        self.assertTrue(
            runtime.start_fallback(transaction, request.fallback_latest_start_ns)
        )
        self.assertTrue(
            runtime.complete_conventional(transaction, request.deadline_ns - 10_000)
        )

    def test_late_result_cannot_commit(self):
        runtime = make_runtime()
        request = make_request(0)
        transaction = runtime.submit(request, 0)
        self.assertFalse(runtime.complete_nrx(transaction, 0, 1, request.deadline_ns + 1))
        self.assertEqual(transaction.commit_state, CommitState.DEADLINE_MISS)

    def test_lease_never_crosses_guard(self):
        runtime = make_runtime()
        units = [
            BackgroundUnit("small", 100_000, 1.0),
            BackgroundUnit("large", 600_000, 9.0),
        ]
        lease = runtime.try_lease(
            "e0",
            now_ns=0,
            earliest_latest_start_ns=500_000,
            qmax_ns=1_000_000,
            drain_guard_ns=50_000,
            units=units,
        )
        self.assertIsNotNone(lease)
        self.assertEqual(lease.unit_id, "small")
        self.assertLessEqual(lease.predicted_finish_ns + 50_000, 500_000)


def run_fault_campaign(iterations: int, seed: int) -> dict:
    rng = random.Random(seed)
    wrong_commit = 0
    stale_accepted = 0
    duplicate_accepted = 0
    deadline_accepted = 0
    invisible_accepted = 0
    restarted_accepted = 0
    committed = {"nrx": 0, "conventional": 0, "deadline_miss": 0}

    for slot in range(iterations):
        runtime = make_runtime()
        request = make_request(slot, epoch=slot + 100)
        transaction = runtime.submit(request, request.release_ns)
        fault = rng.choice((
            "stale", "duplicate", "delayed", "invisible", "restart", "race"
        ))

        if fault == "stale":
            accepted = runtime.complete_nrx(
                transaction,
                request.slot_id,
                request.epoch - 1,
                request.deadline_ns - 100,
            )
            stale_accepted += int(accepted)
            runtime.start_fallback(transaction, request.fallback_latest_start_ns)
            runtime.complete_conventional(transaction, request.deadline_ns - 50)
        elif fault == "duplicate":
            first = runtime.complete_nrx(
                transaction,
                request.slot_id,
                request.epoch,
                request.deadline_ns - 100,
            )
            second = runtime.complete_nrx(
                transaction,
                request.slot_id,
                request.epoch,
                request.deadline_ns - 90,
            )
            duplicate_accepted += int(first and second)
        elif fault == "delayed":
            accepted = runtime.complete_nrx(
                transaction,
                request.slot_id,
                request.epoch,
                request.deadline_ns + 1,
            )
            deadline_accepted += int(accepted)
        elif fault == "invisible":
            accepted = runtime.complete_nrx(
                transaction,
                request.slot_id,
                request.epoch,
                request.deadline_ns - 100,
                payload_visible=False,
            )
            invisible_accepted += int(accepted)
            runtime.start_fallback(transaction, request.fallback_latest_start_ns)
            runtime.complete_conventional(transaction, request.deadline_ns - 50)
        elif fault == "restart":
            endpoint = runtime.endpoints[transaction.reservation.endpoint_id]
            endpoint.restart(request.release_ns + 1)
            accepted = runtime.complete_nrx(
                transaction,
                request.slot_id,
                request.epoch,
                request.deadline_ns - 100,
            )
            restarted_accepted += int(accepted)
            runtime.start_fallback(transaction, request.fallback_latest_start_ns)
            runtime.complete_conventional(transaction, request.deadline_ns - 50)
        else:
            runtime.start_fallback(transaction, request.fallback_latest_start_ns)
            if rng.getrandbits(1):
                nrx = runtime.complete_nrx(
                    transaction,
                    request.slot_id,
                    request.epoch,
                    request.deadline_ns - 100,
                )
                conv = runtime.complete_conventional(transaction, request.deadline_ns - 90)
            else:
                conv = runtime.complete_conventional(transaction, request.deadline_ns - 100)
                nrx = runtime.complete_nrx(
                    transaction,
                    request.slot_id,
                    request.epoch,
                    request.deadline_ns - 90,
                )
            wrong_commit += int(nrx and conv)

        if transaction.commit_state is CommitState.NRX_WON:
            committed["nrx"] += 1
        elif transaction.commit_state is CommitState.CONV_WON:
            committed["conventional"] += 1
        elif transaction.commit_state is CommitState.DEADLINE_MISS:
            committed["deadline_miss"] += 1

    result = {
        "iterations": iterations,
        "seed": seed,
        "wrong_commit": wrong_commit,
        "stale_accepted": stale_accepted,
        "duplicate_accepted": duplicate_accepted,
        "deadline_accepted": deadline_accepted,
        "invisible_accepted": invisible_accepted,
        "restarted_accepted": restarted_accepted,
        "committed": committed,
        "pass": not any((
            wrong_commit,
            stale_accepted,
            duplicate_accepted,
            deadline_accepted,
            invisible_accepted,
            restarted_accepted,
        )),
    }
    return result


def run_fallback_storm_campaign(groups: int, fanout: int, capacity: int) -> dict:
    over_admitted = 0
    reservation_leaks = 0
    for group in range(groups):
        runtime = make_runtime(
            ring_depth=max(fanout, capacity),
            service_ns=NS_PER_MS,
            fallback_capacity=capacity,
        )
        release_ns = group * 100 * NS_PER_MS
        transactions = [
            runtime.submit(
                make_request(
                    group * fanout + index,
                    epoch=group + 1,
                    deadline_ms=10.0,
                    release_ns=release_ns,
                ),
                release_ns,
            )
            for index in range(fanout)
        ]
        admitted = [item for item in transactions if item.reservation is not None]
        over_admitted += max(0, len(admitted) - capacity)
        for transaction in admitted:
            request = transaction.request
            runtime.start_fallback(transaction, request.fallback_latest_start_ns)
            runtime.complete_conventional(transaction, request.deadline_ns - 10_000)
        reservation_leaks += runtime.fallback_calendar.snapshot()["outstanding"]
    return {
        "groups": groups,
        "fanout": fanout,
        "capacity": capacity,
        "over_admitted": over_admitted,
        "reservation_leaks": reservation_leaks,
        "pass": over_admitted == 0 and reservation_leaks == 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output")
    args = parser.parse_args()

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DartRuntimeTests)
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not test_result.wasSuccessful():
        raise SystemExit(1)
    campaign = run_fault_campaign(args.fault_iterations, args.seed)
    campaign["fallback_storm"] = run_fallback_storm_campaign(
        groups=max(1, args.fault_iterations // 100),
        fanout=8,
        capacity=2,
    )
    campaign["pass"] = campaign["pass"] and campaign["fallback_storm"]["pass"]
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(campaign, indent=2), encoding="utf-8")
    print(json.dumps(campaign, sort_keys=True))
    if not campaign["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
