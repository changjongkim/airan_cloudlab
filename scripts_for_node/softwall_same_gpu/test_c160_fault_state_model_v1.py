#!/usr/bin/env python3.11

import unittest

from c160_fault_state_model_v1 import (
    State,
    apply_success_batch,
    commit_lease,
    commit_radio,
    handle_rpc_timeout,
    physical_start,
    publish_completion_fence,
    retire_lease,
    timely_successes,
)


KEYS = frozenset({("h0", "r0"), ("h0", "r1"), ("h1", "r0")})


class C160FaultStateModelTests(unittest.TestCase):
    def setUp(self):
        self.base = State(epoch=7, generation=3, unresolved=KEYS)

    def test_stale_duplicate_and_future_outcome_do_not_mutate(self):
        for epoch in (6, 8):
            decision = apply_success_batch(
                self.base, event_epoch=epoch, expected_generation=3,
                success_keys=[("h0", "r0")], transaction_id=f"o-{epoch}",
            )
            self.assertFalse(decision.accepted)
            self.assertEqual(decision.state, self.base)
        first = apply_success_batch(
            self.base, event_epoch=7, expected_generation=3,
            success_keys=[("h0", "r0")], transaction_id="o-7",
        )
        duplicate = apply_success_batch(
            first.state, event_epoch=7, expected_generation=first.state.generation,
            success_keys=[("h0", "r0")], transaction_id="o-7",
        )
        self.assertTrue(duplicate.accepted)
        self.assertEqual(duplicate.state, first.state)

    def test_batch_is_atomic_on_unknown_or_duplicate_key(self):
        for keys in ([('h0', 'r0'), ('unknown', 'x')],
                     [('h0', 'r0'), ('h0', 'r0')]):
            decision = apply_success_batch(
                self.base, event_epoch=7, expected_generation=3,
                success_keys=keys, transaction_id="invalid",
            )
            self.assertFalse(decision.accepted)
            self.assertEqual(decision.state, self.base)

    def test_late_outcome_is_not_a_success(self):
        outcomes = [
            {"epoch": 7, "completed_ns": 99, "success": True,
             "key": ["h0", "r0"]},
            {"epoch": 7, "completed_ns": 101, "success": True,
             "key": ["h0", "r1"]},
            {"epoch": 6, "completed_ns": 90, "success": True,
             "key": ["h1", "r0"]},
        ]
        self.assertEqual(timely_successes(outcomes, 7, 100), (("h0", "r0"),))

    def test_latest_start_expiry_never_launches_and_can_retire(self):
        committed = commit_lease(
            self.base, expected_generation=3, lease_id="L", latest_start_ns=100,
            transaction_id="lease",
        ).state
        guarded = physical_start(committed, lease_id="L", accepted_ns=101)
        self.assertEqual(guarded.state.physical_launches, 0)
        self.assertEqual(guarded.state.lease.stage, "nonlaunched_fenced")
        retired = retire_lease(
            guarded.state, expected_generation=guarded.state.generation,
            lease_id="L", transaction_id="retire",
        )
        self.assertTrue(retired.accepted)
        self.assertIsNone(retired.state.lease)
        self.assertEqual(retired.state.physical_fences, 1)
        self.assertEqual(retired.state.retired_leases, 1)

    def test_post_fence_timeout_reconciles_and_quarantines(self):
        state = commit_lease(
            self.base, expected_generation=3, lease_id="L", latest_start_ns=100,
            transaction_id="lease",
        ).state
        state = physical_start(state, lease_id="L", accepted_ns=90).state
        state = publish_completion_fence(state, lease_id="L").state
        timeout = handle_rpc_timeout(
            state, lease_id="L", observed_marker="L", transaction_id="timeout"
        )
        self.assertTrue(timeout.accepted)
        self.assertTrue(timeout.state.ai_quarantined)
        self.assertIsNone(timeout.state.lease)
        self.assertEqual(timeout.state.physical_fences, 1)
        self.assertEqual(timeout.state.retired_leases, 1)

    def test_pre_fence_timeout_retains_lease_and_blocks_new_ai(self):
        state = commit_lease(
            self.base, expected_generation=3, lease_id="L", latest_start_ns=100,
            transaction_id="lease",
        ).state
        state = physical_start(state, lease_id="L", accepted_ns=90).state
        timeout = handle_rpc_timeout(
            state, lease_id="L", observed_marker=None, transaction_id="timeout"
        )
        self.assertFalse(timeout.accepted)
        self.assertTrue(timeout.state.ai_quarantined)
        self.assertIsNotNone(timeout.state.lease)
        self.assertEqual(timeout.state.retired_leases, 0)
        second = commit_lease(
            timeout.state, expected_generation=timeout.state.generation,
            lease_id="L2", latest_start_ns=200, transaction_id="lease2",
        )
        self.assertFalse(second.accepted)
        self.assertEqual(second.state, timeout.state)

    def test_wrong_marker_cannot_release_fenced_lease(self):
        state = commit_lease(
            self.base, expected_generation=3, lease_id="L", latest_start_ns=100,
            transaction_id="lease",
        ).state
        state = physical_start(state, lease_id="L", accepted_ns=90).state
        state = publish_completion_fence(state, lease_id="L").state
        timeout = handle_rpc_timeout(
            state, lease_id="L", observed_marker="other", transaction_id="timeout"
        )
        self.assertFalse(timeout.accepted)
        self.assertIsNotNone(timeout.state.lease)

    def test_radio_commit_is_epoch_generation_checked_and_idempotent(self):
        stale = commit_radio(
            self.base, event_epoch=6, request_id="r0", expected_generation=3
        )
        self.assertEqual(stale.state, self.base)
        first = commit_radio(
            self.base, event_epoch=7, request_id="r0", expected_generation=3
        )
        second = commit_radio(
            first.state, event_epoch=7, request_id="r0", expected_generation=3
        )
        self.assertTrue(first.accepted)
        self.assertTrue(second.accepted)
        self.assertEqual(second.reason, "duplicate_radio_commit")
        self.assertEqual(second.state, first.state)


if __name__ == "__main__":
    unittest.main()
