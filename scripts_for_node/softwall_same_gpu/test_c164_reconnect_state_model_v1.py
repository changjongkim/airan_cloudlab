#!/usr/bin/env python3.11

import dataclasses
import unittest

from c164_reconnect_state_model_v1 import (
    DurableRecord, LeaseIdentity, ReconnectState, digest_payload,
    issue_lease, lose_channel, reconnect_and_reconcile, safety_invariants,
    worker_fence, worker_launch, worker_prepare,
)


def identity(token="lease-1", epoch=7, worker="worker-a", payload=None):
    return LeaseIdentity(
        token, epoch, digest_payload(payload or {"context": 128}), worker
    )


def issued(value=None):
    base = ReconnectState(lifecycle_epoch=7)
    lease = identity() if value is None else value
    return issue_lease(base, lease).state, lease


class ReconnectStateModelTests(unittest.TestCase):
    def assert_safe(self, state):
        self.assertTrue(all(safety_invariants(state).values()))

    def test_absent_record_never_releases_lease(self):
        state, _ = issued()
        state = lose_channel(state).state
        result = reconnect_and_reconcile(
            state, observed=None, worker_quiesced=True
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "absent_record_is_ambiguous")
        self.assertIsNotNone(result.state.lease)
        self.assertTrue(result.state.ai_quarantined)

    def test_prepared_record_is_durably_nonlaunch_fenced(self):
        state, lease = issued()
        state = worker_prepare(state, lease).state
        state = lose_channel(state).state
        result = reconnect_and_reconcile(
            state, observed=state.durable_record, worker_quiesced=True
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "prepared_aborted_with_nonlaunch_fence")
        self.assertIsNone(result.state.lease)
        self.assertEqual(result.state.total_physical_launches, 0)
        self.assertEqual(result.state.total_physical_fences, 1)
        self.assert_safe(result.state)

    def test_launched_record_is_retained_until_fence(self):
        state, lease = issued()
        state = worker_prepare(state, lease).state
        state = worker_launch(state, lease).state
        state = lose_channel(state).state
        first = reconnect_and_reconcile(
            state, observed=state.durable_record, worker_quiesced=True
        )
        self.assertFalse(first.accepted)
        self.assertEqual(first.reason, "launched_without_fence_retained")
        completed = worker_fence(first.state, lease).state
        second = reconnect_and_reconcile(
            completed, observed=completed.durable_record, worker_quiesced=True
        )
        self.assertTrue(second.accepted)
        self.assertEqual(second.state.total_physical_launches, 1)
        self.assertEqual(second.state.total_physical_fences, 1)
        self.assertEqual(second.state.total_retirements, 1)
        self.assert_safe(second.state)

    def test_fenced_before_disconnect_retires_once(self):
        state, lease = issued()
        state = worker_prepare(state, lease).state
        state = worker_launch(state, lease).state
        state = worker_fence(state, lease).state
        observed = state.durable_record
        state = lose_channel(state).state
        result = reconnect_and_reconcile(
            state, observed=observed, worker_quiesced=True
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.state.total_retirements, 1)
        duplicate = reconnect_and_reconcile(
            result.state, observed=observed, worker_quiesced=True
        )
        self.assertFalse(duplicate.accepted)
        self.assertEqual(duplicate.reason, "no_active_lease")

    def test_identity_mismatch_and_unlocked_worker_fail_closed(self):
        state, lease = issued()
        state = worker_prepare(state, lease).state
        state = lose_channel(state).state
        wrong = dataclasses.replace(
            state.durable_record,
            identity=identity(payload={"context": 512}),
        )
        for observed, quiesced, reason in (
            (wrong, True, "observed_identity_mismatch"),
            (state.durable_record, False, "worker_token_lock_required"),
        ):
            result = reconnect_and_reconcile(
                state, observed=observed, worker_quiesced=quiesced
            )
            self.assertFalse(result.accepted)
            self.assertEqual(result.reason, reason)
            self.assertEqual(result.state, state)

    def test_duplicate_launch_and_fence_do_not_increment(self):
        state, lease = issued()
        state = worker_prepare(state, lease).state
        launched = worker_launch(state, lease)
        duplicate_launch = worker_launch(launched.state, lease)
        self.assertEqual(duplicate_launch.state, launched.state)
        fenced = worker_fence(duplicate_launch.state, lease)
        duplicate_fence = worker_fence(fenced.state, lease)
        self.assertEqual(duplicate_fence.state, fenced.state)
        self.assertEqual(fenced.state.total_physical_launches, 1)
        self.assertEqual(fenced.state.total_physical_fences, 1)

    def test_quarantine_and_tombstone_block_new_or_reused_token(self):
        state, lease = issued()
        quarantined = lose_channel(state).state
        self.assertFalse(issue_lease(quarantined, identity("lease-2")).accepted)
        prepared = worker_prepare(quarantined, lease).state
        retired = reconnect_and_reconcile(
            prepared, observed=prepared.durable_record, worker_quiesced=True
        ).state
        self.assertFalse(issue_lease(retired, lease).accepted)
        self.assertTrue(issue_lease(retired, identity("lease-2")).accepted)

    def test_stale_lifecycle_epoch_is_rejected(self):
        state = ReconnectState(lifecycle_epoch=8)
        result = issue_lease(state, identity(epoch=7))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "stale_lifecycle_epoch")

    def test_illegal_or_rolled_back_journal_cannot_retire(self):
        state, lease = issued()
        state = worker_prepare(state, lease).state
        state = worker_launch(state, lease).state
        state = lose_channel(state).state
        illegal = DurableRecord(
            lease, "nonlaunch_fenced", state.durable_record.journal_seq + 1,
            physical_fence_count=1,
        )
        decision = reconnect_and_reconcile(
            state, observed=illegal, worker_quiesced=True
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "illegal_journal_stage_transition")

        completed = worker_fence(state, lease).state
        rollback = dataclasses.replace(completed.durable_record, journal_seq=2)
        decision = reconnect_and_reconcile(
            completed, observed=rollback, worker_quiesced=True
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "journal_rollback_detected")


if __name__ == "__main__":
    unittest.main()
