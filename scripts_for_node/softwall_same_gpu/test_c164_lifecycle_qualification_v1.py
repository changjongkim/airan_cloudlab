#!/usr/bin/env python3.11

import dataclasses
import unittest

from c164_lifecycle_qualification_v1 import (
    REQUIRED_COMPONENTS, REQUIRED_FLAGS, QualificationProfile,
    check_optional_admission, invalidate_for_restart, observe_runtime_bound,
    record_evidence, start_mode,
)


def profile(lifecycle="warm_persistent", node="nid-test", max_idle_ns=1000):
    return QualificationProfile(
        mode_id=f"mode-{lifecycle}-{node}", lifecycle=lifecycle, node_id=node,
        hardware_fingerprint="1" * 64, software_fingerprint="2" * 64,
        placement_fingerprint="3" * 64,
        bounds_ns={name: 100 for name in REQUIRED_COMPONENTS},
        minimum_samples={name: 2 for name in REQUIRED_COMPONENTS},
        max_idle_ns=max_idle_ns,
    )


def qualify(value, now_ns=10):
    return record_evidence(
        value, expected_generation=value.generation,
        counts={name: 2 for name in REQUIRED_COMPONENTS},
        maxima_ns={name: 99 for name in REQUIRED_COMPONENTS},
        passed_flags=frozenset(REQUIRED_FLAGS), now_ns=now_ns,
    )


class LifecycleQualificationTests(unittest.TestCase):
    def test_new_mode_cannot_admit_optional_work(self):
        state = start_mode(profile(), 1, 0)
        decision = check_optional_admission(
            state, token="none", profile_fingerprint=state.profile.fingerprint,
            expected_epoch=1, expected_generation=0, now_ns=1,
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "mode_not_qualified")

    def test_complete_evidence_issues_scoped_token(self):
        decision = qualify(start_mode(profile(), 1, 0))
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.state.status, "qualified")
        admitted = check_optional_admission(
            decision.state, token=decision.token,
            profile_fingerprint=decision.state.profile.fingerprint,
            expected_epoch=1, expected_generation=1, now_ns=11,
        )
        self.assertTrue(admitted.accepted)

    def test_partial_evidence_never_qualifies(self):
        state = start_mode(profile(), 1, 0)
        decision = record_evidence(
            state, expected_generation=0, counts={"nrx": 2},
            maxima_ns={"nrx": 90}, passed_flags=frozenset(REQUIRED_FLAGS),
            now_ns=5,
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.state.status, "qualifying")

    def test_bound_violation_quarantines_mode(self):
        state = start_mode(profile(), 1, 0)
        maxima = {name: 99 for name in REQUIRED_COMPONENTS}
        maxima["recovery"] = 101
        decision = record_evidence(
            state, expected_generation=0,
            counts={name: 2 for name in REQUIRED_COMPONENTS},
            maxima_ns=maxima, passed_flags=frozenset(REQUIRED_FLAGS), now_ns=5,
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.state.status, "quarantined")

    def test_stale_generation_and_wrong_fingerprint_do_not_admit(self):
        qualified = qualify(start_mode(profile(), 1, 0))
        for generation, fingerprint, reason in (
            (0, qualified.state.profile.fingerprint, "stale_generation"),
            (1, "f" * 64, "profile_fingerprint_mismatch"),
        ):
            with self.subTest(reason=reason):
                result = check_optional_admission(
                    qualified.state, token=qualified.token,
                    profile_fingerprint=fingerprint, expected_epoch=1,
                    expected_generation=generation, now_ns=11,
                )
                self.assertFalse(result.accepted)
                self.assertEqual(result.reason, reason)

    def test_restart_invalidates_old_token_and_requires_new_evidence(self):
        old = qualify(start_mode(profile(), 1, 0))
        restart_profile = profile("mps_restart_first")
        restarted = invalidate_for_restart(
            old.state, new_epoch=2, now_ns=20,
            lifecycle="mps_restart_first", new_profile=restart_profile,
        )
        self.assertTrue(restarted.accepted)
        self.assertEqual(restarted.state.status, "qualifying")
        rejected = check_optional_admission(
            restarted.state, token=old.token,
            profile_fingerprint=restart_profile.fingerprint,
            expected_epoch=2, expected_generation=0, now_ns=21,
        )
        self.assertFalse(rejected.accepted)

    def test_idle_expiry_atomically_marks_token_stale(self):
        qualified = qualify(start_mode(profile(max_idle_ns=10), 1, 0))
        result = check_optional_admission(
            qualified.state, token=qualified.token,
            profile_fingerprint=qualified.state.profile.fingerprint,
            expected_epoch=1, expected_generation=1, now_ns=21,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.state.status, "stale")
        self.assertEqual(result.state.generation, 2)

    def test_runtime_bound_violation_revokes_token(self):
        qualified = qualify(start_mode(profile(), 1, 0))
        result = observe_runtime_bound(
            qualified.state, component="nrx", observed_ns=101, now_ns=20,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.state.status, "quarantined")
        retry = check_optional_admission(
            result.state, token=qualified.token,
            profile_fingerprint=result.state.profile.fingerprint,
            expected_epoch=1, expected_generation=result.state.generation,
            now_ns=21,
        )
        self.assertFalse(retry.accepted)

    def test_node_and_lifecycle_are_part_of_profile_fingerprint(self):
        base = profile()
        self.assertNotEqual(base.fingerprint, profile(node="nid-other").fingerprint)
        self.assertNotEqual(base.fingerprint, profile("cold_first").fingerprint)

    def test_profile_requires_every_component(self):
        value = profile()
        incomplete = dict(value.bounds_ns)
        del incomplete["ai512"]
        with self.assertRaises(ValueError):
            dataclasses.replace(value, bounds_ns=incomplete)


if __name__ == "__main__":
    unittest.main()
