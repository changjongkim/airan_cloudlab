#!/usr/bin/env python3.11
"""Explore lifecycle-token transitions before the C164 physical campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from c164_lifecycle_qualification_v1 import (
    LIFECYCLES, REQUIRED_COMPONENTS, REQUIRED_FLAGS, QualificationProfile,
    check_optional_admission, invalidate_for_restart, observe_runtime_bound,
    record_evidence, start_mode,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(lifecycle: str, node: str) -> QualificationProfile:
    return QualificationProfile(
        mode_id=f"c164-{lifecycle}-{node}", lifecycle=lifecycle, node_id=node,
        hardware_fingerprint=hashlib.sha256(f"hardware:{node}".encode()).hexdigest(),
        software_fingerprint=hashlib.sha256(b"c164-model-software").hexdigest(),
        placement_fingerprint=hashlib.sha256(b"gpu0,1 owners;gpu2 recovery+qwen;gpu3 nrx").hexdigest(),
        bounds_ns={name: (75_000_000 if name == "ai512" else 45_000_000)
                   for name in REQUIRED_COMPONENTS},
        minimum_samples={name: 2 for name in REQUIRED_COMPONENTS},
        max_idle_ns=30_000_000_000,
    )


def complete(state, now_ns: int):
    maxima = {name: state.profile.bounds_ns[name] - 1
              for name in REQUIRED_COMPONENTS}
    return record_evidence(
        state, expected_generation=state.generation,
        counts={name: 2 for name in REQUIRED_COMPONENTS}, maxima_ns=maxima,
        passed_flags=frozenset(REQUIRED_FLAGS), now_ns=now_ns,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "results/softwall_multigpu/c164_lifecycle_model_v1.json"
    transitions = []
    unsafe_admissions = 0
    tokens = set()
    for lifecycle_index, lifecycle in enumerate(LIFECYCLES, start=1):
        started = start_mode(profile(lifecycle, "nid-model-a"), lifecycle_index, 0)
        cold = check_optional_admission(
            started, token="invalid", profile_fingerprint=started.profile.fingerprint,
            expected_epoch=started.epoch, expected_generation=started.generation,
            now_ns=1,
        )
        unsafe_admissions += int(cold.accepted)
        qualified = complete(started, 10)
        if qualified.token in tokens:
            raise AssertionError("qualification token collision")
        tokens.add(qualified.token)
        valid = check_optional_admission(
            qualified.state, token=qualified.token,
            profile_fingerprint=qualified.state.profile.fingerprint,
            expected_epoch=qualified.state.epoch,
            expected_generation=qualified.state.generation, now_ns=11,
        )
        invalid = []
        for label, token, fingerprint, epoch, generation in (
            ("wrong_token", "0" * 64, qualified.state.profile.fingerprint,
             qualified.state.epoch, qualified.state.generation),
            ("wrong_profile", qualified.token, "f" * 64,
             qualified.state.epoch, qualified.state.generation),
            ("stale_epoch", qualified.token, qualified.state.profile.fingerprint,
             qualified.state.epoch - 1, qualified.state.generation),
            ("stale_generation", qualified.token, qualified.state.profile.fingerprint,
             qualified.state.epoch, qualified.state.generation - 1),
        ):
            decision = check_optional_admission(
                qualified.state, token=token, profile_fingerprint=fingerprint,
                expected_epoch=epoch, expected_generation=generation, now_ns=11,
            )
            unsafe_admissions += int(decision.accepted)
            invalid.append({"case": label, "accepted": decision.accepted,
                            "reason": decision.reason})
        idle = check_optional_admission(
            qualified.state, token=qualified.token,
            profile_fingerprint=qualified.state.profile.fingerprint,
            expected_epoch=qualified.state.epoch,
            expected_generation=qualified.state.generation,
            now_ns=qualified.state.last_activity_ns
                   + qualified.state.profile.max_idle_ns + 1,
        )
        unsafe_admissions += int(idle.accepted)
        violation = observe_runtime_bound(
            qualified.state, component="recovery",
            observed_ns=qualified.state.profile.bounds_ns["recovery"] + 1,
            now_ns=20,
        )
        post_violation = check_optional_admission(
            violation.state, token=qualified.token,
            profile_fingerprint=violation.state.profile.fingerprint,
            expected_epoch=violation.state.epoch,
            expected_generation=violation.state.generation, now_ns=21,
        )
        unsafe_admissions += int(post_violation.accepted)
        transitions.append({
            "lifecycle": lifecycle,
            "cold_reason": cold.reason,
            "qualification_status": qualified.state.status,
            "qualified_token": qualified.token,
            "valid_admission": valid.accepted,
            "invalid_admissions": invalid,
            "idle_status": idle.state.status,
            "idle_reason": idle.reason,
            "violation_status": violation.state.status,
            "post_violation_accepted": post_violation.accepted,
        })

    warm = complete(start_mode(profile("warm_persistent", "nid-model-a"), 20, 0), 10)
    restart_rows = []
    for offset, lifecycle in enumerate(LIFECYCLES[1:], start=1):
        new = profile(lifecycle, "nid-model-a")
        restarted = invalidate_for_restart(
            warm.state, new_epoch=20 + offset, now_ns=20 + offset,
            lifecycle=lifecycle, new_profile=new,
        )
        old_token = check_optional_admission(
            restarted.state, token=warm.token,
            profile_fingerprint=new.fingerprint,
            expected_epoch=20 + offset,
            expected_generation=0, now_ns=30 + offset,
        )
        unsafe_admissions += int(old_token.accepted)
        replacement = complete(restarted.state, 40 + offset)
        restart_rows.append({
            "lifecycle": lifecycle, "restart_accepted": restarted.accepted,
            "old_token_accepted": old_token.accepted,
            "new_token_distinct": replacement.token != warm.token,
            "new_status": replacement.state.status,
        })

    checks = {
        "all_lifecycles_covered": len(transitions) == len(LIFECYCLES),
        "no_unqualified_optional_admission": unsafe_admissions == 0,
        "all_profiles_qualify_only_after_complete_evidence": all(
            row["cold_reason"] == "mode_not_qualified"
            and row["qualification_status"] == "qualified"
            and row["valid_admission"] for row in transitions
        ),
        "identity_mismatch_is_fail_closed": all(
            not item["accepted"]
            for row in transitions for item in row["invalid_admissions"]
        ),
        "idle_revokes_token": all(
            row["idle_status"] == "stale" for row in transitions
        ),
        "bound_violation_quarantines": all(
            row["violation_status"] == "quarantined"
            and not row["post_violation_accepted"] for row in transitions
        ),
        "restart_requires_distinct_requalification": all(
            row["restart_accepted"] and not row["old_token_accepted"]
            and row["new_token_distinct"] and row["new_status"] == "qualified"
            for row in restart_rows
        ),
        "qualification_tokens_unique": len(tokens) == len(LIFECYCLES),
    }
    sources = (
        "scripts_for_node/softwall_same_gpu/c164_lifecycle_qualification_v1.py",
        "scripts_for_node/softwall_same_gpu/test_c164_lifecycle_qualification_v1.py",
        "scripts_for_node/softwall_same_gpu/run_c164_lifecycle_model_v1.py",
    )
    result = {
        "schema": "softwall-c164-lifecycle-model-result-v1",
        "status": "C164_LIFECYCLE_MODEL_PASS" if all(checks.values()) else "C164_LIFECYCLE_MODEL_FAIL",
        "all_pass": all(checks.values()), "checks": checks,
        "counts": {
            "lifecycle_modes": len(transitions),
            "restart_transitions": len(restart_rows),
            "unsafe_optional_admissions": unsafe_admissions,
        },
        "transitions": transitions, "restart_transitions": restart_rows,
        "source_sha256": {name: sha256(root / name) for name in sources},
        "claim_boundary": (
            "This is a pure lifecycle/provenance state-model result. Physical "
            "cold/restart/idle timing remains unqualified until C164 GPU runs."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": result["status"], "checks": checks}, indent=2))
    raise SystemExit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
