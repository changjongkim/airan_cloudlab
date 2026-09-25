#!/usr/bin/env python3
"""V15 pipelined client with an explicit one-unlaunched-token invariant.

V14 correctly kept a staged token across a short local horizon, but its poll
path could also enqueue another prepare while that token was retained.  If the
broker held a second request, the V14 worker discarded the reply because the
first staged slot was occupied.  V15 preserves the V14 wire protocol and fault
semantics while forbidding a new prepare whenever a staged or offered token is
already owned by the home.
"""

from __future__ import annotations

import time

from pipelined_global_trace_client import PipelinedGlobalTraceClient


class SingleTokenPipelinedGlobalTraceClient(PipelinedGlobalTraceClient):
    """Pipelined client with at most one held, not-yet-committed token."""

    ownership_contract = "at-most-one-staged-or-offered-token-per-home"

    def __init__(self, *args, **kwargs):
        self._suppressed_prepare_due_owned_token = 0
        self._maximum_unlaunched_tokens = 0
        super().__init__(*args, **kwargs)

    def peek_edf_fitting(self, now_ns, horizon_ns, guard_ns, bounds_ns):
        """Poll/revalidate one token; never prepare behind a retained token."""
        del now_ns  # Revalidation and prepare use fresh monotonic timestamps.
        begin_ns = time.perf_counter_ns()
        request = None
        abort = None
        submit = False
        with self._lock:
            if self.enabled and not self._closing:
                current_ns = time.perf_counter_ns()
                if self._staged is not None:
                    candidate = self._staged
                    if self._fits(
                            candidate, current_ns, horizon_ns, guard_ns, bounds_ns):
                        self._staged = None
                        request = candidate
                        self._offered[candidate["broker_token"]] = candidate
                    else:
                        bound = bounds_ns.get(candidate["context_length"])
                        if bound is None:
                            bound = bounds_ns.get(str(candidate["context_length"]))
                        permanently_stale = (
                            bound is None
                            or current_ns + bound + guard_ns
                            > candidate["deadline_ns"]
                        )
                        if permanently_stale:
                            self._staged = None
                            abort = candidate

                # The V15 ownership invariant.  A retained staged token or a
                # token handed to the controller suppresses speculative
                # prepare.  Inflight work may coexist with one staged token.
                no_unlaunched_token = (
                    self._staged is None and not self._offered
                )
                current_unlaunched = int(self._staged is not None) + len(self._offered)
                self._maximum_unlaunched_tokens = max(
                    self._maximum_unlaunched_tokens, current_unlaunched
                )
                if (
                    request is None
                    and abort is None
                    and not no_unlaunched_token
                    and not self._prepare_pending
                ):
                    self._suppressed_prepare_due_owned_token += 1
                if (
                    request is None
                    and abort is None
                    and no_unlaunched_token
                    and not self._prepare_pending
                ):
                    self._prepare_pending = True
                    submit = True

        if abort is not None:
            self._tasks.put(("abort", {"request": abort}))
        elif submit:
            self._tasks.put(("prepare", {
                "now_ns": time.perf_counter_ns(),
                "horizon_ns": horizon_ns,
                "guard_ns": guard_ns,
                "bounds_ns": dict(bounds_ns),
            }))
        self._record_handoff("poll_prepare", begin_ns, request is not None or submit)
        return request

    def rpc_telemetry(self):
        telemetry = super().rpc_telemetry()
        with self._lock:
            suppressed = self._suppressed_prepare_due_owned_token
            maximum = self._maximum_unlaunched_tokens
        telemetry["ownership_contract"] = self.ownership_contract
        telemetry["ownership_evidence"] = {
            "suppressed_prepare_due_owned_token": suppressed,
            "maximum_unlaunched_tokens": maximum,
        }
        state = telemetry["state"]
        state["unlaunched_tokens"] = int(state["staged"]) + state["offered"]
        state["ownership_invariant_holds"] = (
            state["unlaunched_tokens"] <= 1 and maximum <= 1
        )
        return telemetry
