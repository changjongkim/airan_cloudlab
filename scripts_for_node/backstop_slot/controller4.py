#!/usr/bin/env python3
"""Slot-scale controller (v4 = v3 + AI packing): dispatches NeuralRx to lanes and grants AI time.

AI packing (``ai_dispatch_order: pack``): a request goes to the lowest-numbered GPU that is
still predicted to finish it within ``ai_pack_fraction`` of its time limit, instead of the GPU
that finishes it first.  AI then concentrates on few GPUs and the others' NeuralRx lanes and
conventional receivers run without AI next to them; lanes serve any cell, so rescues move there.

Dynamic-share baseline (``dynamic_share``: shares [low, high], ``lag_periods``, ``window_periods``):
each GPU has one pre-loaded AI worker per share (``ai_worker_dyn.py``).  The controller makes
the low-share worker the active one while the GPU's radio load is full and the high-share one
otherwise, judged over the last ``window_periods`` periods as they were ``lag_periods`` ago.


Additions over ``controller.py``:
  * partial load: only (cell, period) pairs that carry a TB are scheduled (``activity.py``),
    and the conventional delay budget of a period follows that period's active cells;
  * NeuralRx concurrency by deadline: ``nrx_bound_by_busy_ms`` = [alone, with a second lane];
    a GPU's second lane is used only when both TBs still meet their rescue deadlines
    (``nrx_flags.second_lane`` = deadline), instead of always (= always);
  * NeuralRx order by value: ``nrx_flags.rank`` = value serves TBs with fewer failed code
    blocks first (``nrx_max_cb_fail`` may then admit more than one).
  * NeuralRx-primary cells (``nrx_mode: primary``): every TB of the cell is decoded by
    NeuralRx from arrival, whatever the conventional result (the conventional receiver stays
    the fallback); it must finish by the rescue deadline.
  * queue feasibility (``ai_unit_gating.queue_check``): an AI unit size is granted only if
    every NeuralRx TB that is waiting for a lane, and the primary TBs of the next period,
    still finish by their deadlines when the lanes are slowed by that unit size.

Two deadlines per TB (from data arrival):
  L1 deadline      the conventional CRC must be ready (UL indication).
  rescue deadline  a NeuralRx rescue must finish to replace the HARQ
                   retransmission (the decision point for the retransmission
                   grant); equal to the L1 deadline unless configured longer.

NeuralRx mechanisms (switched by ``nrx_flags``; presets by policy name):
  start   arrival | fail_or_latest | fail_only
  skip    skip NeuralRx when the conventional receiver passes first
  admit   never launch past the latest start (rescue deadline - bound)
  value   after a conventional failure, rescue only if at most
          ``nrx_max_cb_fail`` code blocks failed
  lanes   partner (lanes of the partner GPU) | any (all lanes, EDF)

AI policies:
  backstop_corun  (Our Scheme) AI runs in its MPS share next to radio work; a
                  grant is given only if every NeuralRx TB it may overlap
                  still meets its rescue deadline with the co-run bound
  backstop_units  (Our Scheme, unit-aware) as backstop_corun, but every grant also
                  names the largest AI chunk allowed: a chunk size is allowed only
                  if its co-run bound keeps every overlapping NeuralRx TB inside its
                  rescue deadline, and chunks not marked conventional-safe are kept
                  out of the conventional phase (they must end before the next
                  uplink arrival and may not start while conventional runs)
  backstop        AI only while the GPU has no radio work (Concordia-like)
  idle            fixed budget whenever the GPU has no radio work
  static          no grants; AI runs continuously in its MPS share
  none            no AI
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path

from slot_state import (
    A_BUDGET_NS, A_CUR_CHUNK, A_DONE_SEQ, A_GRANT_SEQ, A_HAS_WORK, A_MAX_CHUNK, A_NOT_AFTER, A_RUNNING,
    C_CONV_CBFAIL, C_CONV_STATUS, C_NRX_REASON, C_NRX_START, DROPPED, FAIL, H_ABORT,
    A_BACKLOG_TOKENS, A_DYN_ACTIVE, A_PULLED, A_RATE_TPS, R_ARRIVAL, R_GPU, R_LENGTH, R_SEQ,
    L_ASSIGN_SEQ, L_CELL, L_DONE_SEQ, L_PERIOD, PASS, PENDING, REASON_ARRIVAL,
    REASON_CONV_FAIL, REASON_LATEST_START, REASON_LOW_VALUE, SKIPPED, SlotState, now_ns,
)
from ul_profiles import PROFILES
from activity import activity_mask, gpu_load

WAITING, ELIGIBLE, ASSIGNED, DONE = 0, 1, 2, 3

PRESETS = {
    "off":            dict(start="none", skip=True, admit=False, value=False, lanes="partner"),
    "parallel":       dict(start="arrival", skip=False, admit=False, value=False, lanes="partner"),
    "parallel_admit": dict(start="arrival", skip=True, admit=True, value=False, lanes="any"),
    "rescue_naive":   dict(start="fail_or_latest", skip=True, admit=False, value=False, lanes="partner"),
    "rescue":         dict(start="fail_or_latest", skip=True, admit=True, value=False, lanes="any"),
    "rescue_value":   dict(start="fail_or_latest", skip=True, admit=True, value=True, lanes="any"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--num-cells", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--periods", type=int, required=True)
    parser.add_argument("--period-ns", type=int, required=True)
    parser.add_argument("--ready-slot", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    gpus = args.num_gpus
    period = args.period_ns
    ms = 1_000_000
    deadline = int(float(config["deadline_ms"]) * ms)
    rescue_deadline = int(float(config.get("rescue_deadline_ms", config["deadline_ms"])) * ms)
    bound = int(float(config["nrx_bound_ms"]) * ms)
    bound_corun = int(float(config.get("nrx_bound_corun_ms", 2.8)) * ms)
    slack = rescue_deadline - bound
    ai_policy = config.get("ai_policy", "none")
    guard = int(float(config.get("ai_guard_us", 50)) * 1e3)
    idle_budget = int(float(config.get("ai_idle_budget_ms", 1.0)) * ms)
    min_budget = int(float(config.get("ai_min_budget_ms", 0.12)) * ms)
    max_piece = int(float(config.get("ai_max_piece_ms", 1.0)) * ms)
    max_cb_fail = int(config.get("nrx_max_cb_fail", 1))
    unit_cfg = config.get("ai_unit_gating", {})
    corun_by_chunk = {int(c): int(float(v) * ms) for c, v in unit_cfg.get("nrx_bound_corun_ms", {}).items()}
    conv_safe = set(int(c) for c in unit_cfg.get("conv_safe_chunks", []))
    # When NeuralRx demand exceeds the free lanes, co-running AI would slow the lanes
    # and push waiting TBs past their latest start: yield the GPUs to NeuralRx.
    yield_to_waiting = bool(unit_cfg.get("yield_to_waiting_nrx", False))
    # Conventional delay budget: each period the conventional receivers of a GPU may be
    # delayed by (L1 deadline - margin - their completion time without overlapping AI).
    # An AI unit of chunk size c that overlaps them for x ms costs alpha[c] * x of it.
    # Chunks without an alpha keep the on/off rule (conv_safe_chunks).
    budget_cfg = unit_cfg.get("conv_budget") or {}
    alpha = {int(c): float(a) for c, a in budget_cfg.get("alpha", {}).items()}
    conv_alone = budget_cfg.get("conv_alone_ms", 0.0)
    budget_margin = float(budget_cfg.get("margin_ms", 0.2))
    delay_used: dict[tuple[int, int], float] = {}
    # NeuralRx slack budget (``nrx_budget``): a NeuralRx TB that overlaps an AI unit of chunk
    # size c for x ms is delayed by delta[c] * x, with delta[c] = 1 - bound alone / co-run
    # bound.  A running TB may be delayed until (deadline - start - bound alone) is used up;
    # a TB that has not started yet only needs the AI piece to end by its latest start.
    nrx_budget = bool(unit_cfg.get("nrx_budget", False))
    nrx_delay: dict[tuple[int, int], float] = {}

    def delay_budget(g: int, k: int) -> float:
        """Conventional delay (ns) AI may still add on GPU g in period k."""
        alone = conv_alone if not isinstance(conv_alone, dict) else conv_alone.get(
            str(load_of(g, k)), max(conv_alone.values()))
        return (float(config["deadline_ms"]) - budget_margin - float(alone)) * ms - delay_used.get((g, k), 0.0)

    def load_of(g: int, k: int) -> int:
        return int(load[g, k]) if 0 <= k < args.periods else len(own_cells[g])
    layer_bound = {int(c): int(float(b["layer"]) * ms)
                   for c, b in config.get("ai", {}).get("unit_bound_ms_by_chunk", {}).items()}
    flags = dict(PRESETS[config["nrx_policy"]])
    flags.update(config.get("nrx_flags", {}))
    state = SlotState(args.state, args.num_cells, args.periods, gpus)

    cells = config["cells"]
    conv_gpu = {int(c["cell"]): int(c["gpu"]) for c in cells}
    own_cells = {g: [int(c["cell"]) for c in cells if int(c["gpu"]) == g] for g in range(gpus)}
    weak = [int(c["cell"]) for c in cells
            if PROFILES[c["profile"]].neural_eligible and c.get("nrx_gpu") is not None]
    partner = {int(c["cell"]): int(c["nrx_gpu"]) for c in cells if int(c["cell"]) in weak}
    use_nrx = flags["start"] != "none" and bool(weak)
    per_gpu = int(config.get("lanes_per_gpu", 1))
    lanes = list(range(gpus * per_gpu)) if use_nrx else []
    lane_gpu = {lane: lane // per_gpu for lane in lanes}
    gpu_lanes = {g: [lane for lane in lanes if lane_gpu[lane] == g] for g in range(gpus)}
    horizon = max(2, math.ceil(rescue_deadline / period) + 1)   # periods a TB stays open
    mask = activity_mask(config)
    load = gpu_load(config, mask)
    bounds_busy = [int(float(b) * ms) for b in config.get("nrx_bound_by_busy_ms", [])]
    bound_alone = bounds_busy[0] if bounds_busy else bound
    bound_pair = bounds_busy[1] if len(bounds_busy) > 1 else bound
    slack = rescue_deadline - bound_alone
    second_lane = flags.get("second_lane", "always")
    by_value = flags.get("rank") == "value"
    two_lane_ai = unit_cfg.get("two_lane_corun_ms")      # None: no AI while two lanes run on a GPU
    primary = {int(c["cell"]) for c in cells if c.get("nrx_mode") == "primary" and int(c["cell"]) in weak}
    queue_check = bool(unit_cfg.get("queue_check", False))

    def queue_feasible(g: int, chunk_bound: int, now: int) -> bool:
        """Do all waiting NeuralRx TBs (and next period's primary TBs) still meet their deadlines
        if the lanes of GPU g run next to an AI unit with co-run bound ``chunk_bound``?"""
        free = []
        for lane in lanes:
            if lane_free(lane):
                free.append(now)
                continue
            cell, k = lane_job[lane]
            started = int(state.cells[cell, k, C_NRX_START]) or now
            slowed = lane_gpu[lane] == g or int(state.gpus[lane_gpu[lane]][A_RUNNING])
            free.append(started + (chunk_bound if slowed else bound_alone))
        jobs = [(epoch + k * period + rescue_deadline, now) for (cell, k), st in job_state.items() if st == ELIGIBLE]
        nxt = k_now + 1
        if primary and nxt < args.periods:
            release = epoch + nxt * period
            jobs += [(release + rescue_deadline, release) for cell in primary if mask[cell, nxt]]
        for due, release in sorted(jobs):
            i = min(range(len(free)), key=free.__getitem__)
            finish = max(free[i], release) + bound_alone
            if finish > due:
                return False
            free[i] = finish
        return True

    def conv_pending(g: int, k: int) -> bool:
        return 0 <= k < args.periods and any(
            mask[c, k] and int(state.cells[c, k, C_CONV_STATUS]) == PENDING for c in own_cells[g])

    job_state: dict[tuple[int, int], int] = {}
    reasons: dict[tuple[int, int], int] = {}
    counters = {"assigned": 0, "dropped": 0, "skipped": 0, "low_value": 0, "grants": 0, "queue_blocked": 0,
                "grant_budget_ms": 0.0, "loops": 0, "max_loop_us": 0.0}
    lane_seq = {lane: 0 for lane in lanes}
    lane_job: dict[int, tuple[int, int] | None] = {lane: None for lane in lanes}
    grant_seq = {g: 0 for g in range(gpus)}

    def lane_free(lane: int) -> bool:
        return int(state.lanes[lane][L_DONE_SEQ]) == lane_seq[lane]

    def assign(lane: int, cell: int, k: int) -> None:
        box = state.lanes[lane]
        state.cells[cell, k, C_NRX_REASON] = reasons[(cell, k)]
        box[L_CELL] = cell
        box[L_PERIOD] = k
        lane_seq[lane] += 1
        box[L_ASSIGN_SEQ] = lane_seq[lane]
        lane_job[lane] = (cell, k)
        job_state[(cell, k)] = ASSIGNED
        counters["assigned"] += 1

    def busy_lanes(g: int) -> list[int]:
        return [lane for lane in gpu_lanes.get(g, []) if not lane_free(lane)]

    def grant(g: int, budget: int, now: int, max_chunk: int = 0) -> None:
        box = state.gpus[g]
        box[A_MAX_CHUNK] = max_chunk
        box[A_BUDGET_NS] = budget
        box[A_NOT_AFTER] = now + guard
        grant_seq[g] += 1
        box[A_GRANT_SEQ] = grant_seq[g]
        counters["grants"] += 1
        counters["grant_budget_ms"] += budget / 1e6

    # Global AI dispatch: one arrival stream for the whole server; each request goes to
    # the GPU whose queue, served at that GPU's recent rate, finishes it first.  Under
    # Our Scheme the rates differ by GPU because radio work differs by GPU.
    ai_cfg = config.get("ai", {})
    dispatch = ai_policy != "none" and ai_cfg.get("dispatch") == "global"
    n_requests = 0
    if dispatch:
        from ai_arrivals import arrivals
        global_cfg = dict(ai_cfg, rate_per_s=float(ai_cfg["rate_per_s"]) * gpus,
                          seed=int(ai_cfg["seed"]) + 999)
        plan = arrivals(global_cfg, 0, args.periods * period)[:len(state.requests)]
        for i, (arrival, length) in enumerate(plan):
            state.requests[i, R_ARRIVAL] = arrival
            state.requests[i, R_LENGTH] = length
            state.requests[i, R_GPU] = -1
            state.requests[i, R_SEQ] = 1
        n_requests = len(plan)
        slo_ns = int(float(ai_cfg.get("slo_ms", 200.0)) * ms)
        prior_tps = float(ai_cfg.get("prior_tokens_per_s", 15000.0))
        pack = config.get("ai_dispatch_order") == "pack"
        pack_ns = float(config.get("ai_pack_fraction", 0.6)) * slo_ns
        assigned_cum = {g: [0] for g in range(gpus)}     # prefix sums of assigned lengths
        counters.update({"ai_dispatched": 0, "ai_rejected": 0})
    next_request = 0
    dynamic = config.get("dynamic_share")
    dynamic_lag = int((dynamic or {}).get("lag_periods", 0))
    dynamic_window = int((dynamic or {}).get("window_periods", 40))
    dynamic_seen = -1
    # A GPU is in a busy phase when its recent load is above the midpoint of the two phase levels
    # (0.9 of full load when the phases are not given).
    activity = config.get("activity") or {}
    dynamic_level = (0.5 * (float(activity.get("phase_high", 1.0)) + float(activity.get("prob", 1.0)))
                     if activity.get("mode") == "phased" else 0.9)
    if dynamic:
        for g in range(gpus):
            state.gpus[g][A_DYN_ACTIVE] = 1

    gc.collect()
    gc.disable()
    state.mark_ready(args.ready_slot)
    epoch = state.wait_epoch()
    end_time = epoch + args.periods * period + rescue_deadline + 2 * ms
    while True:
        loop_start = now_ns()
        if loop_start > end_time or state.header[H_ABORT]:
            break
        k_now = (loop_start - epoch) // period
        if k_now < 0:
            continue
        open_periods = [k for k in range(k_now - horizon + 1, k_now + 1) if 0 <= k < args.periods]
        # --- NeuralRx eligibility ------------------------------------------
        if use_nrx:
            for k in open_periods:
                release = epoch + k * period
                if loop_start > release + rescue_deadline:
                    continue
                for cell in weak:
                    key = (cell, k)
                    if not mask[cell, k]:
                        continue
                    status = job_state.get(key, WAITING)
                    if status >= ASSIGNED:
                        continue
                    if cell in primary:
                        if status == WAITING:
                            reasons[key] = REASON_ARRIVAL
                            job_state[key] = ELIGIBLE
                        continue
                    conv = int(state.cells[cell, k, C_CONV_STATUS])
                    if conv == PASS and flags["skip"]:
                        state.set_nrx(cell, k, SKIPPED, loop_start)
                        job_state[key] = DONE
                        counters["skipped"] += 1
                        continue
                    if status != WAITING:
                        continue
                    if flags["start"] == "arrival":
                        reasons[key] = REASON_ARRIVAL
                        job_state[key] = ELIGIBLE
                    elif conv == FAIL:
                        if flags["value"] and int(state.cells[cell, k, C_CONV_CBFAIL]) > max_cb_fail:
                            state.cells[cell, k, C_NRX_REASON] = REASON_LOW_VALUE
                            state.set_nrx(cell, k, DROPPED, loop_start)
                            job_state[key] = DONE
                            counters["low_value"] += 1
                            continue
                        reasons[key] = REASON_CONV_FAIL
                        job_state[key] = ELIGIBLE
                    elif flags["start"] == "fail_or_latest" and loop_start >= release + slack:
                        reasons[key] = REASON_LATEST_START
                        job_state[key] = ELIGIBLE
        # --- NeuralRx dispatch (earliest latest start first; by value if asked) ----
        if use_nrx:
            def order(item):
                latest, cell, k = item
                rank = 0
                if cell in primary:
                    rank = -1
                elif by_value and int(state.cells[cell, k, C_CONV_STATUS]) == FAIL:
                    rank = max(0, int(state.cells[cell, k, C_CONV_CBFAIL]) - 1)
                return (rank, latest, cell, k)
            pending = sorted(((epoch + k * period + slack, cell, k)
                              for (cell, k), st in job_state.items() if st == ELIGIBLE), key=order)
            # With value ordering, a TB with more failed code blocks (a lower chance of rescue)
            # takes a lane only once every conventional result of the open periods is in, so
            # it cannot get ahead of a better candidate that is about to appear.
            results_pending = by_value and any(
                mask[c, kk] and int(state.cells[c, kk, C_CONV_STATUS]) == PENDING
                for kk in open_periods for c in weak
                if c not in primary and loop_start <= epoch + kk * period + slack)
            for latest, cell, k in pending:
                if results_pending and order((latest, cell, k))[0] > 0:
                    continue
                if flags["admit"] and loop_start > latest:
                    state.set_nrx(cell, k, DROPPED, loop_start)
                    job_state[(cell, k)] = DONE
                    counters["dropped"] += 1
                    continue
                due = epoch + k * period + rescue_deadline
                if flags["lanes"] == "partner":
                    choices = list(gpu_lanes[partner[cell]])
                else:
                    speculative = int(state.cells[cell, k, C_CONV_STATUS]) == PENDING
                    choices = sorted(lanes, key=lambda lane: (
                        speculative and lane_gpu[lane] == conv_gpu[cell],
                        len(busy_lanes(lane_gpu[lane])),
                        int(state.gpus[lane_gpu[lane]][A_RUNNING]),
                        lane_gpu[lane] != partner[cell],
                    ))
                for lane in choices:
                    if not lane_free(lane):
                        continue
                    others = busy_lanes(lane_gpu[lane])
                    if others and second_lane == "deadline":
                        # Two NeuralRx on one GPU slow each other: both must still fit.
                        if loop_start + bound_pair > due:
                            continue
                        fits = True
                        for other in others:
                            oc, ok_ = lane_job[other]
                            started = int(state.cells[oc, ok_, C_NRX_START]) or loop_start
                            if started + bound_pair > epoch + ok_ * period + rescue_deadline:
                                fits = False
                        if not fits or conv_pending(lane_gpu[lane], k_now):
                            continue
                    assign(lane, cell, k)
                    break
        # --- dynamic-share baseline: which pre-loaded worker is active on each GPU ---
        if dynamic and k_now != dynamic_seen:
            dynamic_seen = k_now
            for g in range(gpus):
                hi = max(0, min(args.periods, int(k_now) - dynamic_lag))
                lo = max(0, hi - dynamic_window)
                full = hi <= lo or float(load[g, lo:hi].mean()) >= dynamic_level * len(own_cells[g])
                state.gpus[g][A_DYN_ACTIVE] = 1 if full else 2
        # --- AI grants -----------------------------------------------------
        if ai_policy == "backstop_corun":
            for g in range(gpus):
                box = state.gpus[g]
                if int(box[A_DONE_SEQ]) != grant_seq[g] or not int(box[A_HAS_WORK]):
                    continue
                now = now_ns()
                tight = False
                for lane in busy_lanes(g):
                    cell, k = lane_job[lane]
                    started = int(state.cells[cell, k, C_NRX_START]) or now
                    if started + bound_corun > epoch + k * period + rescue_deadline:
                        tight = True
                        break
                if tight:
                    continue                  # a tight NeuralRx runs without AI
                window_end = now + max_piece
                if use_nrx:
                    for k in open_periods:
                        release_k = epoch + k * period
                        cap = release_k + rescue_deadline - bound_corun
                        # A waiting TB matters only while its NeuralRx can still start.
                        if cap >= window_end or now > release_k + slack:
                            continue
                        for cell in weak:
                            if partner[cell] != g:
                                continue
                            if job_state.get((cell, k), WAITING) in (WAITING, ELIGIBLE) and \
                                    int(state.cells[cell, k, C_CONV_STATUS]) != PASS:
                                window_end = min(window_end, cap)
                budget = window_end - now - guard
                if budget >= min_budget:
                    grant(g, budget, now)
        elif ai_policy == "backstop_units":
            nrx_waiting = yield_to_waiting and any(st == ELIGIBLE for st in job_state.values())
            for g in range(gpus):
                if nrx_waiting:
                    break
                box = state.gpus[g]
                if int(box[A_DONE_SEQ]) != grant_seq[g] or not int(box[A_HAS_WORK]):
                    continue
                now = now_ns()
                next_release = epoch + (k_now + 1) * period
                conv_running = conv_pending(g, k_now)
                if two_lane_ai is None and len(busy_lanes(g)) >= 2:
                    continue                  # two NeuralRx fill the GPU: no AI next to them
                running = []
                running_keys = []
                for lane in busy_lanes(g):
                    cell, k = lane_job[lane]
                    started = int(state.cells[cell, k, C_NRX_START]) or now
                    running.append((started, epoch + k * period + rescue_deadline))
                    running_keys.append((cell, k))
                pending = []
                if use_nrx:
                    for k in open_periods:
                        release_k = epoch + k * period
                        if now > release_k + slack:
                            continue
                        for cell in weak:
                            if partner[cell] == g and mask[cell, k] \
                                    and job_state.get((cell, k), WAITING) in (WAITING, ELIGIBLE) \
                                    and (cell in primary or int(state.cells[cell, k, C_CONV_STATUS]) != PASS):
                                pending.append(release_k + rescue_deadline)
                                break
                chosen = None
                current = int(box[A_CUR_CHUNK])
                for c in sorted(corun_by_chunk, reverse=True):
                    if c < current:
                        break                 # the chunk in progress cannot use a smaller grant
                    bound_c = corun_by_chunk[c]
                    delta_c = max(1e-6, 1.0 - bound_alone / bound_c)
                    nrx_room = None
                    if nrx_budget:
                        nrx_room = min((
                            (due - started - bound_alone - nrx_delay.get(key, 0.0)) / delta_c
                            for (started, due), key in zip(running, running_keys)), default=None)
                    elif any(started + bound_c > due for started, due in running):
                        continue
                    if queue_check and not queue_feasible(g, bound_c, now):
                        counters["queue_blocked"] += 1
                        continue
                    window_end = now + max_piece
                    a = alpha.get(c)
                    if c in conv_safe or a == 0.0:
                        pass
                    elif a is None:
                        if conv_running:
                            continue
                        window_end = min(window_end, next_release - guard)
                    else:
                        if conv_running:
                            window_end = min(window_end, now + delay_budget(g, k_now) / a)
                        window_end = min(window_end, next_release + delay_budget(g, k_now + 1) / a)
                    for due in pending:
                        window_end = min(window_end, due - (bound_alone if nrx_budget else bound_c))
                    if nrx_room is not None:
                        window_end = min(window_end, now + nrx_room)
                    budget = int(window_end - now - guard)
                    # A chunk in progress needs room for one more unit; a new one for two.
                    need = (1 if c == int(box[A_CUR_CHUNK]) else 2) * layer_bound.get(c, 0) \
                        if c != min(corun_by_chunk) else min_budget
                    if budget >= need:
                        chosen = (c, budget)
                        break
                if chosen:
                    c, budget = chosen
                    if nrx_budget:
                        spent = max(1e-6, 1.0 - bound_alone / corun_by_chunk[c]) * budget
                        for key in running_keys:
                            nrx_delay[key] = nrx_delay.get(key, 0.0) + spent
                    a = alpha.get(c)
                    if a and c not in conv_safe:
                        if conv_running:
                            delay_used[(g, k_now)] = delay_used.get((g, k_now), 0.0) + a * min(
                                budget, max(0, next_release - now))
                        spill = now + budget - next_release
                        if spill > 0:
                            delay_used[(g, k_now + 1)] = delay_used.get((g, k_now + 1), 0.0) + a * spill
                    grant(g, budget, now, c)
        elif ai_policy in ("backstop", "idle"):
            for g in range(gpus):
                box = state.gpus[g]
                if int(box[A_DONE_SEQ]) != grant_seq[g] or not int(box[A_HAS_WORK]):
                    continue
                release = epoch + k_now * period
                radio_busy = bool(busy_lanes(g)) or conv_pending(g, k_now)
                if radio_busy:
                    continue
                now = now_ns()
                if ai_policy == "idle":
                    budget = idle_budget
                else:
                    window_end = release + period - guard
                    if use_nrx:
                        for k in open_periods:
                            if now > epoch + k * period + slack:
                                continue
                            for cell in weak:
                                if partner[cell] == g and mask[cell, k] \
                                        and job_state.get((cell, k), WAITING) in (WAITING, ELIGIBLE) \
                                        and int(state.cells[cell, k, C_CONV_STATUS]) != PASS:
                                    window_end = min(window_end, epoch + k * period + slack)
                    budget = window_end - now - guard
                if budget >= min_budget:
                    grant(g, budget, now)
        # --- global AI dispatch -------------------------------------------------
        if dispatch:
            while next_request < n_requests and epoch + int(state.requests[next_request, R_ARRIVAL]) <= loop_start:
                length = int(state.requests[next_request, R_LENGTH])
                best_g, best_finish = 0, None
                for g in range(gpus):
                    box = state.gpus[g]
                    rate = float(box[A_RATE_TPS]) or prior_tps
                    pulled = min(int(box[A_PULLED]), len(assigned_cum[g]) - 1)
                    unpulled = assigned_cum[g][-1] - assigned_cum[g][pulled]
                    backlog = max(float(box[A_BACKLOG_TOKENS]), 0.0) + unpulled
                    finish = loop_start + (backlog + length) / rate * 1e9
                    if best_finish is None or finish < best_finish:
                        best_g, best_finish = g, finish
                    if pack and finish - (epoch + int(state.requests[next_request, R_ARRIVAL])) <= pack_ns:
                        best_g, best_finish = g, finish
                        break
                arrival_abs = epoch + int(state.requests[next_request, R_ARRIVAL])
                if ai_cfg.get("slo_admission") and best_finish - arrival_abs > slo_ns:
                    state.requests[next_request, R_GPU] = -2
                    counters["ai_rejected"] += 1
                else:
                    assigned_cum[best_g].append(assigned_cum[best_g][-1] + length)
                    state.requests[next_request, R_GPU] = best_g
                    counters["ai_dispatched"] += 1
                next_request += 1
        if len(nrx_delay) > 256:
            for key in [key for key in nrx_delay if key[1] < k_now - horizon]:
                nrx_delay.pop(key)
        if delay_used and len(delay_used) > 64:
            for key in [key for key in delay_used if key[1] < k_now - 1]:
                delay_used.pop(key)
        # --- retire jobs past their rescue deadline ------------------------------
        for key in [key for key in job_state if epoch + key[1] * period + rescue_deadline < loop_start]:
            if job_state[key] in (WAITING, ELIGIBLE):
                state.set_nrx(key[0], key[1], DROPPED, loop_start)
                counters["dropped"] += 1
            job_state.pop(key)
            reasons.pop(key, None)
        elapsed = (now_ns() - loop_start) / 1e3
        counters["loops"] += 1
        counters["max_loop_us"] = max(counters["max_loop_us"], elapsed)

    for lane in lanes:
        box = state.lanes[lane]
        box[L_CELL] = -1
        lane_seq[lane] += 1
        box[L_ASSIGN_SEQ] = lane_seq[lane]
    for g in range(gpus):
        box = state.gpus[g]
        box[A_BUDGET_NS] = -1
        grant_seq[g] += 1
        box[A_GRANT_SEQ] = grant_seq[g]
    gc.enable()
    args.output.write_text(json.dumps({
        "schema": "backstop-slot-controller-v3",
        "pid": os.getpid(),
        "nrx_policy": config["nrx_policy"],
        "nrx_flags": flags,
        "ai_policy": ai_policy,
        "lanes_per_gpu": per_gpu,
        "rescue_deadline_ms": rescue_deadline / ms,
        "counters": counters,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
