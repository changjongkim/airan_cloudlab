"""AI worker of the dynamic-share baseline.

Several of these run on one GPU, each started with its own MPS share (the share of a client
cannot change after it starts).  The controller names the active one (``A_DYN_ACTIVE``); only
that worker takes new requests from the global table, the others finish what they hold and
wait.  A worker takes one request at a time, so the queue stays in the table and passes to
whichever worker is active.  This models a system that changes the AI share with the radio
load at no switching cost: the standby workers are already loaded.  Everything else is ``ai_worker2.py`` under the
``static`` policy.
"""

from __future__ import annotations

import argparse
import gc
import math
import json
import os
from pathlib import Path

import numpy as np
import torch

from qwen_units import QwenUnits
from slot_state import (
    A_BACKLOG_TOKENS, A_BUDGET_NS, A_CUR_CHUNK, A_DONE_SEQ, A_GRANT_SEQ, A_HAS_WORK, A_MAX_CHUNK,
    A_DYN_ACTIVE, A_DYN_OWNER, A_DYN_PULLED, A_DYN_SCAN,
    A_NOT_AFTER, A_PIECE_END_NS, A_PULLED, A_RATE_TPS, A_RUNNING, H_ABORT, R_ARRIVAL, R_GPU,
    R_LENGTH, R_SEQ, SlotState, now_ns,
)
from ai_worker import arrivals, dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--num-cells", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--periods", type=int, required=True)
    parser.add_argument("--period-ns", type=int, required=True)
    parser.add_argument("--ready-slot", type=int, required=True)
    parser.add_argument("--share-id", type=int, required=True, help="1..K, index of this worker's share")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    ai = config["ai"]
    policy = config["ai_policy"]
    state = SlotState(args.state, args.num_cells, args.periods, args.num_gpus)
    box = state.gpus[args.gpu]
    chunks = tuple(sorted(int(c) for c in ai["chunks"]))
    small, large = chunks[0], chunks[-1]
    runner = QwenUnits(ai["model"], args.gpu, chunks=chunks, max_ctx=int(ai["max_prompt"]))
    bounds = {int(c): {k: int(float(v) * 1e6) for k, v in b.items()}
              for c, b in ai["unit_bound_ms_by_chunk"].items()}
    big_units = int(ai.get("big_chunk_units", 2))

    # Unit bounds come from isolated measurements; next to radio work a unit takes longer.
    # With ``adaptive_unit_bound`` the worker scales each chunk size's bounds by the 90th
    # percentile of (measured piece time / isolated bound) over its recent pieces.
    adaptive = bool(ai.get("adaptive_unit_bound", False))
    scale = {c: 1.0 for c in chunks}
    ratios = {c: [] for c in chunks}

    def raw_cost(c: int, name: str) -> int:
        return bounds[c]["layer"] if name.startswith("layer") else bounds[c][name]

    def cost(c: int, name: str) -> int:
        return int(raw_cost(c, name) * scale[c])

    def request_work(length: int) -> int:
        """Bound-time of a request if every chunk is the fitting one."""
        work, done = 0, 0
        while done < length:
            room = [c for c in chunks if done + c <= runner.max_ctx] or [small]
            c = next((c for c in room if c >= length - done), room[-1])
            work += bounds[c]["prep"] + runner.layers * bounds[c]["layer"]
            done += c
        return work + bounds[c]["head"]

    horizon = args.periods * args.period_ns
    dispatch = ai.get("dispatch") == "global"
    # Per-GPU mode: this GPU's own Poisson arrivals.  Global mode: the controller assigns
    # requests from one server-wide stream (and makes the admission decision).
    plan = [] if dispatch else arrivals(ai, args.gpu, horizon)
    generator = torch.Generator(device="cpu").manual_seed(int(ai["seed"]) + args.gpu)
    prompts = [torch.randint(0, 32000, (n,), generator=generator).to(runner.device) for _, n in plan]
    requests = [{"arrival_ns": a, "prompt_len": n, "done_tokens": 0, "chunk": None, "unit": 0,
                 "head": False, "first_ns": None, "done_ns": None, "chunks_used": []}
                for a, n in plan]
    pieces = []
    begin_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    slo_ns = int(float(ai.get("slo_ms", 200.0)) * 1e6)
    admission = bool(ai.get("slo_admission", False))
    force_chunk = int(ai.get("force_chunk", 0))
    # Chunk choice.  "budget" (v2): the largest chunk whose two layer units fit what is left
    # of the current piece.  "sustained": the largest chunk the controller has allowed in at
    # least ``chunk_sustain`` of the recent grants; what is left of the piece does not matter,
    # because a chunk continues over many pieces.
    sustained = ai.get("chunk_choice", "budget") == "sustained"
    sustain = float(ai.get("chunk_sustain", 0.7))
    allowed_share = {c: 0.5 for c in chunks}
    share_stamp = [now_ns()]
    tau = 500e6
    # Admission works in prompt tokens, which do not depend on the chunk sizes
    # a request ends up using: backlog = tokens still to prefill in the queue,
    # rate = tokens finished per second while the queue was not empty.
    state_vars = {"decided": 0, "backlog": 0, "work_done": 0.0, "busy_time": 0.0,
                  "last": None, "head": 0}
    prior_rate = float(ai.get("prior_tokens_per_s", 15000.0)) / 1e9   # tokens per ns

    gc.collect()
    gc.disable()
    state.mark_ready(args.ready_slot)
    epoch = state.wait_epoch()
    end_time = epoch + horizon

    def pull() -> bool:
        """Take the requests the controller has assigned to this GPU, in arrival order, while
        this worker is the active one.  The scan position is shared by the GPU's workers; the
        owner word hands it over so that no request is taken twice."""
        active = int(box[A_DYN_ACTIVE]) == args.share_id
        owner = int(box[A_DYN_OWNER])
        if not active:
            if owner == args.share_id:
                box[A_DYN_OWNER] = 0
            return False
        if owner != args.share_id:
            if owner != 0:
                return True           # the previous worker has not let go yet
            box[A_DYN_OWNER] = args.share_id
        # Take one request at a time: what has not been started stays in the table, so a
        # worker that becomes active inherits the queue and the previous one only finishes
        # the request it is in the middle of.
        if any(r["done_ns"] is None and not r.get("rejected") for r in requests[state_vars["head"]:]):
            return True
        table = state.requests
        index, pulled = int(box[A_DYN_SCAN]), int(box[A_DYN_PULLED])
        while index < len(table) and int(table[index, R_SEQ]) == 1:
            target = int(table[index, R_GPU])
            if target == -1:
                break
            if target == args.gpu:
                length = int(table[index, R_LENGTH])
                g = torch.Generator(device="cpu").manual_seed(int(ai["seed"]) * 7919 + index)
                prompts.append(torch.randint(0, 32000, (length,), generator=g).to(runner.device))
                requests.append({"arrival_ns": int(table[index, R_ARRIVAL]), "prompt_len": length,
                                 "done_tokens": 0, "chunk": None, "unit": 0, "head": False,
                                 "first_ns": None, "done_ns": None, "chunks_used": []})
                pulled += 1
                index += 1
                break
            index += 1
        box[A_DYN_SCAN] = index
        box[A_DYN_PULLED] = pulled
        box[A_PULLED] = pulled
        return True

    def decide(now: int) -> None:
        while state_vars["decided"] < len(requests) and \
                epoch + requests[state_vars["decided"]]["arrival_ns"] <= now:
            r = requests[state_vars["decided"]]
            rate = (state_vars["work_done"] / state_vars["busy_time"]
                    if state_vars["busy_time"] > 100e6 else prior_rate)
            finish = now + (state_vars["backlog"] + r["prompt_len"]) / max(rate, 1e-12)
            if admission and not dispatch and finish - (epoch + r["arrival_ns"]) > slo_ns:
                r["rejected"] = True
            else:
                state_vars["backlog"] += r["prompt_len"]
            state_vars["decided"] += 1

    max_chunk = 0

    def next_unit(r: dict, budget_left: int | None):
        """(chunk, unit name, start token) for the request's next unit."""
        if r["head"]:
            return r["chunk_last"], "head", r["done_tokens_last"]
        if r["chunk"] is None:
            remaining = r["prompt_len"] - r["done_tokens"]
            # Chunks must stay inside the static context, and the smallest
            # chunk that covers the rest of the prompt in one go is preferred.
            room = [c for c in chunks if r["done_tokens"] + c <= runner.max_ctx] or [small]
            if force_chunk:
                room = [c for c in room if c <= force_chunk] or [small]
            if max_chunk:
                room = [c for c in room if c <= max_chunk] or [small]
            fit = next((c for c in room if c >= remaining), room[-1])
            if policy == "static" or budget_left is None:
                c = fit
            elif sustained:
                steady = [c for c in room if c <= fit and allowed_share[c] >= sustain]
                c = steady[-1] if steady else room[0]
            else:
                # Under a grant, the largest chunk up to ``fit`` whose layer
                # units still fit ``big_chunk_units`` times into this budget.
                usable = [c for c in room if c <= fit and budget_left >= big_units * cost(c, "layer0")]
                c = usable[-1] if usable else small
            r["chunk"], r["unit"] = c, 0
            r["chunks_used"].append(c)
        name = "prep" if r["unit"] == 0 else f"layer{r['unit'] - 1}"
        return r["chunk"], name, r["done_tokens"]

    def chunk_allowed(r: dict) -> bool:
        """A chunk already in progress may continue only if the grant allows its size."""
        c = r["chunk"] if r["chunk"] is not None else (r.get("chunk_last") if r["head"] else None)
        return c is None or not max_chunk or c <= max_chunk

    def advance(r: dict, name: str) -> bool:
        """Mark one unit done; returns True when the request is finished."""
        if name == "head":
            return True
        r["unit"] += 1
        if r["unit"] > runner.layers:
            r["chunk_last"], r["done_tokens_last"] = r["chunk"], r["done_tokens"]
            progress = min(r["chunk"], r["prompt_len"] - r["done_tokens"])
            state_vars["backlog"] = max(0, state_vars["backlog"] - progress)
            state_vars["work_done"] += progress
            r["done_tokens"] += r["chunk"]
            r["chunk"], r["unit"] = None, 0
            if r["done_tokens"] >= r["prompt_len"]:
                r["head"] = True
        return False

    def run_piece(budget: int | None, max_units: int) -> None:
        start = now_ns()
        launched, bound_sum = 0, 0
        raw_sum, used = 0, set()
        index = state_vars["head"]
        finished = []
        begin_event.record(runner.stream)
        while index < len(requests) and launched < max_units:
            r = requests[index]
            if r.get("rejected") or r["done_ns"] is not None:
                index += 1
                continue
            if epoch + r["arrival_ns"] > start:
                break
            if not chunk_allowed(r):
                break
            left = None if budget is None else budget - bound_sum
            c, name, token = next_unit(r, left)
            unit_cost = cost(c, name)
            if budget is not None and bound_sum + unit_cost > budget:
                if r["chunk"] is not None and r["unit"] == 0 and name == "prep" and (c == large or sustained):
                    r["chunk"] = None                   # re-choose next time
                    r["chunks_used"].pop()
                break
            if r["first_ns"] is None:
                r["first_ns"] = start
                runner.load_prompt(prompts[index])
            runner.launch(c, name, token)
            bound_sum += unit_cost
            raw_sum += raw_cost(c, name)
            used.add(c)
            launched += 1
            if advance(r, name):
                finished.append(r)
                index += 1
        end_event.record(runner.stream)
        if launched:
            box[A_PIECE_END_NS] = start + bound_sum
            box[A_RUNNING] = 1
            while not end_event.query():
                pass
            finish = now_ns()
            box[A_RUNNING] = 0
            for r in finished:
                r["done_ns"] = finish
            pieces.append([start, finish, launched, bound_sum,
                           float(begin_event.elapsed_time(end_event))])
            if adaptive and raw_sum:
                for c in used:
                    window = ratios[c]
                    window.append((finish - start) / raw_sum)
                    if len(window) > 64:
                        del window[0]
                    if len(window) >= 8:
                        scale[c] = min(3.0, max(1.0, float(np.percentile(window, 90))))
        head = state_vars["head"]
        while head < len(requests) and (requests[head].get("rejected") or
                                        requests[head]["done_ns"] is not None):
            head += 1
        state_vars["head"] = head

    def has_work(now: int) -> bool:
        i = state_vars["head"]
        while i < len(requests) and epoch + requests[i]["arrival_ns"] <= now:
            if not requests[i].get("rejected") and requests[i]["done_ns"] is None:
                return True
            i += 1
        return False

    done_seq = 0
    while True:
        now = now_ns()
        if policy == "static":
            max_chunk = 0
        if now > end_time + 50_000_000 or state.header[H_ABORT]:
            break
        active = pull()
        decide(now)
        if state_vars["last"] is not None:
            dt = now - state_vars["last"]
            decay = float(np.exp(-dt / tau))
            state_vars["work_done"] *= decay
            state_vars["busy_time"] = state_vars["busy_time"] * decay + (dt if state_vars["backlog"] > 0 else 0)
        state_vars["last"] = now
        rate_now = (state_vars["work_done"] / state_vars["busy_time"]
                    if state_vars["busy_time"] > 100e6 else prior_rate)
        working = has_work(now)
        if active:
            box[A_RATE_TPS] = int(rate_now * 1e9)
            box[A_BACKLOG_TOKENS] = int(state_vars["backlog"])
            box[A_HAS_WORK] = 1 if working else 0
        if now > end_time:
            break
        if working:
            run_piece(None, int(ai.get("static_units_per_piece", 8)))
        continue
        seq = int(box[A_GRANT_SEQ])
        if seq == done_seq:
            continue
        budget = int(box[A_BUDGET_NS])
        if budget < 0:
            break
        max_chunk = int(box[A_MAX_CHUNK])
        # Share of recent time (not of grants: unused grants repeat every few microseconds)
        # in which each chunk size was allowed; time constant 20 ms.
        stamp = now_ns()
        weight = 1.0 - math.exp(-(stamp - share_stamp[0]) / 20e6)
        share_stamp[0] = stamp
        for c in chunks:
            allowed_share[c] += weight * ((1.0 if not max_chunk or c <= max_chunk else 0.0) - allowed_share[c])
        if now_ns() <= int(box[A_NOT_AFTER]):
            run_piece(budget, 10_000)
        done_seq = seq
        box[A_DONE_SEQ] = seq
    gc.enable()

    finished = [r for r in requests if r["done_ns"] is not None]
    ttft = [(r["done_ns"] - (epoch + r["arrival_ns"])) / 1e6 for r in finished]
    slo = float(ai.get("slo_ms", 200.0))
    measured_s = horizon / 1e9
    chunk_counts = {}
    for r in finished:
        for c in r["chunks_used"]:
            chunk_counts[str(c)] = chunk_counts.get(str(c), 0) + 1
    summary = {
        "arrived": sum(1 for r in requests if r["arrival_ns"] <= horizon),
        "rejected": sum(1 for r in requests if r.get("rejected")),
        "completed": len(finished),
        "prompt_tokens_completed": int(sum(r["prompt_len"] for r in finished)),
        "tokens_per_s": sum(r["prompt_len"] for r in finished) / measured_s,
        "slo_ms": slo,
        "within_slo": sum(t <= slo for t in ttft),
        "tokens_within_slo_per_s": sum(r["prompt_len"] for r, t in zip(finished, ttft) if t <= slo) / measured_s,
        "ttft_ms": dist(ttft),
        "pieces": len(pieces),
        "units": int(sum(p[2] for p in pieces)),
        "piece_wall_ms": dist([(p[1] - p[0]) / 1e6 for p in pieces]),
        "piece_overruns": sum(1 for p in pieces if (p[1] - p[0]) > p[3] + int(ai.get("overrun_tol_us", 100)) * 1000),
        "gpu_busy_ms": float(sum(p[4] for p in pieces)),
        "chunks_used": chunk_counts,
        "unit_bound_scale": {str(c): scale[c] for c in chunks},
    }
    summary["headline"] = {k: summary[k] for k in (
        "arrived", "completed", "tokens_per_s", "within_slo", "tokens_within_slo_per_s", "piece_overruns")}
    summary["headline"]["ttft_p50_ms"] = summary["ttft_ms"].get("p50")
    summary["headline"]["ttft_p99_ms"] = summary["ttft_ms"].get("p99")
    args.output.write_text(json.dumps({
        "schema": "backstop-slot-ai-worker-v2", "gpu": args.gpu, "pid": os.getpid(),
        "policy": policy, "epoch_ns": epoch, "summary": summary,
        "pieces_columns": ["start_ns", "end_ns", "units", "bound_ns", "gpu_ms"], "pieces": pieces,
        "requests": [[r["arrival_ns"], r["prompt_len"], r["first_ns"], r["done_ns"]] for r in requests],
    }), encoding="utf-8")


if __name__ == "__main__":
    main()
