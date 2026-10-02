#!/usr/bin/env python3
"""Per-GPU Qwen prefill worker that runs units only inside granted budgets.

Requests arrive as a Poisson process with BurstGPT prompt lengths and are
served first come, first served.  Under a grant, the worker launches the next
units of the queue whose validated bounds fit the budget; in ``static`` mode
it runs units back to back in its fixed MPS share without grants.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

from qwen_pieces import QwenPieces
from slot_state import (
    A_BUDGET_NS, A_DONE_SEQ, A_GRANT_SEQ, A_HAS_WORK, A_NOT_AFTER, A_PIECE_END_NS,
    A_RUNNING, H_ABORT, SlotState, now_ns,
)


def dist(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}


from ai_arrivals import arrivals  # noqa: E402  (re-exported for ai_worker2)


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    ai = config["ai"]
    policy = config["ai_policy"]
    state = SlotState(args.state, args.num_cells, args.periods, args.num_gpus)
    box = state.gpus[args.gpu]
    runner = QwenPieces(ai["model"], args.gpu, chunk=int(ai["chunk"]),
                        max_ctx=int(ai["max_prompt"]))
    bounds = {k: int(float(v) * 1e6) for k, v in ai["unit_bound_ms"].items()}

    def bound_of(name: str) -> int:
        return bounds["layer"] if name.startswith("layer") else bounds[name]

    horizon = args.periods * args.period_ns
    plan = arrivals(ai, args.gpu, horizon)
    generator = torch.Generator(device="cpu").manual_seed(int(ai["seed"]) + args.gpu)
    prompts = [torch.randint(0, 32000, (length,), generator=generator).to(runner.device)
               for _, length in plan]
    # warm the graphs on this GPU before readiness
    runner.load_prompt(prompts[0] if prompts else torch.zeros(8, dtype=torch.long,
                                                               device=runner.device))
    for name, chunk in runner.units_for(int(prompts[0].numel()) if prompts else 8):
        runner.launch(name, chunk)
    runner.stream.synchronize()

    requests = [{"arrival_ns": a, "prompt_len": n, "units": None, "next": 0,
                 "first_ns": None, "done_ns": None} for a, n in plan]
    pieces = []
    begin_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    gc.collect()
    gc.disable()
    state.mark_ready(args.ready_slot)
    epoch = state.wait_epoch()
    end_time = epoch + horizon
    head = 0          # first unfinished request
    done_seq = 0
    # SLO admission: a request is admitted at arrival only if the backlog ahead
    # of it, served at the recently observed unit rate, finishes within the SLO.
    slo_ns = int(float(ai.get("slo_ms", 200.0)) * 1e6)
    admission = bool(ai.get("slo_admission", False))
    prior_rate = float(ai.get("prior_units_per_ms", 1.5)) / 1e6       # units per ns
    tau = 500e6
    decided = 0
    backlog_units = 0
    rate_units = 0.0
    rate_time = 0.0
    last_loop = None

    def admitted(now: int) -> int:
        """Index one past the last request that has arrived by ``now``."""
        index = head
        while index < len(requests) and epoch + requests[index]["arrival_ns"] <= now:
            index += 1
        return index

    def run_piece(budget: int | None, max_units: int) -> tuple[int, int]:
        nonlocal head, backlog_units, rate_units
        start = now_ns()
        launched, bound_sum = 0, 0
        arrived = admitted(start)
        index = head
        begin_event.record(runner.stream)
        while index < arrived and launched < max_units:
            request = requests[index]
            if request.get("rejected"):
                index += 1
                continue
            if request["units"] is None:
                request["units"] = runner.units_for(request["prompt_len"])
            name, chunk = request["units"][request["next"]]
            cost = bound_of(name)
            if budget is not None and bound_sum + cost > budget:
                break
            if request["next"] == 0:
                runner.load_prompt(prompts[index])
                request["first_ns"] = start
            runner.launch(name, chunk)
            bound_sum += cost
            launched += 1
            request["next"] += 1
            if request["next"] == len(request["units"]):
                request["done_ns"] = -1          # set after the piece completes
                index += 1
        end_event.record(runner.stream)
        if launched:
            box[A_PIECE_END_NS] = start + bound_sum
            box[A_RUNNING] = 1
            while not end_event.query():
                pass
            finish = now_ns()
            box[A_RUNNING] = 0
            for request in requests[head:index]:
                if request["done_ns"] == -1:
                    request["done_ns"] = finish
            head = index
            pieces.append([start, finish, launched, bound_sum,
                           float(begin_event.elapsed_time(end_event))])
        backlog_units -= launched
        rate_units += launched
        return launched, bound_sum

    def decide(now: int) -> None:
        nonlocal decided, backlog_units
        while decided < len(requests) and epoch + requests[decided]["arrival_ns"] <= now:
            request = requests[decided]
            request["units"] = runner.units_for(request["prompt_len"])
            need = len(request["units"])
            rate = rate_units / rate_time if rate_time > 50e6 else prior_rate
            finish = now + (backlog_units + need) / max(rate, 1e-9)
            if admission and finish - (epoch + request["arrival_ns"]) > slo_ns:
                request["rejected"] = True
            else:
                backlog_units += need
            decided += 1

    while True:
        now = now_ns()
        if now > end_time + 50_000_000 or state.header[H_ABORT]:
            break
        decide(now)
        while head < len(requests) and requests[head].get("rejected"):
            head += 1
        if last_loop is not None:
            dt = now - last_loop
            decay = float(np.exp(-dt / tau))
            rate_units *= decay
            rate_time = rate_time * decay + (dt if backlog_units > 0 else 0)
        last_loop = now
        box[A_HAS_WORK] = 1 if admitted(now) > head else 0
        if policy == "static":
            if now > end_time:
                break
            if box[A_HAS_WORK]:
                run_piece(None, int(ai.get("static_units_per_piece", 8)))
            continue
        seq = int(box[A_GRANT_SEQ])
        if seq == done_seq:
            continue
        budget = int(box[A_BUDGET_NS])
        if budget < 0:
            break
        if now_ns() <= int(box[A_NOT_AFTER]):
            run_piece(budget, 10_000)
        done_seq = seq
        box[A_DONE_SEQ] = seq
    gc.enable()

    finished = [r for r in requests if r["done_ns"] not in (None, -1)]
    ttft = [(r["done_ns"] - (epoch + r["arrival_ns"])) / 1e6 for r in finished]
    slo = float(ai.get("slo_ms", 200.0))
    measured_s = horizon / 1e9
    overruns = sum(1 for p in pieces if (p[1] - p[0]) > p[3] + int(ai.get("overrun_tol_us", 100)) * 1000)
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
        "piece_overruns": overruns,
        "gpu_busy_ms": float(sum(p[4] for p in pieces)),
    }
    summary["headline"] = {k: summary[k] for k in (
        "arrived", "completed", "tokens_per_s", "within_slo", "tokens_within_slo_per_s",
        "piece_overruns")}
    summary["headline"]["ttft_p50_ms"] = summary["ttft_ms"].get("p50")
    summary["headline"]["ttft_p99_ms"] = summary["ttft_ms"].get("p99")
    args.output.write_text(json.dumps({
        "schema": "backstop-slot-ai-worker-v1",
        "gpu": args.gpu, "pid": os.getpid(), "policy": policy,
        "epoch_ns": epoch,
        "summary": summary,
        "pieces_columns": ["start_ns", "end_ns", "units", "bound_ns", "gpu_ms"],
        "pieces": pieces,
        "requests": [[r["arrival_ns"], r["prompt_len"], r["first_ns"], r["done_ns"]]
                     for r in requests],
    }), encoding="utf-8")


if __name__ == "__main__":
    main()
