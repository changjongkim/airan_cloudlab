#!/usr/bin/env python3
"""Per-GPU AI worker with several job classes (v3).

Each class has its own model context (prompt buffer, KV cache, CUDA-graph units) and its own
first-come-first-served queue, so the worker can switch class between any two units:

  interactive  Poisson arrivals with a completion SLO (time to first token of a prefill);
               optional admission: a request is taken only if it is predicted to meet the SLO
  batch        an always-full queue of long prompts without an SLO (throughput filler)

Within a granted piece the worker repeatedly takes the next unit of the most urgent class
whose unit fits: interactive classes by earliest SLO deadline, batch last.  A class whose
chunk in progress is larger than the grant allows, or whose next unit does not fit the rest
of the piece, is skipped in favour of the next class.  ``class_order: fifo`` disables this
choice (classes are served strictly in arrival order) for the ablation.
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

import qwen_units
from ai_arrivals import arrivals
from qwen_units import QwenUnits
from slot_state import (
    A_BUDGET_NS, A_CUR_CHUNK, A_DONE_SEQ, A_GRANT_SEQ, A_HAS_WORK, A_MAX_CHUNK, A_NOT_AFTER,
    A_PIECE_END_NS, A_RUNNING, H_ABORT, SlotState, now_ns,
)


def dist(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}


class _SharedWeights:
    """Load each model once per process; contexts of the same model share its weights."""

    cache: dict = {}
    real = qwen_units.AutoModelForCausalLM

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        if name not in cls.cache:
            cls.cache[name] = cls.real.from_pretrained(name, **kwargs)
        return cls.cache[name]


qwen_units.AutoModelForCausalLM = _SharedWeights


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
    chunks = tuple(sorted(int(c) for c in ai["chunks"]))
    small, large = chunks[0], chunks[-1]
    horizon = args.periods * args.period_ns
    adaptive = bool(ai.get("adaptive_unit_bound", False))
    by_urgency = ai.get("class_order", "urgency") == "urgency"
    force_chunk = int(ai.get("force_chunk", 0))
    big_units = int(ai.get("big_chunk_units", 2))
    sustained = ai.get("chunk_choice", "budget") == "sustained"
    sustain = float(ai.get("chunk_sustain", 0.7))
    allowed_share = {c: 0.5 for c in chunks}
    share_stamp = [now_ns()]

    classes = []
    for index, spec in enumerate(ai["classes"]):
        runner = QwenUnits(spec["model"], args.gpu, chunks=chunks, max_ctx=int(spec.get("max_prompt", 1024)))
        bounds = {int(c): {k: int(float(v) * 1e6) for k, v in b.items()}
                  for c, b in ai["unit_bound_ms_by_model"][spec["model"]].items()}
        kind = spec.get("kind", "interactive")
        if kind == "interactive":
            cfg = dict(ai, rate_per_s=float(spec["rate_per_s"]), seed=int(ai["seed"]) + 7 * index,
                       max_prompt=int(spec.get("max_prompt", 1024)))
            plan = arrivals(cfg, args.gpu, horizon)
        else:
            plan = []
        generator = torch.Generator(device="cpu").manual_seed(int(ai["seed"]) + args.gpu + 131 * index)
        classes.append({
            "name": spec["name"], "kind": kind, "runner": runner, "bounds": bounds,
            "slo_ns": int(float(spec.get("slo_ms", 0)) * 1e6),
            "admission": bool(spec.get("admission", ai.get("slo_admission", False))),
            "batch_len": int(spec.get("prompt_len", 1024)),
            "generator": generator,
            "requests": [{"arrival_ns": a, "prompt_len": n, "done_tokens": 0, "chunk": None, "unit": 0,
                          "head": False, "first_ns": None, "done_ns": None} for a, n in plan],
            "prompts": [torch.randint(0, 32000, (n,), generator=generator).to(runner.device) for _, n in plan],
            "head": 0, "decided": 0,
            "scale": {c: 1.0 for c in chunks}, "ratios": {c: [] for c in chunks},
        })
    # One stream for every context: units of different classes run one after another.
    stream = classes[0]["runner"].stream
    for cls in classes:
        cls["runner"].stream = stream

    def raw_cost(cls: dict, c: int, name: str) -> int:
        b = cls["bounds"][c]
        return b["layer"] if name.startswith("layer") else b[name]

    def cost(cls: dict, c: int, name: str) -> int:
        return int(raw_cost(cls, c, name) * cls["scale"][c])

    def request_work(cls: dict, length: int) -> int:
        """Unit-bound time of a request if every chunk is the fitting one.

        Tried and dropped (job 59155110): prompt tokens (short prompts cost far more per
        token, so long requests were over-rejected) and the chunk sizes the worker would
        pick right now (over-rejects under a granted policy).  This estimate under-counts
        when a granted policy falls back to small chunks, so it admits too much in overload.
        """
        runner, work, done, c = cls["runner"], 0, 0, small
        while done < length:
            room = [x for x in chunks if done + x <= runner.max_ctx] or [small]
            c = next((x for x in room if x >= length - done), room[-1])
            work += cls["bounds"][c]["prep"] + runner.layers * cls["bounds"][c]["layer"]
            done += c
        return work + cls["bounds"][c]["head"]

    # Admission for interactive classes, in unit-bound time: the interactive backlog (unit
    # bounds of the admitted requests, with the chunk sizes the worker would pick) divided by
    # the recent rate at which interactive unit time was served while a backlog existed.
    # Prompt tokens are not used here: short prompts cost far more per token than long ones.
    inter = {"backlog": 0.0, "work_done": 0.0, "busy_time": 0.0, "last": None}
    prior_rate = float(ai.get("prior_work_rate", 0.5))
    tau = 500e6
    pieces = []
    begin_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    gc.collect()
    gc.disable()
    state.mark_ready(args.ready_slot)
    epoch = state.wait_epoch()
    end_time = epoch + horizon
    max_chunk = 0

    def decide(now: int) -> None:
        for cls in classes:
            if cls["kind"] != "interactive":
                continue
            reqs = cls["requests"]
            while cls["decided"] < len(reqs) and epoch + reqs[cls["decided"]]["arrival_ns"] <= now:
                r = reqs[cls["decided"]]
                r["work"] = request_work(cls, r["prompt_len"])
                rate = inter["work_done"] / inter["busy_time"] if inter["busy_time"] > 100e6 else prior_rate
                finish = now + (inter["backlog"] + r["work"]) / max(rate, 1e-9)
                r["dbg"] = [round(inter["backlog"] / 1e6, 2), round(rate, 3), round((finish - (epoch + r["arrival_ns"])) / 1e6, 1)]
                if cls["admission"] and finish - (epoch + r["arrival_ns"]) > cls["slo_ns"]:
                    r["rejected"] = True
                else:
                    inter["backlog"] += r["work"]
                cls["decided"] += 1

    def head_request(cls: dict, now: int):
        """The class's request to serve now (FCFS), or None."""
        reqs = cls["requests"]
        h = cls["head"]
        while h < len(reqs) and (reqs[h].get("rejected") or reqs[h]["done_ns"] is not None):
            h += 1
        cls["head"] = h
        if cls["kind"] == "batch" and h >= len(reqs) and now < end_time:
            n = cls["batch_len"]
            reqs.append({"arrival_ns": now - epoch, "prompt_len": n, "done_tokens": 0, "chunk": None, "unit": 0,
                         "head": False, "first_ns": None, "done_ns": None})
            cls["prompts"].append(torch.randint(0, 32000, (n,), generator=cls["generator"]).to(cls["runner"].device))
        if h < len(reqs) and epoch + reqs[h]["arrival_ns"] <= now:
            return reqs[h]
        return None

    def next_unit(cls: dict, r: dict, budget_left):
        """(chunk, unit name, start token) of the request's next unit, or None if not allowed."""
        runner = cls["runner"]
        if r["head"]:
            c = r["chunk_last"]
            return (c, "head", r["done_tokens_last"]) if not max_chunk or c <= max_chunk else None
        if r["chunk"] is None:
            remaining = r["prompt_len"] - r["done_tokens"]
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
                usable = [c for c in room if c <= fit and budget_left >= big_units * cost(cls, c, "layer0")]
                c = usable[-1] if usable else small
            return (c, "prep", r["done_tokens"])
        if max_chunk and r["chunk"] > max_chunk:
            return None
        return (r["chunk"], f"layer{r['unit'] - 1}", r["done_tokens"])

    def advance(cls: dict, r: dict, c: int, name: str) -> bool:
        if cls["kind"] == "interactive":
            done = raw_cost(cls, c, name)
            inter["backlog"] = max(0.0, inter["backlog"] - done)
            inter["work_done"] += done
        if name == "head":
            return True
        if name == "prep":
            r["chunk"], r["unit"] = c, 1
            r.setdefault("chunks_used", []).append(c)
            return False
        r["unit"] += 1
        if r["unit"] > cls["runner"].layers:
            r["chunk_last"], r["done_tokens_last"] = r["chunk"], r["done_tokens"]
            r["done_tokens"] += r["chunk"]
            r["chunk"], r["unit"] = None, 0
            if r["done_tokens"] >= r["prompt_len"]:
                r["head"] = True
        return False

    def ordered(now: int) -> list:
        live = []
        for index, cls in enumerate(classes):
            r = head_request(cls, now)
            if r is None:
                continue
            if cls["kind"] == "batch":
                key = (1, 0, index) if by_urgency else (0, epoch + r["arrival_ns"], index)
            else:
                key = (0, epoch + r["arrival_ns"] + (cls["slo_ns"] if by_urgency else 0), index)
            live.append((key, cls, r))
        live.sort(key=lambda item: item[0])
        return live

    def run_piece(budget, max_units: int) -> None:
        start = now_ns()
        launched, bound_sum, raw_sum = 0, 0, 0
        used, finished = set(), []
        begin_event.record(stream)
        while launched < max_units:
            picked = None
            for _, cls, r in ordered(start):
                left = None if budget is None else budget - bound_sum
                unit = next_unit(cls, r, left)
                if unit is None:
                    if by_urgency:
                        continue
                    break
                c, name, token = unit
                unit_cost = cost(cls, c, name)
                if budget is not None and bound_sum + unit_cost > budget:
                    if by_urgency:
                        continue
                    break
                picked = (cls, r, c, name, token, unit_cost)
                break
            if picked is None:
                break
            cls, r, c, name, token, unit_cost = picked
            runner = cls["runner"]
            if r["first_ns"] is None:
                r["first_ns"] = start
                runner.load_prompt(cls["prompts"][cls["head"]])
            runner.launch(c, name, token)
            bound_sum += unit_cost
            raw_sum += raw_cost(cls, c, name)
            used.add((id(cls), c))
            launched += 1
            if advance(cls, r, c, name):
                r["done_ns"] = -1                 # finished; time stamped below
                finished.append(r)
        end_event.record(stream)
        if launched:
            box[A_PIECE_END_NS] = start + bound_sum
            box[A_RUNNING] = 1
            while not end_event.query():
                pass
            finish = now_ns()
            box[A_RUNNING] = 0
            for r in finished:
                r["done_ns"] = finish
            pieces.append([start, finish, launched, bound_sum, float(begin_event.elapsed_time(end_event))])
            if adaptive and raw_sum:
                for cls in classes:
                    for c in chunks:
                        if (id(cls), c) not in used:
                            continue
                        window = cls["ratios"][c]
                        window.append((finish - start) / raw_sum)
                        if len(window) > 64:
                            del window[0]
                        if len(window) >= 8:
                            cls["scale"][c] = min(3.0, max(1.0, float(np.percentile(window, 90))))

    done_seq = 0
    while True:
        now = now_ns()
        if now > end_time + 50_000_000 or state.header[H_ABORT]:
            break
        decide(now)
        if inter["last"] is not None:
            dt = now - inter["last"]
            decay = float(np.exp(-dt / tau))
            inter["work_done"] *= decay
            inter["busy_time"] = inter["busy_time"] * decay + (dt if inter["backlog"] > 0 else 0)
        inter["last"] = now
        heads = [head_request(cls, now) for cls in classes] if now < end_time else []
        has_work = any(r is not None for r in heads)
        box[A_HAS_WORK] = 1 if has_work else 0
        box[A_CUR_CHUNK] = max([int(r["chunk"] or (r.get("chunk_last", 0) if r["head"] else 0))
                                for r in heads if r is not None], default=0)
        if policy == "static":
            max_chunk = 0
            if now > end_time:
                break
            if has_work:
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

    measured_s = horizon / 1e9
    per_class = {}
    for cls in classes:
        done = [r for r in cls["requests"] if r["done_ns"] is not None and r["done_ns"] > 0
                and r["done_ns"] <= end_time + 50_000_000]
        latency = [(r["done_ns"] - (epoch + r["arrival_ns"])) / 1e6 for r in done]
        slo = cls["slo_ns"] / 1e6
        tokens = sum(r["prompt_len"] for r in done)
        entry = {
            "kind": cls["kind"], "model": str(cls["runner"].hf.config._name_or_path),
            "arrived": sum(1 for r in cls["requests"] if r["arrival_ns"] <= horizon) if cls["kind"] != "batch" else len(done),
            "rejected": sum(1 for r in cls["requests"] if r.get("rejected")),
            "completed": len(done), "tokens_per_s": tokens / measured_s,
            "partial_tokens_per_s": sum(r["done_tokens"] for r in cls["requests"] if r["done_ns"] is None) / measured_s,
            "latency_ms": dist(latency), "unit_bound_scale": {str(c): cls["scale"][c] for c in chunks},
            "admission_debug": [[r["prompt_len"], bool(r.get("rejected"))] + r.get("dbg", []) for r in cls["requests"][:40]],
        }
        if cls["kind"] == "interactive":
            entry["slo_ms"] = slo
            entry["within_slo"] = int(sum(t <= slo for t in latency))
            entry["tokens_within_slo_per_s"] = sum(r["prompt_len"] for r, t in zip(done, latency) if t <= slo) / measured_s
        per_class[cls["name"]] = entry
    inter_classes = [v for v in per_class.values() if v["kind"] == "interactive"]
    summary = {
        "arrived": sum(v["arrived"] for v in inter_classes),
        "rejected": sum(v["rejected"] for v in inter_classes),
        "completed": sum(v["completed"] for v in inter_classes),
        "tokens_per_s": sum(v["tokens_per_s"] for v in per_class.values()),
        "within_slo": sum(v["within_slo"] for v in inter_classes),
        "tokens_within_slo_per_s": sum(v["tokens_within_slo_per_s"] for v in inter_classes),
        "batch_tokens_per_s": sum(v["tokens_per_s"] + v["partial_tokens_per_s"]
                                  for v in per_class.values() if v["kind"] == "batch"),
        "slo_ms": max([v["slo_ms"] for v in inter_classes], default=0.0),
        "ttft_ms": dist([]),
        "pieces": len(pieces), "units": int(sum(p[2] for p in pieces)),
        "piece_wall_ms": dist([(p[1] - p[0]) / 1e6 for p in pieces]),
        "piece_overruns": sum(1 for p in pieces if (p[1] - p[0]) > p[3] + int(ai.get("overrun_tol_us", 100)) * 1000),
        "gpu_busy_ms": float(sum(p[4] for p in pieces)),
        "classes": per_class,
    }
    args.output.write_text(json.dumps({
        "schema": "backstop-slot-ai-worker-v3", "gpu": args.gpu, "pid": os.getpid(),
        "policy": policy, "epoch_ns": epoch, "summary": summary,
        "pieces_columns": ["start_ns", "end_ns", "units", "bound_ns", "gpu_ms"], "pieces": pieces,
    }), encoding="utf-8")


if __name__ == "__main__":
    main()
