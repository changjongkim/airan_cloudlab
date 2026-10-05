#!/usr/bin/env python3
"""Per-GPU AI worker for several models and kinds of work (v5).

Classes (``ai.classes5``), each with its own model context(s) and first-come-first-served queue:

  prefill  LLM prompt processing with a time limit to the first token (as ai_worker2/3)
  chat     prefill, then ``response`` output tokens; token i is due at
           arrival + ttft_ms + i * tbt_ms.  A class has ``sessions`` contexts (KV caches), so
           several requests decode at the same time, one unit at a time each
  encoder  encoder-only model (text embedding, image classification): items that wait are
           put into a batch of one of the configured sizes; every item has a time limit
  batch    always-full queue of long prompts without a time limit

Every piece of work is a sequence of units, each one captured CUDA graph with a time bound
measured alone (``bench_units5.py``).  A unit belongs to a size class by its bound
(``ai.size_class_ms``: 128 / 512 / 1024, the classes the controller grants), so the controller
rules that were made for prefill chunk sizes apply to every kind of unit.

Within a piece the worker takes the next unit of the job with the earliest latest-start time
(time limit of its next result minus the work that result still needs) whose unit is allowed by
the grant and fits the rest of the piece.  Encoder items wait for a batch for at most
``batch_delay_ms`` (default 0.3 of the time limit).  With ``ai.stop_check`` the worker puts
one unit group (up to ``ai.stop_group_ms`` of bound time) on the GPU at a time and reads the
controller's stop word in between.

``ai.dispatch: global``: the requests of all classes come from one server-level arrival list
(``ai_arrivals5.plan5``); the controller assigns each to a GPU and admits or rejects it.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import qwen_units
from model_units import ChatUnits, EncoderUnits
from ai_arrivals5 import plan5
from slot_state import (
    A_BACKLOG_TOKENS, A_BUDGET_NS, A_CHATS_DONE, A_CUR_CHUNK, A_DONE_SEQ, A_GRANT_SEQ, A_HAS_WORK,
    A_MAX_CHUNK, A_NOT_AFTER, A_PIECE_END_NS, A_PULLED, A_RATE_TPS, A_RUNNING, A_STOP, H_ABORT,
    R_GPU, R_SEQ, SlotState, now_ns,
)

FAR = 1 << 62


def dist(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


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
    horizon = args.periods * args.period_ns
    bounds_all = json.loads(Path(ai["bounds_file"]).read_text())
    class_ms = {int(k): float(v) for k, v in ai.get("size_class_ms", {"128": 0.45, "512": 0.90}).items()}
    sizes = (128, 512, 1024)
    sustained = ai.get("chunk_choice", "budget") == "sustained"
    sustain = float(ai.get("chunk_sustain", 0.7))
    allowed_share = {c: 0.5 for c in sizes}
    share_stamp = [now_ns()]
    stop_check = bool(ai.get("stop_check", False)) and policy != "static"
    # Units that go on the GPU between two looks at the stop word: up to this much bound time.
    stop_group_ns = int(float(ai.get("stop_group_ms", 0.9)) * 1e6)
    admit_fraction = float(ai.get("admission_fraction", 1.0))
    pairs = json.loads(Path(ai["length_pairs"]).read_text())["pairs"]
    # Global dispatch: the controller assigns the requests of one server-level arrival list.
    dispatch = ai.get("dispatch") == "global"
    plan = plan5(ai, args.num_gpus, horizon) if dispatch else []

    def size_class(bound_ms: float) -> int:
        return 128 if bound_ms <= class_ms[128] else 512 if bound_ms <= class_ms[512] else 1024

    def ns(ms: float) -> int:
        return int(ms * 1e6)

    # ---- classes ---------------------------------------------------------------------------
    classes = []
    for index, spec in enumerate(ai["classes5"]):
        kind, model = spec["kind"], spec["model"]
        units = bounds_all[model]["units_ms"]
        rng = np.random.default_rng(int(ai["seed"]) + 1000 * args.gpu + 17 * index)
        cls = {"name": spec["name"], "kind": kind, "model": model, "index": index, "rng": rng,
               "requests": [], "head": 0, "decided": 0, "active": [], "chats": 0,
               "slo_ns": ns(float(spec.get("slo_ms", spec.get("ttft_ms", 0)))),
               "tbt_ns": ns(float(spec.get("tbt_ms", 0)))}
        if kind == "encoder":
            batches = tuple(int(b) for b in spec.get("batches", (1, 4, 16)))
            cls["runner"] = EncoderUnits(model, args.gpu, batches=batches)
            cls["batches"] = batches
            cls["cost"] = {b: {u: ns(units[str(b)][u]["bound"]) for u in ("embed", "layer", "pool")} for b in batches}
            cls["generator"] = torch.Generator().manual_seed(int(ai["seed"]) + args.gpu + 131 * index)
            # Dynamic batching: items wait for a batch until the largest batch is full or the
            # oldest item has waited this long (a share of the time limit).
            cls["batch_delay_ns"] = ns(float(spec.get("batch_delay_ms", 0.3 * float(spec.get("slo_ms", 0)))))
        else:
            chunks = tuple(int(c) for c in spec.get("chunks", (128, 512, 1024)))
            sessions = int(spec.get("sessions", 1))
            groups = int(bounds_all[model].get("decode_groups", 4))
            cls["chunks"] = chunks
            cls["sessions"] = [ChatUnits(model, args.gpu, chunks=chunks, max_ctx=int(spec.get("max_ctx", 1152)),
                                         decode_groups=groups) for _ in range(sessions)]
            cls["free"] = list(range(sessions))
            cls["runner"] = cls["sessions"][0]
            cls["cost"] = {c: {u: ns(units[str(c)][u]["bound"]) for u in ("prep", "layer", "head")} for c in chunks}
            cls["decode_cost"] = [ns(units["decode"][f"dec{g}"]["bound"]) for g in range(groups)]
            cls["generator"] = torch.Generator(device="cpu").manual_seed(int(ai["seed"]) + args.gpu + 131 * index)
            cls["batch_len"] = int(spec.get("prompt_len", 1024))
            cls["response_cap"] = int(spec.get("response_max", 64))
        # Arrivals: Poisson per class and GPU.
        if kind != "batch" and not dispatch:
            t, rate = 0.0, float(spec["rate_per_s"])
            while True:
                t += rng.exponential(1.0 / rate)
                if t * 1e9 >= horizon:
                    break
                prompt, response = pairs[int(rng.integers(len(pairs)))]
                cls["requests"].append({"arrival_ns": int(t * 1e9), "prompt_len": int(min(prompt, 1024)),
                                        "response": int(min(response, cls.get("response_cap", 0))),
                                        "done_ns": None, "first_ns": None})
        classes.append(cls)
    stream = classes[0]["runner"].stream
    for cls in classes:                       # one stream: units of all classes run one after another
        if cls["kind"] == "encoder":
            cls["runner"].stream = stream
        else:
            for session in cls["sessions"]:
                session.stream = stream

    # ---- admission in unit-bound time (all classes with a time limit share the GPU) --------------
    inter = {"backlog": 0.0, "work_done": 0.0, "busy_time": 0.0, "last": None}
    prior_rate = float(ai.get("prior_work_rate", 0.5))
    tau = 500e6

    def prefill_work(cls: dict, length: int) -> int:
        runner, work, done = cls["runner"], 0, 0
        chunks = cls["chunks"]
        c = chunks[0]
        while done < length:
            c = next((x for x in chunks if x >= length - done), chunks[-1])
            work += cls["cost"][c]["prep"] + runner.layers * cls["cost"][c]["layer"]
            done += c
        return work + cls["cost"][c]["head"]

    def request_work(cls: dict, r: dict) -> tuple[int, int]:
        """(bound time until the time limit is met, bound time after it) of a request."""
        if cls["kind"] == "encoder":
            b = cls["batches"][min(1, len(cls["batches"]) - 1)]
            cost = cls["cost"][b]
            return (cost["embed"] + cls["runner"].layers * cost["layer"] + cost["pool"]) // b, 0
        first = prefill_work(cls, r["prompt_len"])
        later = r["response"] * sum(cls["decode_cost"]) if cls["kind"] == "chat" else 0
        return first, later

    token_cost = {cls["index"]: sum(cls.get("decode_cost", [])) for cls in classes}
    for cls in classes:
        cls["bound_served"] = {}
    pieces = []
    begin_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    stops = {"count": 0}
    units_by_size = {c: 0 for c in sizes}

    gc.collect()
    gc.disable()
    state.mark_ready(args.ready_slot)
    epoch = state.wait_epoch()
    end_time = epoch + horizon
    max_chunk = 0

    scan = {"index": 0, "pulled": 0}

    def pull() -> None:
        """Take the requests the controller has assigned to this GPU, in arrival order."""
        table = state.requests
        while scan["index"] < len(plan) and int(table[scan["index"], R_SEQ]) == 1:
            owner = int(table[scan["index"], R_GPU])
            if owner == -1:
                break
            if owner == args.gpu:
                p = plan[scan["index"]]
                cls = classes[p["cls"]]
                cls["requests"].append({"arrival_ns": p["arrival_ns"], "prompt_len": p["prompt_len"],
                                        "response": p["response"], "done_ns": None, "first_ns": None})
                cls["decided"] = len(cls["requests"])
                inter["backlog"] += p["work_ns"]
                if cls["kind"] == "chat":
                    cls["chats"] += 1
                scan["pulled"] += 1
            scan["index"] += 1
        box[A_PULLED] = scan["pulled"]

    def chat_done(cls: dict) -> None:
        if cls["kind"] == "chat":
            cls["chats"] = max(0, cls["chats"] - 1)
            box[A_CHATS_DONE] = int(box[A_CHATS_DONE]) + 1

    def decode_share() -> float:
        """Share of the GPU that the responses being generated need (bound time per time)."""
        need = 0.0
        for cls in classes:
            if cls["kind"] == "chat":
                need += cls["chats"] * sum(cls["decode_cost"]) / max(1, cls["tbt_ns"])
        return min(0.9, need)

    def decide(now: int) -> None:
        """Admission at arrival.  A request with a time limit is taken if the work that is due
        before it (prompt processing and encoder batches; output tokens are due later and are
        counted as a share of the GPU) is predicted to finish within the limit.  A chat request
        also needs a free session: at most ``sessions`` responses are admitted at a time."""
        for cls in classes:
            if cls["kind"] == "batch" or dispatch:
                continue
            reqs = cls["requests"]
            while cls["decided"] < len(reqs) and epoch + reqs[cls["decided"]]["arrival_ns"] <= now:
                r = reqs[cls["decided"]]
                first, later = request_work(cls, r)
                # Rate at which work with a time limit before the first result has been served
                # while such work was waiting (output tokens of running responses take their part
                # of the GPU, so the measured rate already leaves them out).
                if inter["busy_time"] > 100e6:
                    rate = inter["work_done"] / inter["busy_time"]
                else:
                    rate = prior_rate * (1.0 - decode_share())
                finish = now + (inter["backlog"] + first) / max(rate, 1e-9)
                full = cls["kind"] == "chat" and cls["chats"] >= len(cls["sessions"])
                if ai.get("slo_admission") and (full or finish - (epoch + r["arrival_ns"]) > admit_fraction * cls["slo_ns"]):
                    r["rejected"] = True
                else:
                    inter["backlog"] += first
                    if cls["kind"] == "chat":
                        cls["chats"] += 1
                cls["decided"] += 1

    def served(cls: dict, job: dict, amount: int) -> None:
        if cls["kind"] == "batch" or job.get("phase") == "decode":
            return
        inter["work_done"] += amount
        inter["backlog"] = max(0.0, inter["backlog"] - amount)

    # ---- jobs -------------------------------------------------------------------------------------
    def start_jobs(now: int) -> None:
        """Move waiting requests into free contexts."""
        if dispatch:
            pull()
        for cls in classes:
            reqs = cls["requests"]
            if cls["kind"] == "encoder":
                if cls["active"]:
                    continue
                h = cls["head"]
                while h < len(reqs) and reqs[h].get("rejected"):
                    h += 1
                cls["head"] = h
                waiting = []
                while h < len(reqs) and epoch + reqs[h]["arrival_ns"] <= now and len(waiting) < cls["batches"][-1]:
                    if not reqs[h].get("rejected"):
                        waiting.append(reqs[h])
                    h += 1
                if not waiting:
                    continue
                if len(waiting) < cls["batches"][-1] and now - (epoch + waiting[0]["arrival_ns"]) < cls["batch_delay_ns"]:
                    continue
                # The smallest batch size that takes all waiting items (unused places are padding).
                allowed = [b for b in cls["batches"]
                           if not max_chunk or size_class(cls["cost"][b]["layer"] / 1e6) <= max_chunk] or [cls["batches"][0]]
                b = next((x for x in allowed if x >= len(waiting)), allowed[-1])
                items = waiting[:b]
                cls["head"] = reqs.index(items[-1]) + 1
                cost = cls["cost"][b]
                cls["active"].append({"kind": "encoder", "batch": b, "items": items, "unit": 0, "loaded": False,
                                      "left": cost["embed"] + cls["runner"].layers * cost["layer"] + cost["pool"],
                                      "due": epoch + items[0]["arrival_ns"] + cls["slo_ns"]})
                continue
            while cls["free"]:
                h = cls["head"]
                while h < len(reqs) and (reqs[h].get("rejected") or reqs[h].get("started")):
                    h += 1
                cls["head"] = h
                if cls["kind"] == "batch" and h >= len(reqs) and now < end_time:
                    reqs.append({"arrival_ns": now - epoch, "prompt_len": cls["batch_len"], "response": 0,
                                 "done_ns": None, "first_ns": None})
                if h >= len(reqs) or epoch + reqs[h]["arrival_ns"] > now:
                    break
                r = reqs[h]
                r["started"] = True
                slot = cls["free"].pop()
                n = r["prompt_len"]
                prompt = torch.randint(0, 32000, (n,), generator=cls["generator"]).to(cls["sessions"][slot].device)
                due = FAR if cls["kind"] == "batch" else epoch + r["arrival_ns"] + cls["slo_ns"]
                cls["active"].append({"kind": "llm", "r": r, "slot": slot, "prompt": prompt, "loaded": False,
                                      "done_tokens": 0, "chunk": None, "unit": 0, "head": False, "phase": "prefill",
                                      "out": 0, "group": 0, "due": due, "token_late": 0,
                                      "left": prefill_work(cls, n)})

    def next_unit(cls: dict, job: dict, budget_left):
        """(size class, bound ns, launch function, label) of the job's next unit, or None."""
        if job["kind"] == "encoder":
            b, runner = job["batch"], cls["runner"]
            name = "embed" if job["unit"] == 0 else "pool" if job["unit"] == runner.layers + 1 else "layer"
            bound = cls["cost"][b][name]
            unit = "embed" if name == "embed" else "pool" if name == "pool" else f"layer{job['unit'] - 1}"
            return size_class(bound / 1e6), bound, (lambda: runner.launch(b, unit)), name
        runner = cls["sessions"][job["slot"]]
        if job["phase"] == "decode":
            bound = cls["decode_cost"][job["group"]]
            position = job["r"]["prompt_len"] + job["out"]
            g = job["group"]
            return size_class(bound / 1e6), bound, (lambda: runner.launch_decode(g, position)), "dec"
        chunks = cls["chunks"]
        if job["head"]:
            c = job["chunk_last"]
            bound = cls["cost"][c]["head"]
            start = job["done_last"]
            return size_class(cls["cost"][c]["layer"] / 1e6), bound, (lambda: runner.launch(c, "head", start)), "head"
        if job["chunk"] is None:
            remaining = job["r"]["prompt_len"] - job["done_tokens"]
            room = [c for c in chunks if job["done_tokens"] + c <= 1024] or [chunks[0]]
            if max_chunk:
                room = [c for c in room if size_class(cls["cost"][c]["layer"] / 1e6) <= max_chunk] or [chunks[0]]
            fit = next((c for c in room if c >= remaining), room[-1])
            if policy == "static" or budget_left is None:
                c = fit
            elif cls["kind"] == "batch":
                # No time limit: the chunk size with the most tokens per GPU time, counting only
                # the share of the grants that allow its units.
                c = max((c for c in room if c <= fit), key=lambda c: (
                    allowed_share[size_class(cls["cost"][c]["layer"] / 1e6)] * c / cls["cost"][c]["layer"]))
            elif sustained:
                steady = [c for c in room if c <= fit and allowed_share[size_class(cls["cost"][c]["layer"] / 1e6)] >= sustain]
                c = steady[-1] if steady else room[0]
            else:
                usable = [c for c in room if c <= fit and budget_left >= 2 * cls["cost"][c]["layer"]]
                c = usable[-1] if usable else chunks[0]
            job["pending_chunk"] = c
            start = job["done_tokens"]
            return size_class(cls["cost"][c]["layer"] / 1e6), cls["cost"][c]["prep"], (lambda: runner.launch(c, "prep", start)), "prep"
        c = job["chunk"]
        index = job["unit"] - 1
        start = job["done_tokens"]
        return (size_class(cls["cost"][c]["layer"] / 1e6), cls["cost"][c]["layer"],
                (lambda: runner.launch(c, f"layer{index}", start)), "layer")

    def advance(cls: dict, job: dict, label: str, bound: int, finished: list) -> None:
        served(cls, job, bound)
        job["left"] = max(0, job["left"] - bound)
        key = "decode" if job.get("phase") == "decode" else "first"
        cls["bound_served"][key] = cls["bound_served"].get(key, 0) + bound
        if job["kind"] == "encoder":
            job["unit"] += 1
            if job["unit"] > cls["runner"].layers + 1:
                job["wait"] = True                 # no further unit until this one has left the GPU
                finished.append((cls, job, "encoder"))
            return
        runner = cls["sessions"][job["slot"]]
        if label == "dec":
            job["group"] += 1
            if job["group"] == len(cls["decode_cost"]):
                job["group"] = 0
                job["wait"] = True
                finished.append((cls, job, "token"))
            return
        if label == "head":
            job["wait"] = True
            finished.append((cls, job, "first"))
            return
        if label == "prep":
            job["chunk"], job["unit"] = job["pending_chunk"], 1
            return
        job["unit"] += 1
        if job["unit"] > runner.layers:
            job["chunk_last"], job["done_last"] = job["chunk"], job["done_tokens"]
            job["done_tokens"] += job["chunk"]
            job["chunk"], job["unit"] = None, 0
            if job["done_tokens"] >= job["r"]["prompt_len"]:
                job["head"] = True

    def stamp(finished: list, when: int) -> None:
        """Record what the units that just left the GPU completed."""
        for cls, job, what in finished:
            job["wait"] = False
            if what == "encoder":
                for item in job["items"]:
                    item["done_ns"] = when
                cls["active"].remove(job)
            elif what == "first":
                r = job["r"]
                r["first_ns"] = when
                if cls["kind"] == "chat" and r["response"] > 0:
                    job["phase"], job["head"] = "decode", False
                    job["due"] = epoch + r["arrival_ns"] + cls["slo_ns"] + cls["tbt_ns"]
                    job["left"] = token_cost[cls["index"]]
                else:
                    r["done_ns"] = when
                    chat_done(cls)
                    cls["free"].append(job["slot"])
                    cls["active"].remove(job)
            else:
                r = job["r"]
                job["out"] += 1
                if when > job["due"]:
                    job["token_late"] += 1
                job["due"] += cls["tbt_ns"]
                job["left"] = token_cost[cls["index"]]
                if job["out"] >= r["response"]:
                    r["done_ns"], r["tokens_late"] = when, job["token_late"]
                    chat_done(cls)
                    cls["free"].append(job["slot"])
                    cls["active"].remove(job)

    def ordered() -> list:
        """Jobs by latest start: the time limit of the next result minus the work it still needs."""
        live = [(job["due"] - job["left"], cls["index"], id(job), cls, job) for cls in classes for job in cls["active"]
                if not job.get("wait")]
        live.sort(key=lambda item: item[:3])
        return live

    def run_piece(budget, max_units: int, stoppable: bool) -> None:
        start = now_ns()
        launched, bound_sum, gpu_ms = 0, 0, 0.0
        running = False
        while launched < max_units:
            if stoppable and int(box[A_STOP]):
                stops["count"] += 1
                break
            start_jobs(now_ns())
            group, group_bound, finished = 0, 0, []
            limit = max_units - launched
            begin_event.record(stream)
            while group < limit:
                picked = None
                for _, _, _, cls, job in ordered():
                    unit = next_unit(cls, job, None if budget is None else budget - bound_sum)
                    size, bound, launch, label = unit
                    if max_chunk and size > max_chunk:
                        continue
                    if budget is not None and bound_sum + bound > budget:
                        continue
                    picked = (cls, job, size, bound, launch, label)
                    break
                if picked is None:
                    break
                if stoppable and group and group_bound + picked[3] > stop_group_ns:
                    break                          # the stop word is read before this unit goes on the GPU
                cls, job, size, bound, launch, label = picked
                if not job["loaded"]:
                    job["loaded"] = True
                    if job["kind"] == "encoder":
                        cls["runner"].load_inputs(job["batch"], cls["generator"])
                    else:
                        cls["sessions"][job["slot"]].load_prompt(job["prompt"])
                launch()
                units_by_size[size] += 1
                bound_sum += bound
                group_bound += bound
                group += 1
                advance(cls, job, label, bound, finished)
            if not group:
                break
            end_event.record(stream)
            if not running:
                box[A_RUNNING], running = 1, True
            box[A_PIECE_END_NS] = start + bound_sum
            while not end_event.query():
                pass
            when = now_ns()
            stamp(finished, when)
            gpu_ms += float(begin_event.elapsed_time(end_event))
            launched += group
        if running:
            box[A_RUNNING] = 0
        if launched:
            pieces.append([start, now_ns(), launched, bound_sum, gpu_ms])

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
            # Time counts as busy only while a job with a time limit is open: items that wait for
            # a batch have work in the backlog but nothing to run yet.
            open_jobs = any(cls["active"] for cls in classes if cls["kind"] != "batch")
            inter["busy_time"] = inter["busy_time"] * decay + (dt if open_jobs else 0)
        inter["last"] = now
        rate_now = (inter["work_done"] / inter["busy_time"] if inter["busy_time"] > 100e6
                    else prior_rate * (1.0 - decode_share()))
        box[A_RATE_TPS] = int(rate_now * 1e9)          # ns of unit bounds served per second
        box[A_BACKLOG_TOKENS] = int(inter["backlog"])  # ns of unit bounds waiting for a first result
        if now < end_time:
            start_jobs(now)
        has_work = any(cls["active"] for cls in classes)
        box[A_HAS_WORK] = 1 if has_work else 0
        # Several jobs are open at a time: a grant for small units is still useful while a chunk of
        # large units waits, so the controller is not told about a chunk in progress.
        box[A_CUR_CHUNK] = 0
        if policy == "static":
            max_chunk = 0
            if now > end_time:
                break
            if has_work:
                run_piece(None, int(ai.get("static_units_per_piece", 8)), False)
            continue
        seq = int(box[A_GRANT_SEQ])
        if seq == done_seq:
            continue
        budget = int(box[A_BUDGET_NS])
        if budget < 0:
            break
        max_chunk = int(box[A_MAX_CHUNK])
        stamp_ns = now_ns()
        weight = 1.0 - math.exp(-(stamp_ns - share_stamp[0]) / 20e6)
        share_stamp[0] = stamp_ns
        for c in sizes:
            allowed_share[c] += weight * ((1.0 if not max_chunk or c <= max_chunk else 0.0) - allowed_share[c])
        if now_ns() <= int(box[A_NOT_AFTER]) and has_work:
            run_piece(budget, 10_000, stop_check)
        done_seq = seq
        box[A_DONE_SEQ] = seq
    gc.enable()

    measured_s = horizon / 1e9
    per_class = {}
    for cls in classes:
        reqs = [r for r in cls["requests"] if r["arrival_ns"] <= horizon]
        entry = {"kind": cls["kind"], "model": cls["model"], "arrived": len(reqs),
                 "rejected": sum(1 for r in reqs if r.get("rejected"))}
        slo = cls["slo_ns"]
        if cls["kind"] == "encoder":
            done = [r for r in reqs if r["done_ns"]]
            latency = [(r["done_ns"] - (epoch + r["arrival_ns"])) / 1e6 for r in done]
            entry.update({"completed": len(done), "latency_ms": dist(latency),
                          "items_within_limit_per_s": sum(t <= slo / 1e6 for t in latency) / measured_s})
        else:
            first = [r for r in reqs if r["first_ns"]]
            ttft = [(r["first_ns"] - (epoch + r["arrival_ns"])) / 1e6 for r in first]
            ok = [t <= slo / 1e6 for t in ttft] if slo else [True] * len(ttft)
            entry.update({"completed": len(first), "ttft_ms": dist(ttft),
                          "prompt_tokens_per_s": sum(r["prompt_len"] for r in first) / measured_s,
                          "prompt_tokens_within_limit_per_s": sum(r["prompt_len"] for r, good in zip(first, ok) if good) / measured_s})
            if cls["kind"] == "chat":
                finished = [r for r in reqs if r["done_ns"] and r["response"] > 0]
                out_tokens = sum(r["response"] for r in finished)
                late = sum(r.get("tokens_late", 0) for r in finished)
                entry.update({
                    "responses_completed": len(finished),
                    "output_tokens_per_s": out_tokens / measured_s,
                    "output_tokens_on_time_per_s": (out_tokens - late) / measured_s,
                    "responses_fully_on_time": sum(1 for r in finished if r.get("tokens_late", 0) == 0
                                                   and r["first_ns"] - (epoch + r["arrival_ns"]) <= slo),
                    "response_ms": dist([(r["done_ns"] - (epoch + r["arrival_ns"])) / 1e6 for r in finished])})
        entry["bound_ms_served"] = {k: v / 1e6 for k, v in cls["bound_served"].items()}
        per_class[cls["name"]] = entry
    timed = [v for v in per_class.values() if v["kind"] != "batch"]
    summary = {
        "arrived": sum(v["arrived"] for v in timed), "rejected": sum(v["rejected"] for v in timed),
        "completed": sum(v["completed"] for v in timed),
        "tokens_per_s": sum(v.get("prompt_tokens_per_s", 0.0) for v in per_class.values()),
        "tokens_within_slo_per_s": sum(v.get("prompt_tokens_within_limit_per_s", 0.0) for v in timed),
        "within_slo": 0, "slo_ms": 0.0, "ttft_ms": dist([]),
        "output_tokens_on_time_per_s": sum(v.get("output_tokens_on_time_per_s", 0.0) for v in timed),
        "items_within_limit_per_s": sum(v.get("items_within_limit_per_s", 0.0) for v in timed),
        "batch_tokens_per_s": sum(v.get("prompt_tokens_per_s", 0.0) for v in per_class.values() if v["kind"] == "batch"),
        "pieces": len(pieces), "units": int(sum(p[2] for p in pieces)),
        "piece_wall_ms": dist([(p[1] - p[0]) / 1e6 for p in pieces]),
        "piece_overruns": 0, "pieces_stopped": stops["count"],
        "gpu_busy_ms": float(sum(p[4] for p in pieces)), "classes": per_class,
        "units_by_size_class": {str(k): int(v) for k, v in units_by_size.items()},
        "allowed_share_at_end": {str(k): float(v) for k, v in allowed_share.items()},
    }
    args.output.write_text(json.dumps({
        "schema": "backstop-slot-ai-worker-v5", "gpu": args.gpu, "pid": os.getpid(),
        "policy": policy, "epoch_ns": epoch, "summary": summary,
        "pieces_columns": ["start_ns", "end_ns", "units", "bound_ns", "gpu_ms"], "pieces": pieces,
    }), encoding="utf-8")


if __name__ == "__main__":
    main()
