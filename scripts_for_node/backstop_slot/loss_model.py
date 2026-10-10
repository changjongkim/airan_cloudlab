#!/usr/bin/env python3
"""Recovery-loss model: neural receivers as a loss system whose customers cannot wait.

Model
-----
N neural receivers serve the candidates of W two-user cells.  Uplink slots arrive every P ms.
A candidate of the slot that arrived at time kP becomes known at kP + c (the conventional
receiver has finished) and must start by kP + L, with L = recovery deadline - time bound of
the neural receiver.  A neural receiver run takes S0 ms alone and S1 ms next to AI; a run that
is next to AI for part of the time advances at rate S0/S1 during that part.  A candidate that
finds no free neural receiver by its latest start is lost.

A neural receiver that starts at offset s of slot k serves a candidate of slot k+m only if
s + S <= mP + L.  The number of slots it is held is therefore  h = ceil((s + S - L) / P):
with P = 2.5, L = 3.9 and s = 1.4, a run of 6.3 ms holds it for two slots and a run longer
than 7.5 ms for three.  AI costs recoveries by moving runs from h = 2 to h = 3, not by making
them miss the recovery deadline.

Two evaluations of the model:
  simulate()   event simulation with the measured candidate arrivals of a run (or synthetic
               Bernoulli arrivals), any AI rule, and a stop latency for AI pieces in flight
  chain()      closed form: a Markov chain over the neural receivers that are held for one or
               two more slots, with Binomial(W, q) arrivals and a share f of three-slot runs
  chain_markov()  the same with dependent arrivals: the number of candidates per slot follows a
               first-order Markov chain measured from a run (a cell that fails in one slot is
               more likely to fail in the next)
  lane_chain()  closed form that needs no measured three-slot share: it derives the share from
               the distributions of the time a candidate becomes known and of the run length

AI rules (when AI may run next to a neural receiver):
  none        no AI
  always      whenever the neural receiver runs (fixed share, low priority)
  never       never
  reserve:R   while at least R neural receivers are free (the v13 rule)
  reuse:R     the rule of controller5: while the run, finished alone from now on, still ends
              before its reuse time (the latest start of the candidates two slots later; three
              slots later for a run that started too late for two).  While at least R neural
              receivers are free (R > 0) only the three-slot reuse time limits the overlap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Setting:
    lanes: int = 4
    period_ms: float = 2.5
    latest_start_ms: float = 3.9        # L = recovery deadline - neural receiver bound
    run_alone_ms: float = 6.28          # S0
    run_alone_sd_ms: float = 0.08
    run_ai_ms: float = 7.97             # S1 (next to AI for the whole run)
    known_shift_ms: float = 0.0         # AI delays the conventional result by this much
    stop_latency_ms: float = 0.0        # an AI piece in flight keeps running this long
    yield_to_waiting: bool = True       # AI stops everywhere while a candidate waits
    slack_margin_ms: float = 0.4        # run length used for the prediction minus the median run
    step_ms: float = 0.05


def synthetic_arrivals(cells: int, q: float, periods: int, known_ms: np.ndarray, seed: int = 0):
    """(period, cell, known offset) of candidates drawn independently with probability q."""
    rng = np.random.default_rng(seed)
    hits = rng.random((periods, cells)) < q
    period, cell = np.nonzero(hits)
    return period, cell, rng.choice(known_ms, size=len(period))


def simulate(period, known_ms, rule: str, setting: Setting, seed: int = 0, periods: int | None = None) -> dict:
    """Run the model on a list of candidates.  Returns losses, run lengths and the time AI may run."""
    s = setting
    rng = np.random.default_rng(seed)
    n = len(period)
    order = np.lexsort((known_ms, period))
    period, known_ms = np.asarray(period)[order], np.asarray(known_ms)[order]
    release = period * s.period_ms
    known = release + known_ms + s.known_shift_ms
    latest = release + s.latest_start_ms
    reuse = release + 2 * s.period_ms + s.latest_start_ms          # reuse time of a run of this slot
    work = np.maximum(1.0, rng.normal(s.run_alone_ms, s.run_alone_sd_ms, n))
    slow = s.run_alone_ms / s.run_ai_ms                              # progress rate next to AI
    reserve = int(rule.split(":")[1]) if ":" in rule else 0
    reuse3 = np.minimum(release + 3 * s.period_ms + s.latest_start_ms, release + s.latest_start_ms + s.run_alone_ms + 1.3)
    horizon = (int(periods) if periods else int(period.max()) + 4) * s.period_ms
    dt = s.step_ms

    lane_job = [-1] * s.lanes
    lane_left = [0.0] * s.lanes
    lane_co_until = [0.0] * s.lanes          # AI piece in flight on this GPU runs until then
    started = np.full(n, -1.0)
    finished = np.full(n, -1.0)
    co_time = np.zeros(n)
    lost = np.zeros(n, dtype=bool)
    waiting: list[int] = []
    nxt = 0
    ai_free = ai_co = 0.0                    # GPU-time AI runs with no neural receiver / next to one
    ai_on = [rule != "none"] * s.lanes       # is AI running on the GPU right now (for the stop latency)
    t = 0.0
    while t < horizon:
        while nxt < n and known[nxt] <= t:
            waiting.append(nxt)
            nxt += 1
        if waiting:
            waiting.sort(key=lambda j: latest[j])
            for j in list(waiting):
                if t > latest[j]:
                    lost[j] = True
                    waiting.remove(j)
                    continue
                free = [i for i in range(s.lanes) if lane_job[i] < 0]
                if not free:
                    break
                # Prefer a GPU where AI is not running, as the controller does.
                i = min(free, key=lambda g: ai_on[g])
                lane_job[i], lane_left[i] = j, work[j]
                started[j] = t
                if ai_on[i] and rule not in ("none",):
                    lane_co_until[i] = t + s.stop_latency_ms
                waiting.remove(j)
        busy = sum(1 for i in range(s.lanes) if lane_job[i] >= 0)
        hold = s.yield_to_waiting and bool(waiting) and rule not in ("always", "none")
        for i in range(s.lanes):
            j = lane_job[i]
            if j < 0:
                ai_on[i] = rule != "none" and not hold
                ai_free += dt if ai_on[i] else 0.0
                continue
            if rule == "none":
                allow = False
            elif rule == "always":
                allow = True
            elif hold or rule == "never":
                allow = False
            elif rule.startswith("reserve"):
                allow = s.lanes - busy >= reserve
            else:   # reuse: finished alone from now on, the run must still end by its reuse time
                limit = reuse3[j]
                plenty = reserve > 0 and s.lanes - busy >= reserve
                if not plenty and started[j] + work[j] + s.slack_margin_ms <= reuse[j]:
                    limit = reuse[j]
                allow = t + lane_left[i] + s.slack_margin_ms <= limit
            if allow:
                lane_co_until[i] = t + s.stop_latency_ms
            co = allow or t < lane_co_until[i]
            ai_on[i] = co
            if co:
                lane_left[i] -= dt * slow
                co_time[j] += dt
                ai_co += dt
            else:
                lane_left[i] -= dt
            if lane_left[i] <= 0:
                finished[j] = t + dt
                lane_job[i] = -1
        t += dt
    ran = started >= 0
    run = finished[ran] - started[ran]
    offset = started[ran] - release[ran]
    held = np.ceil((offset + run - s.latest_start_ms) / s.period_ms)
    gpu_time = horizon * s.lanes
    return {
        "candidates": int(n), "lost": int(lost.sum()), "lost_share": float(lost.mean()) if n else 0.0,
        "run_ms_p50": float(np.median(run)) if ran.any() else 0.0,
        "held_3_or_more": float((held >= 3).mean()) if ran.any() else 0.0,
        "late_share": float(((offset + run) > s.latest_start_ms + s.run_alone_ms + 1.3).mean()) if ran.any() else 0.0,
        "ai_time_free": ai_free / gpu_time, "ai_time_next_to_nrx": ai_co / gpu_time,
        "ai_time": (ai_free + ai_co) / gpu_time,
        "lost_mask": lost[np.argsort(order)],
        "end_offset_ms": finished[ran] - release[ran],    # end of each run, from the arrival of its slot
    }


def choose_reserve(period, known_ms, setting: Setting, tolerance: float = 0.003, periods: int | None = None) -> dict:
    """The rule the model picks: the smallest reserve (most AI time) whose loss stays within
    ``tolerance`` (share of the candidates) of the loss without any AI next to a neural receiver.
    A reserve equal to the number of neural receivers means AI never runs next to one."""
    never = simulate(period, known_ms, "never", setting, periods=periods)
    table = {setting.lanes: (never["lost_share"], never["ai_time"])}
    chosen = setting.lanes
    for reserve in range(setting.lanes - 1, 0, -1):
        r = simulate(period, known_ms, f"reserve:{reserve}", setting, periods=periods)
        table[reserve] = (r["lost_share"], r["ai_time"])
        if r["lost_share"] <= never["lost_share"] + tolerance:
            chosen = reserve
        else:
            break
    return {"reserve": chosen, "by_reserve": table}


def chain(lanes: int, cells: int, q: float, three_slot_share: float) -> float:
    """Share of candidates lost in the slot model: every run holds its neural receiver for two
    slots, or three with probability ``three_slot_share``; arrivals are Binomial(cells, q)."""
    f = three_slot_share
    arrivals = [math.comb(cells, a) * q ** a * (1 - q) ** (cells - a) for a in range(cells + 1)]
    states = [(n1, n2) for n1 in range(lanes + 1) for n2 in range(lanes + 1 - n1)]
    index = {state: i for i, state in enumerate(states)}
    move = np.zeros((len(states), len(states)))
    loss = np.zeros(len(states))
    for (n1, n2), i in index.items():
        free = lanes - n1 - n2
        for a, pa in enumerate(arrivals):
            served = min(a, free)
            loss[i] += pa * (a - served)
            for three in range(served + 1):
                p3 = math.comb(served, three) * f ** three * (1 - f) ** (served - three)
                # held for one more slot: the two-slot runs of this slot and the three-slot runs of the last
                move[i, index[(n2 + served - three, three)]] += pa * p3
    dist = np.full(len(states), 1.0 / len(states))
    for _ in range(2000):
        dist = dist @ move
    return float(dist @ loss) / (cells * q)


def arrival_transitions(period, periods: int, most: int) -> np.ndarray:
    """T[a, b] = P(b candidates in a slot | a candidates in the slot before), from a list of candidates."""
    per = np.minimum(most, np.bincount(np.asarray(period), minlength=periods)[:periods])
    table = np.zeros((most + 1, most + 1))
    np.add.at(table, (per[:-1], per[1:]), 1.0)
    overall = np.bincount(per, minlength=most + 1) / len(per)
    for a in range(most + 1):
        table[a] = table[a] / table[a].sum() if table[a].sum() else overall
    return table


def chain_markov(lanes: int, transitions: np.ndarray, three_slot_share: float) -> float:
    """Closed form with dependent arrivals: the number of candidates per slot is a first-order
    Markov chain (``transitions`` from ``arrival_transitions``); every run holds its neural
    receiver for two slots, or three with probability ``three_slot_share``.
    State: (receivers held one more slot, receivers held two more slots, candidates of this slot).
    Returns the share of candidates that find no free neural receiver."""
    f = three_slot_share
    most = transitions.shape[0]
    states = [(n1, n2, a) for n1 in range(lanes + 1) for n2 in range(lanes + 1 - n1) for a in range(most)]
    index = {state: i for i, state in enumerate(states)}
    move = np.zeros((len(states), len(states)))
    loss = np.zeros(len(states))
    arrivals = np.zeros(len(states))
    for (n1, n2, a), i in index.items():
        served = min(a, lanes - n1 - n2)
        loss[i], arrivals[i] = a - served, a
        for three in range(served + 1):
            p3 = math.comb(served, three) * f ** three * (1 - f) ** (served - three)
            for nxt in range(most):
                move[i, index[(n2 + served - three, three, nxt)]] += p3 * transitions[a, nxt]
    dist = np.full(len(states), 1.0 / len(states))
    for _ in range(4000):
        dist = dist @ move
    return float(dist @ loss) / max(1e-12, float(dist @ arrivals))


def _outcomes(count: int, probs) -> list[tuple[tuple[int, int, int], float]]:
    """All splits of ``count`` runs into (two-slot, three-slot, four-slot) with their probabilities."""
    out = []
    for h3 in range(count + 1):
        for h4 in range(count + 1 - h3):
            h2 = count - h3 - h4
            weight = (math.factorial(count) / (math.factorial(h2) * math.factorial(h3) * math.factorial(h4))
                      * probs[0] ** h2 * probs[1] ** h3 * probs[2] ** h4)
            if weight > 0.0:
                out.append(((h2, h3, h4), weight))
    return out


def lane_chain(lanes: int, transitions: np.ndarray, known_ms, run_ms, period_ms: float = 2.5,
               latest_start_ms: float = 3.9, rounds: int = 6, samples: int = 60000, seed: int = 0) -> dict:
    """Closed form without a measured three-slot share.

    Inputs: number of neural receivers, transition matrix of the candidates per slot, samples of
    the time at which a candidate becomes known (offset in its slot) and of the run length of
    the neural receiver under the policy.  The share of runs that hold their receiver for three
    slots follows from these two distributions:

      a run that starts at offset s and takes S holds its receiver for h = ceil((s + S - L) / P)
      slots and frees it at offset s + S - hP of slot k + h;
      a candidate that finds an idle receiver starts when it is known (s = c);
      one that only finds a receiver that frees in its own slot starts at max(c, free time).

    Lane states: idle; J2 / J3 = frees in this slot after a two-slot / longer run; A1 = held this
    slot, J2 next; B1, B2, B3 = held one, two, three more slots, J3 afterwards.  Candidates take
    idle receivers first, then J3 (they free early in the slot), then J2.  The free-time
    distributions of J2 and J3 and the chain are iterated to a fixed point.
    Returns the lost share, the three-slot share and the share of runs that waited for a receiver.
    """
    rng = np.random.default_rng(seed)
    known, run = np.asarray(known_ms, dtype=float), np.asarray(run_ms, dtype=float)
    P, L = period_ms, latest_start_ms
    c = rng.choice(known, samples)
    S = rng.choice(run, samples)

    def held(start):
        """(slots held, free offset) of runs that start at ``start`` (array) with run lengths S."""
        end = start + S
        h = np.clip(np.ceil((end - L) / P), 2, 4)
        return h, end - h * P

    most = transitions.shape[0]
    kinds = ("idle", "j3", "j2")
    states = [st for st in np.ndindex(*([lanes + 1] * 6)) if sum(st) <= lanes]      # (J2, J3, A1, B1, B2, B3)
    index = {(st, a): i for i, (st, a) in enumerate((st, a) for st in states for a in range(most))}
    rates = {"idle": 1.0, "j3": 0.0, "j2": 0.0}
    phi2 = phi3 = None
    result = {}
    for _ in range(rounds):
        start = {"idle": c,
                 "j3": c if phi3 is None else np.maximum(c, rng.choice(phi3, samples)),
                 "j2": c if phi2 is None else np.maximum(c, rng.choice(phi2, samples))}
        probs, free2, free3 = {}, [], []
        for kind in kinds:
            h, phi = held(start[kind])
            probs[kind] = (float((h == 2).mean()), float((h == 3).mean()), float((h == 4).mean()))
            take = max(1, int(samples * rates[kind] / max(1e-9, sum(rates.values()))))
            two, more = phi[h == 2], phi[h > 2]
            if len(two):
                free2.append(rng.choice(two, min(take, samples)))
            if len(more):
                free3.append(rng.choice(more, min(take, samples)))
        phi2 = np.concatenate(free2) if free2 else None
        phi3 = np.concatenate(free3) if free3 else None
        move = np.zeros((len(index), len(index)))
        lost = np.zeros(len(index))
        arrivals = np.zeros(len(index))
        started = {kind: np.zeros(len(index)) for kind in kinds}
        long_runs = np.zeros(len(index))
        for (st, a), i in index.items():
            j2, j3, a1, b1, b2, b3 = st
            idle = lanes - sum(st)
            x = min(a, idle)
            y3 = min(a - x, j3)
            y2 = min(a - x - y3, j2)
            lost[i], arrivals[i] = a - x - y3 - y2, a
            started["idle"][i], started["j3"][i], started["j2"][i] = x, y3, y2
            for (h2x, h3x, h4x), wx in _outcomes(x, probs["idle"]):
                for (h2y, h3y, h4y), wy in _outcomes(y3, probs["j3"]):
                    for (h2z, h3z, h4z), wz in _outcomes(y2, probs["j2"]):
                        weight = wx * wy * wz
                        nxt = (a1, b1, h2x + h2y + h2z, b2, b3 + h3x + h3y + h3z, h4x + h4y + h4z)
                        long_runs[i] += weight * (h3x + h3y + h3z + h4x + h4y + h4z)
                        for a2 in range(most):
                            move[i, index[(nxt, a2)]] += weight * transitions[a, a2]
        dist = np.full(len(index), 1.0 / len(index))
        for step in range(3000):
            nxt_dist = dist @ move
            if step % 20 == 19 and np.abs(nxt_dist - dist).max() < 1e-10:
                dist = nxt_dist
                break
            dist = nxt_dist
        rates = {kind: float(dist @ started[kind]) for kind in kinds}
        runs = sum(rates.values())
        result = {"lost_share": float(dist @ lost) / max(1e-12, float(dist @ arrivals)),
                  "three_slot_share": float(dist @ long_runs) / max(1e-12, runs),
                  "waited_share": (rates["j2"] + rates["j3"]) / max(1e-12, runs),
                  "three_slot_by_start": {kind: probs[kind][1] + probs[kind][2] for kind in kinds}}
    return result


if __name__ == "__main__":
    for f in (0.0, 0.1, 0.5, 1.0):
        print(f"three-slot share {f:.1f}: lost {100 * chain(4, 4, 0.16, f):.1f}% of the candidates (4 receivers, 4 cells, q=0.16)")
