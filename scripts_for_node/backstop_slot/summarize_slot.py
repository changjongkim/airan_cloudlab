"""Single-commit accounting for one slot-scale run.

For every transport block the first CRC-valid result commits.  When no path
passes, the UL indication carries a NACK as soon as every launched path has
finished.  A result is on time when it is ready within the deadline measured
from data arrival.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def pct(values, q) -> float | None:
    return float(np.percentile(values, q)) if len(values) else None


def dist(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"n": 0}
    return {
        "n": int(values.size), "mean": float(values.mean()),
        "p50": pct(values, 50), "p90": pct(values, 90), "p99": pct(values, 99),
        "p999": pct(values, 99.9), "max": float(values.max()),
    }


def summarize(config: dict, work: Path, epoch: int) -> dict:
    deadline_ms = float(config["deadline_ms"])
    rescue_ms = float(config.get("rescue_deadline_ms", deadline_ms))
    extra_ms = float(config.get("report_extra_deadline_ms", 0.5))
    skip = int(config.get("skip_periods", 20))
    cells = config["cells"]
    per_profile: dict[str, dict] = {}
    rows = []
    final_path = work / "final_cells.npy"
    final = np.load(final_path) if final_path.is_file() else None
    lane_rows = {}
    for path in sorted(work.glob("lane*.json")):
        for r in json.loads(path.read_text())["records"]:
            lane_rows[(r[0], r[1])] = r
    reason_names = {1: "conv_fail", 2: "latest_start", 3: "arrival", 4: "low_value"}
    for cell in cells:
        index = int(cell["cell"])
        conv = json.loads((work / f"conv{index}.json").read_text())
        nrx_path = work / f"nrx{index}.json"
        nrx = json.loads(nrx_path.read_text()) if nrx_path.is_file() else None
        nrx_rows = {r[0]: r for r in nrx["records"]} if nrx else {}
        if nrx is None and (lane_rows or final is not None) and cell.get("nrx_gpu") is not None \
                and config["nrx_policy"] != "off":
            for period in range(int(config["periods"])):
                ran = lane_rows.get((index, period))
                reason = reason_names.get(int(final[index, period, 7]), None) if final is not None else None
                status = int(final[index, period, 3]) if final is not None else 0
                if ran is not None and final is not None and int(final[index, period, 7]) == 4:
                    ran = None
                if ran is not None:
                    nrx_rows[period] = [period, 0, 0, ran[2], ran[3], ran[4], ran[5],
                                        reason or "arrival", *ran[6:8]]
                elif status == 3:
                    nrx_rows[period] = [period, 0, 0, 0, 0, -1, -1, "skipped"]
                elif status == 5:
                    nrx_rows[period] = [period, 0, 0, 0, 0, -1, -1,
                                        "low_value" if reason == "low_value" else "dropped"]
        # A MU-MIMO cell carries several TBs (one per UE) in a slot: both receivers decode all
        # of them in one run, so the timing is shared and the outcome is per UE.
        num_ue = int(cell.get("num_ue", 1))
        for record, ue in ((r, u) for r in conv["records"] for u in range(num_ue)):
            period, _, release, c_start, c_done, c_ok, c_payload = record[:7]
            if period < skip:
                continue
            if num_ue > 1:
                c_ok, c_payload = (record[8] >> ue) & 1, (record[9] >> ue) & 1
            row = {
                "cell": index, "gpu": cell["gpu"], "profile": cell["profile"],
                "period": period, "ue": ue,
                "conv_ms": (c_done - release) / 1e6,
                "conv_start_lag_ms": (c_start - release) / 1e6,
                "conv_ok": bool(c_ok), "conv_payload_ok": bool(c_payload),
                "nrx_ran": False, "nrx_ok": False, "nrx_ms": None,
                "nrx_busy_ms": 0.0, "nrx_reason": None,
            }
            ready = [(c_done, bool(c_ok))]
            n = nrx_rows.get(period)
            if n is not None:
                row["nrx_reason"] = n[7]
                if n[7] not in ("skipped", "dropped", "low_value"):
                    _, _, _, n_start, n_done, n_ok, n_payload, _ = n[:8]
                    if num_ue > 1:
                        n_ok, n_payload = (n[8] >> ue) & 1, (n[9] >> ue) & 1
                    row.update({
                        "nrx_ran": True, "nrx_ok": bool(n_ok),
                        "nrx_payload_ok": bool(n_payload),
                        "nrx_ms": (n_done - release) / 1e6,
                        "nrx_busy_ms": (n_done - n_start) / 1e6 / num_ue,
                        "nrx_start_ms": (n_start - release) / 1e6,
                    })
                    ready.append((n_done, bool(n_ok)))
            # The controller never waits past the deadline for an optional
            # path: once the mandatory conventional result is in, a TB without
            # a valid CRC by the deadline is committed as a NACK at the
            # deadline.  The UL indication is late only when the conventional
            # result itself (or a valid NeuralRx result) misses the deadline.
            deadline_abs = release + deadline_ms * 1e6
            valid = [t for t, ok in ready if ok]
            first_valid = min(valid) if valid else None
            conv_ready = c_done
            if first_valid is not None and first_valid <= deadline_abs:
                commit = first_valid
                row["decoded_on_time"] = True
            elif conv_ready <= deadline_abs:
                pending = [t for t, _ in ready]
                commit = min(max(pending), deadline_abs)
                row["decoded_on_time"] = False
            else:
                commit = min(t for t in [conv_ready, first_valid] if t is not None)
                row["decoded_on_time"] = False
            row["decoded"] = first_valid is not None
            row["ready_ms"] = (commit - release) / 1e6
            row["on_time"] = commit <= deadline_abs
            row["on_time_extra"] = commit <= deadline_abs + extra_ms * 1e6
            row["nrx_late"] = bool(row["nrx_ran"] and row["nrx_ms"] > rescue_ms)
            # A rescue needs the right payload, not only a passing CRC.
            row["rescued"] = (not row["conv_ok"]) and row["nrx_ok"] and bool(
                row.get("nrx_payload_ok", True)
            ) and bool(row["nrx_ms"] is not None and row["nrx_ms"] <= rescue_ms)
            row["decoded_final"] = (row["conv_ok"] and row["conv_ms"] <= deadline_ms) or row["rescued"]
            row["false_pass"] = (row["conv_ok"] and not row["conv_payload_ok"]) or (
                row["nrx_ok"] and not row.get("nrx_payload_ok", True)
            )
            rows.append(row)

    def block(selected: list[dict]) -> dict:
        if not selected:
            return {"tbs": 0}
        count = len(selected)
        return {
            "tbs": count,
            "ul_indication_on_time": sum(r["on_time"] for r in selected) / count,
            "ul_indication_on_time_plus_extra": sum(r["on_time_extra"] for r in selected) / count,
            "late_tbs": sum(not r["on_time"] for r in selected),
            "decoded_on_time": sum(r["decoded_on_time"] for r in selected) / count,
            "decoded_final": sum(r["decoded_final"] for r in selected) / count,
            "retransmissions": sum(not r["decoded_final"] for r in selected),
            "conv_crc_pass": sum(r["conv_ok"] for r in selected) / count,
            "nrx_runs": sum(r["nrx_ran"] for r in selected),
            "conv_fail_tbs": sum(not r["conv_ok"] for r in selected),
            "nrx_late": sum(r["nrx_late"] for r in selected),
            "nrx_rescues_on_time": sum(r["rescued"] for r in selected),
            "nrx_gpu_busy_ms_per_tb": float(np.mean([r["nrx_busy_ms"] for r in selected])),
            "false_crc_pass": sum(r["false_pass"] for r in selected),
            "conv_done_ms": dist([r["conv_ms"] for r in selected]),
            "conv_start_lag_ms": dist([r["conv_start_lag_ms"] for r in selected]),
            "nrx_done_ms": dist([r["nrx_ms"] for r in selected if r["nrx_ran"]]),
            "ready_ms": dist([r["ready_ms"] for r in selected]),
            "nrx_start_reasons": {
                reason: sum(r["nrx_reason"] == reason for r in selected)
                for reason in ("arrival", "conv_fail", "latest_start", "skipped", "dropped", "low_value")
            },
        }

    for name in sorted({r["profile"] for r in rows}):
        per_profile[name] = block([r for r in rows if r["profile"] == name])
    per_gpu = {
        str(g): block([r for r in rows if r["gpu"] == g])
        for g in sorted({r["gpu"] for r in rows})
    }
    everything = block(rows)
    ai = {}
    for path in sorted(work.glob("ai*.json")):
        ai[path.stem] = json.loads(path.read_text()).get("summary")
    controller_path = work / "controller.json"
    controller = json.loads(controller_path.read_text()) if controller_path.is_file() else None
    ai_total = None
    if ai:
        ai_total = {
            "completed": sum(v["completed"] for v in ai.values()),
            "arrived": sum(v["arrived"] for v in ai.values()),
            "tokens_per_s": sum(v["tokens_per_s"] for v in ai.values()),
            "tokens_within_slo_per_s": sum(v["tokens_within_slo_per_s"] for v in ai.values()),
            "within_slo": sum(v["within_slo"] for v in ai.values()),
            "piece_overruns": sum(v["piece_overruns"] for v in ai.values()),
            "pieces": sum(v["pieces"] for v in ai.values()),
            "gpu_busy_ms": sum(v["gpu_busy_ms"] for v in ai.values()),
        }
    headline = {
        "cells": len(cells),
        "nrx_policy": config["nrx_policy"],
        "ai_policy": config.get("ai_policy", "none"),
        "deadline_ms": deadline_ms,
        "tbs": everything["tbs"],
        "ul_indication_on_time": everything.get("ul_indication_on_time"),
        "late_tbs": everything.get("late_tbs"),
        "decoded_on_time": everything.get("decoded_on_time"),
        "rescue_deadline_ms": rescue_ms,
        "decoded_final": everything.get("decoded_final"),
        "retransmissions": everything.get("retransmissions"),
        "nrx_rescues_on_time": everything.get("nrx_rescues_on_time"),
        "nrx_runs": everything.get("nrx_runs"),
        "false_crc_pass": everything.get("false_crc_pass"),
        "ready_ms_p99": everything.get("ready_ms", {}).get("p99"),
        "nrx_late": everything.get("nrx_late"),
        "ai_total": ai_total,
    }
    return {
        "schema": "backstop-slot-run-v1",
        "headline": headline,
        "all": everything,
        "per_profile": per_profile,
        "per_gpu": per_gpu,
        "ai": ai,
        "controller": controller,
    }
