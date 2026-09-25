#!/usr/bin/env python3.11
"""Posthoc timing localization for Confirm82's rejected NRx bound candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
BASE = ROOT / "results/softwall_same_gpu"
RAW = BASE / "raw"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    runs = []
    for arm in ("a", "b"):
        prefix = f"confirm82_bound_{arm}_job58738952"
        controller_path = RAW / f"{prefix}_controller.json"
        controller = read(controller_path)
        workers = [read(RAW / f"{prefix}_worker{i}.json") for i in range(2)]
        late = []
        for row in controller["records"]:
            if row["nrx_response_ms"] is None or row["nrx_response_ms"] <= 30:
                continue
            endpoint = int(row["endpoint_id"][-1])
            dispatch_ns = row["nrx_dispatched_ns"]
            window = min(
                workers[endpoint]["execution_windows"],
                key=lambda w: abs(w["observed_ns"] - dispatch_ns),
            )
            group = controller["records"][4 * row["index"]:4 * row["index"] + 4]
            feature_end_ns = max(x["feature_return_ns"] for x in group)
            first_dispatch_ns = min(x["nrx_dispatch_begin_ns"] for x in group
                                    if x["nrx_dispatch_begin_ns"] is not None)
            late.append({
                "release_index": row["index"], "cell": row["cell"],
                "endpoint": row["endpoint_id"],
                "response_ms": row["nrx_response_ms"],
                "release_to_dispatch_ms": (dispatch_ns - row["release_ns"]) / 1e6,
                "dispatch_to_observation_ms":
                    (row["nrx_observed_ns"] - dispatch_ns) / 1e6,
                "features_done_to_first_dispatch_ms":
                    (first_dispatch_ns - feature_end_ns) / 1e6,
                "worker_gpu_ms": window["gpu_ms"],
                "worker_observed_after_dispatch_ms":
                    (window["observed_ns"] - dispatch_ns) / 1e6,
                "worker_published_after_observation_ms":
                    (window["published_ns"] - window["observed_ns"]) / 1e6,
            })
        worst_conv = max(
            (x for x in controller["records"]
             if x["conventional_host_path_ms"] is not None),
            key=lambda x: x["conventional_host_path_ms"],
        )
        requal = read(RAW / f"{prefix}_requalification.json")
        runs.append({
            "arm": arm, "controller_sha256":
                hashlib.sha256(controller_path.read_bytes()).hexdigest(),
            "nrx_above_30_ms": late,
            "worst_conventional": {
                "release_index": worst_conv["index"],
                "cell": worst_conv["cell"],
                "host_path_ms": worst_conv["conventional_host_path_ms"],
                "gpu_ms": worst_conv["conventional_gpu_ms"],
            },
            "requalification_max_gpu_ms":
                max(x["gpu_ms"] for x in requal["records"]),
        })
    report = {
        "schema": "softwall-confirm82-tail-localization-posthoc-v1",
        "runs": runs,
        "interpretation": "The >30 ms NRx response samples have a large feature-completion-to-dispatch gap while the matched endpoint worker GPU intervals remain near 2 ms. The uninstrumented gap includes admission and prepare_neural_ipc; this audit cannot uniquely attribute it. The separate one-cell requalification GPU tail is outside the four-cell co-run candidate mode.",
    }
    output = BASE / "confirm82_tail_localization_posthoc.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"late_nrx": [len(x["nrx_above_30_ms"]) for x in runs],
                      "predispatch_gap_ms": [
                          [round(x["features_done_to_first_dispatch_ms"], 3)
                           for x in run["nrx_above_30_ms"]]
                          for run in runs],
                      "requalification_max_gpu_ms": [
                          run["requalification_max_gpu_ms"] for run in runs]}, indent=2))


if __name__ == "__main__":
    main()
