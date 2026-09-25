#!/usr/bin/env python3
"""Build the deterministic context-64 trace for the corrected AI35 class."""

import argparse
import json
from pathlib import Path


def build(count=720, interarrival_ms=45.0, deadline_ms=1000.0):
    requests = []
    for index in range(count):
        requests.append({
            "request_id": "softwall-ai35-%04d" % index,
            "source_row": index,
            "source_timestamp_s": None,
            "arrival_ms": index * interarrival_ms,
            "deadline_ms": deadline_ms,
            "raw_request_tokens": 64,
            "raw_response_tokens": 0,
            "context_length": 64,
            "value_tokens": 64,
            "model": "Qwen2.5-1.5B",
            "log_type": "synthetic mechanism qualification",
            "source_order": index,
        })
    return {
        "schema": "softwall-context64-mechanism-trace-v1",
        "source": "deterministic synthetic mechanism trace; not BurstGPT",
        "license": "repository experiment artifact",
        "selection": {
            "requests": count,
            "interarrival_ms": interarrival_ms,
            "deadline_ms": deadline_ms,
        },
        "mapping": {
            "phase": "Qwen2.5-1.5B prefill",
            "allowed_context_lengths": [64],
            "rule": "every request is the corrected raw AI35 candidate class",
            "value": "64 tokens completed before the synthetic request deadline",
        },
        "summary": {
            "requests": count,
            "arrival_duration_ms": (count - 1) * interarrival_ms,
            "offered_value_tokens": count * 64,
            "bucket_counts": {"64": count},
        },
        "requests": requests,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build()
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
