#!/usr/bin/env python3
"""Check semantic reproducibility of the V15 finite model across hash seeds."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def canonicalize(value):
    if isinstance(value, dict):
        return {key: canonicalize(item) for key, item in value.items()}
    if isinstance(value, list):
        items = [canonicalize(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":")
            ),
        )
    return value


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def audit(verifier, runs):
    verifier = Path(verifier).resolve()
    observations = []
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for seed in range(runs):
            output = directory / ("model_%d.json" % seed)
            environment = dict(os.environ, PYTHONHASHSEED=str(seed))
            subprocess.run(
                [sys.executable, str(verifier), "--output", str(output)],
                check=True,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            raw = output.read_bytes()
            document = json.loads(raw.decode("utf-8"))
            canonical = json.dumps(
                canonicalize(document), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            observations.append({
                "python_hash_seed": seed,
                "raw_sha256": digest_bytes(raw),
                "canonical_sha256": digest_bytes(canonical),
                "all_pass": document.get("all_pass") is True,
                "fault_states": document["fault_protocol_model"]["states"],
                "fault_edges": document["fault_protocol_model"]["edges"],
                "ownership_states": document["single_token_ownership_model"]["states"],
                "ownership_edges": document["single_token_ownership_model"]["edges"],
                "violations": (
                    len(document["fault_protocol_model"]["violations"])
                    + len(document["single_token_ownership_model"]["violations"])
                ),
            })
    canonical_hashes = sorted({row["canonical_sha256"] for row in observations})
    raw_hashes = sorted({row["raw_sha256"] for row in observations})
    gates = {
        "all_runs_pass": all(row["all_pass"] for row in observations),
        "one_canonical_result": len(canonical_hashes) == 1,
        "expected_counts": all(
            row["fault_states"] == 16
            and row["fault_edges"] == 20
            and row["ownership_states"] == 4
            and row["ownership_edges"] == 8
            for row in observations
        ),
        "zero_violations": all(row["violations"] == 0 for row in observations),
    }
    return {
        "schema": "softwall-pipelined-control-model-v2-reproducibility-v1",
        "runs": runs,
        "raw_serialization_variants": len(raw_hashes),
        "canonical_variants": len(canonical_hashes),
        "raw_sha256": raw_hashes,
        "canonical_sha256": canonical_hashes,
        "observations": observations,
        "gates": gates,
        "all_pass": all(gates.values()),
        "interpretation": (
            "State-list order may vary with Python hash seed; canonicalized "
            "state/transition multisets, counts, and invariant outcomes must agree."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--runs", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.verifier, args.runs)
    output = Path(args.output)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    if not result["all_pass"]:
        raise SystemExit("V15 model reproducibility audit failed")


if __name__ == "__main__":
    main()
