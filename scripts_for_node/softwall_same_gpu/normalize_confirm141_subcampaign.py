#!/usr/bin/env python3
"""Relabel reused C139/C140 analyzer output for the C141 independent-node campaign."""

import argparse
import hashlib
import json
from pathlib import Path


def normalize(document, source_path, kind):
    value = dict(document)
    value["derived_from"] = {
        "path": str(source_path),
        "sha256": hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        "reason": "C141 reused a validated prior analyzer; this view corrects campaign-specific schema and claim text only.",
    }
    if kind == "control":
        value["schema"] = "softwall-confirm141-control-requalification-v1"
        value["claim_boundary"] = (
            "Six-arm finite-sample requalification of the corrected 7 ms control "
            "admission bound on nid001372, independent of the C139/C140 node. Not "
            "WCET, production d_MAC, cross-family, restart, or exactly-once evidence."
        )
    else:
        value["schema"] = "softwall-confirm141-ai35-requalification-v1"
        value["claim_boundary"] = (
            "Two-arm finite-sample requalification of the corrected AI35 conditional "
            "class on nid001372, independent of the C139/C140 node. Not throughput "
            "superiority, WCET, production d_MAC, or cross-family evidence."
        )
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--kind", choices=("control", "ai35"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    value = normalize(json.loads(source.read_text()), source, args.kind)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
