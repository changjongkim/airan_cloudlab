#!/usr/bin/env python3.11
"""Build one branch's physical peer list from the frozen holdout protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from integrated_shared_recovery_holdout_plan_v1 import build_branch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    matching = [row for row in protocol["arms"] if row["branch"] == args.branch]
    if len(matching) != 1:
        raise ValueError(f"branch {args.branch} missing or duplicated")
    arm = matching[0]
    plan = build_branch(args.branch)
    peers = []
    for index, key in enumerate(plan["physical_recovery_keys"]):
        name = "/".join(key)
        home_index = int(key[0].removeprefix("home"))
        safe_key = f"{key[0]}_{key[1]}"
        peers.append({
            "key": list(key),
            "home_index": home_index,
            "source_device": home_index,
            "receiver_seed": arm["seeds"]["receiver"][name],
            "channel_seed": arm["seeds"]["channel"][name],
            "tag": f"{args.tag_prefix}_{safe_key}",
            "owner_output": str(args.raw_dir / f"{args.tag_prefix}_{safe_key}_owner.json"),
            "peer_position": index,
        })
    value = {
        "schema": "softwall-c154-peer-spec-v1",
        "protocol": str(args.protocol),
        "branch": args.branch,
        "arm_index": arm["arm_index"],
        "physical_recovery_keys": [list(key) for key in plan["physical_recovery_keys"]],
        "peers": peers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    for row in peers:
        print("\t".join(map(str, (
            row["home_index"], row["key"][1], row["source_device"],
            row["receiver_seed"], row["channel_seed"], row["tag"],
            row["owner_output"],
        ))))


if __name__ == "__main__":
    main()

