#!/usr/bin/env python3.11
"""Advance an ambiguous journal only from a valid GPU-quiescence certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def durable_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--new-worker-epoch", required=True)
    args = parser.parse_args()
    journal = json.loads(args.journal.read_text())
    certificate = json.loads(args.certificate.read_text())
    before = certificate.get("journal_before", {})
    if certificate.get("all_pass") is not True:
        raise RuntimeError("quiescence certificate did not pass")
    if (journal.get("identity") != before.get("identity")
            or journal.get("stage") != before.get("stage")
            or journal.get("journal_seq") != before.get("journal_seq")):
        raise RuntimeError("journal changed after certificate input")
    terminal = certificate["decision"]["terminal_stage"]
    journal.update({
        "stage": terminal, "journal_seq": journal["journal_seq"] + 1,
        "quiescence_fence_count": 1,
        "quiescence_certificate": str(args.certificate),
        "quiescence_certificate_sha256": sha256(args.certificate),
        "recovered_by_worker_epoch": args.new_worker_epoch,
        "recovered_ns": time.perf_counter_ns(),
    })
    durable_json(args.journal, journal)
    print(json.dumps(journal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
