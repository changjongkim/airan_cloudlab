#!/usr/bin/env python3
"""Build TRANSPORT_SUMMARY.csv from the checked-in CPU-RDMA/GDR logs.

The first iteration is reported as the cold sample.  Statistics are computed
only from iterations 2 through 10.  The script validates the matching consumer
log before emitting any CSV, so a producer timing is never reported without a
complete receive/verification trace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path


EXPECTED_SEQS = list(range(1, 11))
CPU_LATENCY_RE = re.compile(
    r"^\[prod\] seq=(\d+) write took ([0-9]+(?:\.[0-9]+)?) us$",
    re.MULTILINE,
)
GDR_LATENCY_RE = re.compile(
    r"^\[prod\] seq=(\d+) payload\+seq took ([0-9]+(?:\.[0-9]+)?) us$",
    re.MULTILINE,
)
PAYLOAD_RE = re.compile(r"^\[(?:prod|cons)\] endpoint ready .* payload=(\d+)$", re.MULTILINE)
SESSION_RE = re.compile(
    r"^\[(?:prod|cons)\] endpoint ready session=([0-9a-f]+) payload=\d+$",
    re.MULTILINE,
)
CONSUMER_SEQ_RE = re.compile(r"^\[cons\] got seq=(\d+)\b(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class RunSpec:
    transport: str
    label: str
    payload_bytes: int
    producer_name: str
    consumer_name: str
    gdr: bool


RUNS = (
    RunSpec(
        "CPU_RDMA",
        "1MiB",
        1_048_576,
        "rdma_test_final_prod.log",
        "rdma_test_final_cons.log",
        False,
    ),
    RunSpec(
        "GDR",
        "64KiB",
        65_536,
        "gdr_64k_prod.log",
        "gdr_64k_cons.log",
        True,
    ),
    RunSpec(
        "GDR",
        "NRx_fwd",
        1_415_232,
        "gdr_fwd_1415232_prod.log",
        "gdr_fwd_1415232_cons.log",
        True,
    ),
    RunSpec(
        "GDR",
        "NRx_bwd",
        1_257_984,
        "gdr_bwd_1257984_prod.log",
        "gdr_bwd_1257984_cons.log",
        True,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def linear_percentile(values: list[float], quantile: float) -> float:
    """NumPy-compatible default (linear) percentile without a NumPy dependency."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def require_complete_sequences(pairs: list[tuple[str, str]], path: Path, kind: str) -> None:
    seqs = [int(seq) for seq, _ in pairs]
    if seqs != EXPECTED_SEQS:
        raise ValueError(f"{path}: expected {kind} seq 1..10 exactly once, found {seqs}")


def parse_run(root: Path, spec: RunSpec) -> dict[str, object]:
    producer = root / spec.producer_name
    consumer = root / spec.consumer_name
    missing = [str(path) for path in (producer, consumer) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing raw transport log(s): " + ", ".join(missing))

    producer_text = producer.read_text(encoding="utf-8")
    consumer_text = consumer.read_text(encoding="utf-8")
    latency_re = GDR_LATENCY_RE if spec.gdr else CPU_LATENCY_RE
    timing_pairs = latency_re.findall(producer_text)
    require_complete_sequences(timing_pairs, producer, "producer timing")

    consumer_pairs = CONSUMER_SEQ_RE.findall(consumer_text)
    require_complete_sequences(consumer_pairs, consumer, "consumer receive")
    if "[prod] done" not in producer_text or "[cons] done" not in consumer_text:
        raise ValueError(f"{spec.label}: producer or consumer completion marker is missing")

    if spec.gdr:
        for path, text, role in (
            (producer, producer_text, "prod"),
            (consumer, consumer_text, "cons"),
        ):
            payloads = [int(value) for value in PAYLOAD_RE.findall(text)]
            if payloads != [spec.payload_bytes]:
                raise ValueError(
                    f"{path}: expected one payload={spec.payload_bytes} ready marker, "
                    f"found {payloads}"
                )
            if f"GDR_RDMA_TEST_OK role={role}" not in text:
                raise ValueError(f"{path}: missing GDR success marker for role={role}")
        producer_sessions = SESSION_RE.findall(producer_text)
        consumer_sessions = SESSION_RE.findall(consumer_text)
        if (
            len(producer_sessions) != 1
            or len(consumer_sessions) != 1
            or producer_sessions != consumer_sessions
        ):
            raise ValueError(
                f"{spec.label}: producer/consumer session mismatch: "
                f"{producer_sessions} vs {consumer_sessions}"
            )
        bad_verification = [
            seq for seq, suffix in consumer_pairs if "verified=1" not in suffix
        ]
        if bad_verification:
            raise ValueError(f"{consumer}: unverified consumer seq(s): {bad_verification}")
    else:
        bad_first_words = [
            seq for seq, suffix in consumer_pairs if f"first_word={seq}" not in suffix
        ]
        if bad_first_words:
            raise ValueError(
                f"{consumer}: first-word mismatch for seq(s): {bad_first_words}"
            )

    timings = [float(value) for _, value in timing_pairs]
    cold = timings[0]
    steady = timings[1:]
    return {
        "transport": spec.transport,
        "label": spec.label,
        "payload_bytes": spec.payload_bytes,
        "iterations": len(timings),
        "cold_first_us": f"{cold:.2f}",
        "steady_n": len(steady),
        "steady_mean_us": f"{statistics.fmean(steady):.2f}",
        "steady_p50_us": f"{linear_percentile(steady, 0.50):.2f}",
        "steady_p95_us": f"{linear_percentile(steady, 0.95):.2f}",
        "steady_min_us": f"{min(steady):.2f}",
        "steady_max_us": f"{max(steady):.2f}",
        "steady_values_us": ";".join(f"{value:.2f}" for value in steady),
        "producer_log": producer.relative_to(root).as_posix(),
        "producer_sha256": sha256(producer),
        "consumer_log": consumer.relative_to(root).as_posix(),
        "consumer_sha256": sha256(consumer),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=default_root,
        help=f"directory containing raw logs (default: {default_root})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "TRANSPORT_SUMMARY.csv",
        help="output CSV path",
    )
    args = parser.parse_args()

    root = args.input_root.resolve()
    rows = [parse_run(root, spec) for spec in RUNS]
    write_csv(args.output.resolve(), rows)
    for row in rows:
        print(
            f"{row['transport']} {row['label']} payload={row['payload_bytes']} "
            f"cold={row['cold_first_us']} us steady_mean={row['steady_mean_us']} us"
        )
    print(f"wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
