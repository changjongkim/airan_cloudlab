#!/usr/bin/env python3
"""Stage and fingerprint public data/model assets used by workload qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from torchvision.datasets import CIFAR10
from torchvision.models import ResNet50_Weights, resnet50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--alpaca", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    alpaca = Path(args.alpaca)
    if not alpaca.is_file():
        raise RuntimeError(f"missing text dataset: {alpaca}")
    records = json.loads(alpaca.read_text(encoding="utf-8"))
    if len(records) < 50000:
        raise RuntimeError(f"unexpected Alpaca record count: {len(records)}")

    train = CIFAR10(root=str(dataset_root), train=True, download=True)
    _ = resnet50(weights=ResNet50_Weights.DEFAULT)
    archive = dataset_root / "cifar-10-python.tar.gz"
    if not archive.is_file():
        raise RuntimeError(f"CIFAR-10 archive was not retained: {archive}")
    result = {
        "alpaca": {
            "path": str(alpaca),
            "sha256": sha256(alpaca),
            "records": len(records),
            "source": "https://github.com/tatsu-lab/stanford_alpaca",
            "production_trace": False,
        },
        "cifar10": {
            "root": str(dataset_root),
            "archive": str(archive),
            "sha256": sha256(archive),
            "train_samples": len(train),
            "source": "https://www.cs.toronto.edu/~kriz/cifar.html",
        },
        "resnet50": {
            "weights": str(ResNet50_Weights.DEFAULT),
            "source": "torchvision model registry",
        },
        "pass": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(f"[DATA-PREP] PASS output={output}", flush=True)


if __name__ == "__main__":
    main()
