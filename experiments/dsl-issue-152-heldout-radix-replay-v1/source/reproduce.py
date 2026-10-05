#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent
PACKAGE = types.ModuleType("domain_scaling_lab")
PACKAGE.__path__ = [str(ROOT / "src" / "domain_scaling_lab")]
sys.modules["domain_scaling_lab"] = PACKAGE
sys.path.insert(0, str(ROOT))

from domain_scaling_lab.closed_log_issue152_validation import run_issue152_validation
import compare_reproduction

PREREG_SHA256 = "ced3b77e37a9a58dbc62b4b6b3a47bd5113e9f3210283d54060561d581861490"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reproduction"))
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    prereg = ROOT / "preregistration.json"
    if sha256(prereg) != PREREG_SHA256:
        raise SystemExit("frozen preregistration identity differs")

    shutil.copytree(ROOT / "traces", args.output_dir / "traces")
    candidate = run_issue152_validation(
        preregistration_path=prereg,
        output=args.output_dir,
        capture=False,
        expected_preregistration_sha256=PREREG_SHA256,
    )
    reference = compare_reproduction.load(PACKAGE_ROOT / "evidence" / "reference" / "summary.json")
    comparison = compare_reproduction.compare(reference, candidate)
    (args.output_dir / "reproduction-comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    receipt = {
        "schemaVersion": 1,
        "status": "matched" if comparison["status"] == "matched" else "different",
        "decision": candidate["decision"],
        "validatedPairCount": candidate["validated_pair_count"],
        "requiredPairCount": candidate["required_pair_count"],
        "referenceSummarySha256": sha256(PACKAGE_ROOT / "evidence" / "reference" / "summary.json"),
        "candidateSummarySha256": sha256(args.output_dir / "summary.json"),
        "traceManifestSha256": sha256(ROOT / "traces" / "manifest.json"),
    }
    (args.output_dir / "reproduction-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "matched" else 2


if __name__ == "__main__":
    raise SystemExit(main())
