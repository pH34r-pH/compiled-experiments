#!/usr/bin/env python3
"""Fail closed on incomplete or internally inconsistent public experiment artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def verify_experiment(root: Path) -> list[str]:
    errors: list[str] = []
    required = (
        "README.md",
        "experiment.json",
        "ro-crate-metadata.json",
        "dependency-closure.json",
        "artifact/SHA256SUMS",
        "receipts/experiment-compiler-verify-receipt.json",
        "receipts/offline-replay-receipt.json",
    )
    for rel in required:
        if not (root / rel).is_file():
            errors.append(f"{root.name}: missing {rel}")
    if errors:
        return errors

    recipe = load_json(root / "experiment.json")
    if recipe.get("id") != root.name:
        errors.append(
            f"{root.name}: experiment.json id {recipe.get('id')!r} does not match directory"
        )

    closure = load_json(root / "dependency-closure.json")
    if closure.get("experimentId") != root.name:
        errors.append(f"{root.name}: dependency closure experimentId differs")

    sums_path = root / "artifact" / "SHA256SUMS"
    for line in sums_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            expected, filename = line.split(None, 1)
        except ValueError:
            errors.append(f"{root.name}: malformed SHA256SUMS line {line!r}")
            continue
        filename = filename.lstrip("* ")
        target = sums_path.parent / filename
        if not target.is_file():
            errors.append(f"{root.name}: checksum target missing: {filename}")
            continue
        observed = sha256(target)
        if observed != expected:
            errors.append(
                f"{root.name}: SHA-256 mismatch for {filename}: "
                f"expected {expected}, got {observed}"
            )

    verify = load_json(root / "receipts" / "experiment-compiler-verify-receipt.json")
    if verify.get("status") != "integrity-verified":
        errors.append(f"{root.name}: Compiler verify receipt is not integrity-verified")

    replay = load_json(root / "receipts" / "offline-replay-receipt.json")
    if replay.get("status") != "matched":
        errors.append(f"{root.name}: offline replay receipt is not matched")

    artifact_lines = [
        line for line in sums_path.read_text().splitlines() if line.strip()
    ]
    if len(artifact_lines) != 1:
        errors.append(f"{root.name}: artifact/SHA256SUMS must name exactly one canonical ZIP")
    else:
        expected, filename = artifact_lines[0].split(None, 1)
        filename = filename.lstrip("* ")
        if verify.get("packageSha256") != expected:
            errors.append(f"{root.name}: Compiler receipt digest differs from artifact checksum")
        if verify.get("packageSizeBytes") != (sums_path.parent / filename).stat().st_size:
            errors.append(f"{root.name}: Compiler receipt size differs from canonical artifact")

    return errors


def main() -> int:
    errors: list[str] = []
    load_json(ROOT / ".zenodo.json")
    directories = sorted(path for path in EXPERIMENTS.iterdir() if path.is_dir())
    for directory in directories:
        errors.extend(verify_experiment(directory))
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"archive verification passed for {len(directories)} experiment(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
