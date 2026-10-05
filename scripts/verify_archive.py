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
        "artifacts/SHA256SUMS",
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

    prerequisites = {
        item.get("item"): item.get("requirement")
        for item in closure.get("classifications", {}).get("hostPrerequisites", [])
        if isinstance(item, dict)
    }
    requirements = (root / "source" / "requirements.txt").read_text().splitlines()
    numpy_requirements = [
        line.split("==", 1)[1]
        for line in requirements
        if line.startswith("numpy==") and "==" in line
    ]
    if len(numpy_requirements) != 1:
        errors.append(f"{root.name}: source/requirements.txt must pin exactly one NumPy version")
    elif prerequisites.get("NumPy") != numpy_requirements[0]:
        errors.append(
            f"{root.name}: dependency closure NumPy {prerequisites.get('NumPy')!r} "
            f"differs from replay requirement {numpy_requirements[0]!r}"
        )

    sums_path = root / "artifacts" / "SHA256SUMS"
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
        errors.append(f"{root.name}: artifacts/SHA256SUMS must name exactly one canonical ZIP")
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
    released = [
        directory
        for directory in directories
        if (directory / "artifacts" / "SHA256SUMS").is_file()
    ]
    staging = [directory for directory in directories if directory not in released]
    for directory in released:
        errors.extend(verify_experiment(directory))
    for directory in staging:
        print(f"staging experiment (no canonical artifact declared yet): {directory.name}")
    if errors:
        for error in errors:
            print(error)
        return 1
    print(
        f"archive verification passed for {len(released)} released experiment(s); "
        f"{len(staging)} staging"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
