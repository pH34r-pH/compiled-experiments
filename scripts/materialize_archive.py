#!/usr/bin/env python3
"""Stage metadata for an exact reviewed package using its retained replay receipt."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

from verify_archive import ROOT, load_json, sha256, verify_experiment


NAME = r"[a-z0-9][a-z0-9._-]*"
DIGEST = re.compile(r"[0-9a-f]{64}")
METADATA = ("experiment-package-manifest.json", "ro-crate-metadata.json", "dependency-closure.json")
CHECKSUM = "artifacts/SHA256SUMS"
VERIFICATION = "receipts/experiment-compiler-verify-receipt.json"
REPLAY = "receipts/offline-replay-receipt.json"
GENERATED = (*METADATA, CHECKSUM, VERIFICATION)
FINALIZATION = re.compile(r"evidence/finalization/[0-9a-f]{64}/parent-attempt-package\.zip")


def local_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file() or any(item.is_symlink() for item in (path, *path.parents) if item.is_relative_to(root)):
        raise ValueError(f"Expected an ordinary staged file at {relative}")
    return path


def reviewed_inputs(root: Path, experiment_id: str, request_name: str) -> tuple[Path, Path, str, str]:
    if not re.fullmatch(NAME, experiment_id) or not re.fullmatch(rf"{NAME}\.json", request_name):
        raise ValueError("Expected an experiment ID and a release request JSON filename")
    request = load_json(local_file(root, f"release-requests/{request_name}"))
    if (type(request.get("schemaVersion")) is not int or request["schemaVersion"] != 1
            or request.get("experimentId") != experiment_id
            or request.get("requestedAction") != "publish-reviewed-compiled-experiment-release"):
        raise ValueError("Release request does not identify the selected experiment")
    experiment = root / "experiments" / experiment_id
    for member in GENERATED:
        output = experiment / member
        if output.exists() or output.is_symlink():
            raise ValueError(f"Generated archive file already exists at {member}")
    package_digest, replay_digest = (request.get(key) for key in ("canonicalPackageSha256", "offlineReplayReceiptSha256"))
    if not all(isinstance(value, str) and DIGEST.fullmatch(value) for value in (package_digest, replay_digest)):
        raise ValueError("Release request must pin the exact package and replay receipt")
    artifacts = experiment / "artifacts"
    if artifacts.is_symlink() or experiment.is_symlink() or experiment.parent.is_symlink():
        raise ValueError("Expected an ordinary staged experiment")
    packages = list(artifacts.glob("*.zip"))
    if len(packages) != 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.zip", packages[0].name):
        raise ValueError("Expected exactly one staged ZIP")
    package = local_file(experiment, f"artifacts/{packages[0].name}")
    if sha256(package) != package_digest or sha256(local_file(experiment, REPLAY)) != replay_digest:
        raise ValueError("Staged package or replay digest differs from the reviewed release request")
    return experiment, package, package_digest, replay_digest


def prepare_preview(experiment: Path, package: Path, digest: str, replay_digest: str, preview: Path) -> None:
    package_member = f"artifacts/{package.name}"
    for member in ("README.md", "source/requirements.txt", REPLAY, package_member):
        target = preview / member
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_file(experiment, member), target)
    if sha256(preview / REPLAY) != replay_digest:
        raise ValueError("Replay changed while preparing the preview")
    staged_package = preview / package_member
    subprocess.run(
        [sys.executable, "-P", "-m", "experiment_compiler", "verify", str(staged_package),
         "--expected-sha256", digest, "--receipt", str(preview / VERIFICATION)],
        check=True, timeout=300,
    )
    with ZipFile(staged_package) as archive:
        for member in METADATA:
            (preview / member).write_bytes(archive.read(member))
        manifest = load_json(preview / METADATA[0])
        if manifest.get("profile") == "compiled-experiment-lifecycle-v1":
            names = set(archive.namelist())
            records = {"evidence/scientific-decision.md", "evidence/publication-review.md"}
            if not records <= names or not any(FINALIZATION.fullmatch(name) for name in names):
                raise ValueError("Lifecycle archive must retain the selected attempt, decision, and review")
    (preview / CHECKSUM).write_text(f"{digest}  {package.name}\n", encoding="utf-8", newline="\n")
    if errors := verify_experiment(preview):
        raise ValueError("\n".join(errors))


def materialize(root: Path, experiment_id: str, request_name: str) -> None:
    experiment, package, digest, replay_digest = reviewed_inputs(root, experiment_id, request_name)
    with TemporaryDirectory(prefix="reviewed-archive-") as temporary:
        preview = Path(temporary) / experiment_id
        prepare_preview(experiment, package, digest, replay_digest, preview)
        for member in GENERATED:
            with (experiment / member).open("xb") as destination:
                destination.write((preview / member).read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("release_request")
    args = parser.parse_args()
    try:
        materialize(ROOT, args.experiment_id, args.release_request)
    except (OSError, ValueError, KeyError, BadZipFile, subprocess.SubprocessError) as error:
        parser.exit(1, f"Archive materialization failed: {error}\n")
    print("Reviewed archive metadata staged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
