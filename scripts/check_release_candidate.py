#!/usr/bin/env python3
"""Verify a reviewed archive checkout and upstream signed build provenance."""

import argparse
import re
import subprocess

import verify_archive

COMPILER_REPO = "pH34r-pH/experiment-compiler"
SIGNER = f"{COMPILER_REPO}/.github/workflows/lifecycle.yml"


def check_candidate(experiment_id: str, sha256: str, archive_commit: str, compiler_commit: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_id):
        raise ValueError("Expected one archive experiment directory name")
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("Expected a full lowercase package SHA-256")
    if not all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (archive_commit, compiler_commit)):
        raise ValueError("Expected full lowercase archive and Compiler commit SHAs")

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=verify_archive.ROOT, text=True).strip()
    if head != archive_commit:
        raise ValueError("Archive checkout differs from the reviewed commit")
    if verify_archive.main() != 0:
        raise ValueError("Archive verification failed")

    directory = verify_archive.EXPERIMENTS / experiment_id
    checksum = directory / "artifacts" / "SHA256SUMS"
    if directory.is_symlink() or not checksum.is_file():
        raise ValueError("Experiment has no canonical archived package")
    digest, filename = checksum.read_text(encoding="utf-8").strip().split(maxsplit=1)
    if digest != sha256:
        raise ValueError("Canonical package SHA-256 differs from the reviewed digest")

    # Archive verification has checked this checksum, filename, and the exact ZIP bytes.
    package = checksum.parent / filename.lstrip("*")
    subprocess.run(
        [
            "gh", "attestation", "verify", str(package),
            "--repo", COMPILER_REPO,
            "--signer-workflow", SIGNER,
            "--source-ref", "refs/heads/main",
            "--source-digest", compiler_commit,
            "--signer-digest", compiler_commit,
            "--deny-self-hosted-runners",
            "--predicate-type", "https://slsa.dev/provenance/v1",
        ],
        check=True,
        timeout=120,
    )
    if verify_archive.sha256(package) != sha256:
        raise ValueError("Canonical package changed during provenance verification")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--archive-commit", required=True)
    parser.add_argument("--compiler-commit", required=True)
    args = parser.parse_args()
    try:
        check_candidate(args.experiment_id, args.sha256, args.archive_commit, args.compiler_commit)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Release candidate verification failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
