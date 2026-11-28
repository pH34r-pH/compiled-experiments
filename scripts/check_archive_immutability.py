#!/usr/bin/env python3
"""Protect materialized experiment files against changes in the candidate Git tree."""
from __future__ import annotations

import argparse
from datetime import date
import re
import subprocess
import sys


CHECKSUM = re.compile(r"(experiments/[^/]+)/artifacts/SHA256SUMS")
DATED_RECEIPT = re.compile(r"receipts/([0-9]{4}-[0-9]{2}-[0-9]{2})-[a-z0-9]+(?:-[a-z0-9]+)*\.json")
Tree = dict[str, str]


def archive_tree(commit: str) -> Tree:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Base and candidate must be exact Git commit SHAs")
    output = subprocess.check_output(["git", "ls-tree", "-rz", f"{commit}^{{commit}}", "--", "experiments/"])
    entries = {}
    for record in output.decode("utf-8", errors="surrogateescape").split("\0"):
        if record:
            entry, path = record.split("\t", 1)
            entries[path] = entry
    return entries


def dated_receipt(relative: str) -> bool:
    match = DATED_RECEIPT.fullmatch(relative)
    if match is None:
        return False
    try:
        date.fromisoformat(match[1])
    except ValueError:
        return False
    return True


def regular_file(entry: str | None) -> bool:
    return bool(entry and entry.startswith("100644 blob "))


def changed_archive_files(base: Tree, candidate: Tree) -> list[str]:
    releases = {match[1] for path in base if (match := CHECKSUM.fullmatch(path))}
    additions = set()
    changes = []
    for path in sorted(base.keys() | candidate.keys()):
        parts = path.split("/", 2)
        if len(parts) != 3:
            continue
        parent, identifier, relative = parts
        release = f"{parent}/{identifier}"
        before, after = base.get(path), candidate.get(path)
        if release not in releases or before == after:
            continue
        if before is None and regular_file(after) and dated_receipt(relative):
            additions.add(release)
        else:
            changes.append((path, release, relative, before, after))
    return [path for path, release, relative, before, after in changes if not (
        release in additions and relative == "README.md" and regular_file(before) and regular_file(after)
    )]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("candidate")
    args = parser.parse_args()
    try:
        changed = changed_archive_files(archive_tree(args.base), archive_tree(args.candidate))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Cannot verify archive immutability: {exc}", file=sys.stderr)
        return 1
    for path in changed:
        print(f"Materialized experiment file changed: {path!r}", file=sys.stderr)
    if changed:
        return 1
    print("Materialized experiment files are intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
