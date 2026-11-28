"""Regression tests for the archive publication preflight."""

import subprocess
import unittest
from unittest.mock import patch

import check_release_candidate as release


class ReleaseCandidateTest(unittest.TestCase):
    experiment = "dsl-issue-152-heldout-radix-replay-v1"
    digest = "25612d81a480dd5f8148a0dc247db58cb2fd719fe619e75ef0cd4945b3c1364b"
    archive_commit = "a" * 40
    builder_workflow = "pH34r-pH/experiment-compiler/.github/workflows/lifecycle.yml"
    builder_commit = "b" * 40

    def setUp(self):
        self.enterContext(patch.object(release.subprocess, "check_output", return_value=self.archive_commit + "\n"))
        self.attestation = self.enterContext(patch.object(release.subprocess, "run"))

    def check(self, **changes):
        values = {
            "experiment_id": self.experiment,
            "sha256": self.digest,
            "archive_commit": self.archive_commit,
            "builder_workflow": self.builder_workflow,
            "builder_commit": self.builder_commit,
        }
        release.check_candidate(**(values | changes))

    def test_verifies_exact_archive_and_pins_upstream_signer(self):
        self.check()
        package = release.verify_archive.EXPERIMENTS / self.experiment / "artifacts" / "issue152-replay.zip"
        self.attestation.assert_called_once_with(
            [
                "gh", "attestation", "verify", str(package),
                "--repo", "pH34r-pH/experiment-compiler",
                "--signer-workflow", "pH34r-pH/experiment-compiler/.github/workflows/lifecycle.yml",
                "--source-ref", "refs/heads/main",
                "--source-digest", self.builder_commit,
                "--signer-digest", self.builder_commit,
                "--deny-self-hosted-runners",
                "--hostname", "github.com",
                "--predicate-type", "https://slsa.dev/provenance/v1",
            ],
            check=True,
            timeout=120,
        )

    def test_wrong_archive_commit_stops_before_signature_check(self):
        with self.assertRaisesRegex(ValueError, "differs from the reviewed commit"):
            self.check(archive_commit="c" * 40)
        self.attestation.assert_not_called()

    def test_wrong_package_digest_stops_before_signature_check(self):
        with self.assertRaisesRegex(ValueError, "differs from the reviewed digest"):
            self.check(sha256="0" * 64)
        self.attestation.assert_not_called()

    def test_unavailable_or_invalid_provenance_blocks_release(self):
        self.attestation.side_effect = subprocess.CalledProcessError(1, ["gh", "attestation", "verify"])
        with self.assertRaises(subprocess.CalledProcessError):
            self.check()

    def test_source_repository_can_sign_its_compiled_package(self):
        signer = "pH34r-pH/domain-scaling-lab/.github/workflows/example.yml"
        self.check(builder_workflow=signer)
        command = self.attestation.call_args.args[0]
        self.assertEqual(command[command.index("--repo") + 1], "pH34r-pH/domain-scaling-lab")
        self.assertEqual(command[command.index("--signer-workflow") + 1], signer)

    def test_builder_outside_archive_owner_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reviewed builder workflow"):
            self.check(builder_workflow="other/experiment-compiler/.github/workflows/lifecycle.yml")
        self.attestation.assert_not_called()
