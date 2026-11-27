"""Validate materialization without a hosted push or scientific execution."""
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from materialize_archive import CHECKSUM, GENERATED, METADATA, REPLAY, VERIFICATION, materialize
from verify_archive import sha256, verify_experiment


class MaterializationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.experiment = self.root / "experiments/sample-v2"
        for directory in ("artifacts", "source", "receipts"):
            (self.experiment / directory).mkdir(parents=True)
        (self.experiment / "README.md").write_text("# Synthetic staging fixture\n")
        (self.experiment / "source/requirements.txt").write_text("# No extra Python requirements\n")
        self.package = self.experiment / "artifacts/frozen.zip"
        self.members = {
            METADATA[0]: {"packageType": "compiled-experiment", "profile": "compiled-experiment-lifecycle-v1"},
            METADATA[1]: {},
            METADATA[2]: {"experimentId": self.experiment.name},
        }
        with ZipFile(self.package, "w") as archive:
            for name, data in self.members.items():
                archive.writestr(name, json.dumps(data))
            archive.writestr("evidence/scientific-decision.md", "Synthetic fixture")
            archive.writestr("evidence/publication-review.md", "Synthetic fixture review")
            archive.writestr(f"evidence/finalization/{'a' * 64}/parent-attempt-package.zip", b"stub")
        self.replay = self.experiment / REPLAY
        self.replay.write_text('{"status":"matched"}\n')
        self.request = self.root / "release-requests/sample-v2.json"
        self.request.parent.mkdir()
        self.write_request()

    def write_request(self, **changes):
        self.request.write_text(json.dumps({
            "schemaVersion": 1, "experimentId": self.experiment.name,
            "requestedAction": "publish-reviewed-compiled-experiment-release",
            "canonicalPackageSha256": sha256(self.package), "offlineReplayReceiptSha256": sha256(self.replay),
            **changes,
        }))

    def compile_receipt(self, args, **_):
        self.assertEqual(args[1:5], ["-P", "-m", "experiment_compiler", "verify"])
        package = Path(args[5])
        self.assertEqual(args[6:8], ["--expected-sha256", sha256(self.package)])
        self.assertEqual(args[8], "--receipt")
        Path(args[9]).write_text(json.dumps({
            "status": "integrity-verified", "packageSha256": sha256(package), "packageSizeBytes": package.stat().st_size,
        }))

    def stage(self):
        materialize(self.root, self.experiment.name, self.request.name)

    def test_stages_metadata_for_verified_package_and_preserves_replay(self):
        replay = self.replay.read_bytes()
        with patch("materialize_archive.subprocess.run", side_effect=self.compile_receipt):
            self.stage()
        self.assertEqual((self.experiment / CHECKSUM).read_text(), f"{sha256(self.package)}  frozen.zip\n")
        self.assertEqual(self.replay.read_bytes(), replay)
        self.assertTrue((self.experiment / VERIFICATION).is_file())
        self.assertEqual(verify_experiment(self.experiment), [])
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.stage()

    def test_rejects_wrong_request_and_digest_before_compiler(self):
        for changes in ({"experimentId": "other-v2"}, {"canonicalPackageSha256": "a" * 64},
                        {"offlineReplayReceiptSha256": "b" * 64}, {"schemaVersion": True}):
            with self.subTest(changes=changes), patch("materialize_archive.subprocess.run") as compiler:
                self.write_request(**changes)
                with self.assertRaises(ValueError):
                    self.stage()
                compiler.assert_not_called()
        self.assertFalse(any((self.experiment / path).exists() for path in GENERATED))

    def test_compiler_failure_or_nonmatching_replay_yields_no_outputs(self):
        with patch("materialize_archive.subprocess.run", side_effect=subprocess.CalledProcessError(1, "verify")):
            with self.assertRaises(subprocess.CalledProcessError):
                self.stage()
        self.replay.write_text('{"status":"failed"}\n')
        self.write_request()
        with patch("materialize_archive.subprocess.run", side_effect=self.compile_receipt):
            with self.assertRaisesRegex(ValueError, "offline replay receipt is not matched"):
                self.stage()
        self.assertFalse(any((self.experiment / path).exists() for path in GENERATED))

    def test_lifecycle_attempt_without_finalization_evidence_is_not_materialized(self):
        with ZipFile(self.package, "w") as archive:
            for name, data in self.members.items():
                archive.writestr(name, json.dumps(data))
        self.write_request()
        with patch("materialize_archive.subprocess.run", side_effect=self.compile_receipt):
            with self.assertRaisesRegex(ValueError, "must retain the selected attempt"):
                self.stage()
        self.assertFalse(any((self.experiment / path).exists() for path in GENERATED))


if __name__ == "__main__":
    unittest.main()
