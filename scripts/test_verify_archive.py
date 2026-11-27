"""Exercise the archive checks without running the packaged experiment."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from verify_archive import EXPERIMENTS, verify_experiment
from verify_radix_scope import RECORD, digest


class RadixScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        released = EXPERIMENTS / "dsl-issue-152-heldout-radix-replay-v1"
        self.root = Path(temporary.name) / released.name
        shutil.copytree(released, self.root)
        self.scope_path = self.root / RECORD
        self.scope = json.loads(self.scope_path.read_text(encoding="utf-8"))

    def assert_scope_rejected(self, reason: str) -> None:
        self.scope_path.write_text(json.dumps(self.scope), encoding="utf-8")
        self.assertIn(reason, "\n".join(verify_experiment(self.root)))

    def test_released_scope_verifies(self):
        self.assertEqual(verify_experiment(self.root), [])

    def test_scope_requires_canonical_artifact(self):
        self.scope["canonicalPackageSha256"] = "a" * 64
        self.assert_scope_rejected("does not identify this archive")

    def test_offline_receipt_requires_the_original_trace_manifest(self):
        path = self.root / "receipts/offline-replay-receipt.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["traceManifestSha256"] = "a" * 64
        contents = json.dumps(receipt)
        path.write_text(contents, encoding="utf-8")
        self.scope["evidenceSha256"]["archive-replay"] = digest(contents.encode("utf-8"))
        self.assert_scope_rejected("does not match the replay and producer records")

    def test_missing_stage_fails(self):
        del self.scope["stages"]["retained-trace-analysis"]
        self.assert_scope_rejected("all five reproduction stages must be present")

    def test_screening_cannot_claim_independent_replay(self):
        self.scope["stages"]["screening-selection"]["result"] = "independently-replayed"
        self.assert_scope_rejected("unsupported result, performer or evidence for screening-selection")

    def test_digest_only_screening_input_cannot_claim_available_bytes(self):
        self.scope["unavailableInputs"]["screening-product-csv"]["availability"] = "embedded"
        self.assert_scope_rejected("checksum-only inputs must remain explicitly unavailable")

    def test_fresh_measurement_cannot_claim_independent_collection(self):
        self.scope["stages"]["fresh-measurement"]["performedBy"] = "archive-replay-workflow"
        self.assert_scope_rejected("unsupported result, performer or evidence for fresh-measurement")

    def test_workflow_failure_cannot_be_presented_as_success(self):
        self.scope["archiveReplayWorkflow"]["workflowConclusion"] = "success"
        self.assert_scope_rejected("workflow identities, environment or replay status differ")

    def test_private_producer_run_details_are_excluded(self):
        self.scope["producerWorkflow"]["runId"] = 123
        self.assert_scope_rejected("workflow identities, environment or replay status differ")

    def test_numerical_differences_remain_visible_when_decision_matches(self):
        self.scope["stages"]["numerical-agreement"]["reportedDifferences"] = []
        self.assert_scope_rejected("numerical differences must remain visible")

    def test_numerical_agreement_cannot_be_inferred_from_decision_match(self):
        self.scope["stages"]["numerical-agreement"]["result"] = "matched"
        self.assert_scope_rejected("unsupported result, performer or evidence for numerical-agreement")

    def test_readable_summary_mirror_cannot_disagree_with_release(self):
        (self.root / "receipts/fresh-reproduction-summary.json").write_text("{}", encoding="utf-8")
        self.assert_scope_rejected("archival mirror differs from the canonical ZIP")


if __name__ == "__main__":
    unittest.main()
