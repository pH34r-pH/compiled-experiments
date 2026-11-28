"""Exercise the independent decision check using the released reference data."""
import copy
import json
import re
import unittest
from pathlib import Path
from zipfile import ZipFile

from verify_radix_decision import verify_decision

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "experiments/dsl-issue-152-heldout-radix-replay-v1/artifacts/issue152-replay.zip"


class DecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with ZipFile(ARCHIVE) as package:
            cls.reference = json.loads(package.read("evidence/reference/summary.json"))

    def test_identical_reference_has_no_differences(self):
        self.assertEqual(verify_decision(self.reference, self.reference), [])

    def test_retained_difference_is_explicit(self):
        candidate = copy.deepcopy(self.reference)
        row = candidate["pair_decisions"][0]
        row["dot_nmse_symmetric_geomean_ratio"] *= 1.01
        self.assertEqual(len(verify_decision(self.reference, candidate)), 1)

    def test_recorded_four_metric_differences_recompute(self):
        path = ROOT / "experiments/dsl-issue-152-heldout-radix-replay-v1/receipts/2026-11-27-clean-replay.json"
        recorded = json.loads(path.read_text(encoding="utf-8"))["comparison"]["toleratedDifferences"]
        candidate = copy.deepcopy(self.reference)
        rows = {row["pair_id"]: row for row in candidate["pair_decisions"]}
        for item in recorded:
            match = re.fullmatch(r"([^.]+)\.([a-z0-9_]+): expected ([^,]+), got ([^;]+); .*", item)
            self.assertIsNotNone(match)
            pair, metric, _, actual = match.groups()
            rows[pair][metric] = float(actual)
        self.assertEqual(verify_decision(self.reference, candidate), recorded)

    def test_crossing_boundary_is_rejected(self):
        candidate = copy.deepcopy(self.reference)
        candidate["pair_decisions"][0]["dot_nmse_symmetric_geomean_ratio"] = 1.0
        with self.assertRaises(ValueError):
            verify_decision(self.reference, candidate)

    def test_capture_and_pair_identity_cannot_be_substituted(self):
        candidate = copy.deepcopy(self.reference)
        candidate["capture_status"][0]["prompt_sha256"] = "a" * 64
        with self.assertRaises(ValueError):
            verify_decision(self.reference, candidate)
        candidate = copy.deepcopy(self.reference)
        candidate["pair_decisions"].append(candidate["pair_decisions"][0])
        with self.assertRaises(ValueError):
            verify_decision(self.reference, candidate)


if __name__ == "__main__":
    unittest.main()
