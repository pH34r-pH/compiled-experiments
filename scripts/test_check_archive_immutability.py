"""Archive changes must preserve prior Git entries, including file modes."""
import unittest

from check_archive_immutability import changed_archive_files


RELEASE = "experiments/sample-v1/"
CHECKSUM = RELEASE + "artifacts/SHA256SUMS"
PACKAGE = RELEASE + "artifacts/package.zip"
SOURCE = RELEASE + "source/run.py"
REPLAY = RELEASE + "receipts/2026-11-01-replay.json"
README = RELEASE + "README.md"
LATER = RELEASE + "receipts/2026-11-27-clean-replay.json"
ORIGINAL = "100644 blob " + "1" * 40
UPDATED = "100644 blob " + "2" * 40


class ArchiveImmutabilityTests(unittest.TestCase):
    def setUp(self):
        self.base = dict.fromkeys((CHECKSUM, PACKAGE, SOURCE, REPLAY, README), ORIGINAL)

    def test_changed_package_and_checksum_are_blocked_together(self):
        candidate = self.base | {PACKAGE: UPDATED, CHECKSUM: UPDATED}
        self.assertEqual(changed_archive_files(self.base, candidate), [CHECKSUM, PACKAGE])

    def test_existing_source_and_receipts_are_protected(self):
        candidate = self.base | {SOURCE: ORIGINAL.replace("100644", "100755"), LATER: UPDATED}
        del candidate[REPLAY]
        self.assertEqual(changed_archive_files(self.base, candidate), [REPLAY, SOURCE])

    def test_dated_receipt_and_same_archive_readme_can_be_added(self):
        candidate = self.base | {LATER: UPDATED, README: UPDATED}
        self.assertEqual(changed_archive_files(self.base, candidate), [])
        self.assertEqual(changed_archive_files(self.base, self.base | {README: UPDATED}), [README])

    def test_other_archive_receipt_cannot_authorize_a_readme_update(self):
        second = {path.replace("sample-v1", "other-v1"): entry for path, entry in self.base.items()}
        base = self.base | second
        candidate = base | {LATER.replace("sample-v1", "other-v1"): UPDATED, README: UPDATED}
        self.assertEqual(changed_archive_files(base, candidate), [README])

    def test_added_receipt_must_be_dated_regular_file_in_receipts(self):
        for path, entry in (
            (LATER, UPDATED.replace("100644", "120000")),
            (LATER.replace("11-27", "02-30"), UPDATED),
            (LATER.replace("receipts/", "source/"), UPDATED),
            (RELEASE + "receipts/replay.json", UPDATED),
            (RELEASE + "receipts/nested/2026-11-27-replay.json", UPDATED),
        ):
            with self.subTest(path=path, entry=entry):
                self.assertEqual(changed_archive_files(self.base, self.base | {path: entry}), [path])

    def test_new_version_request_and_unmaterialized_candidate_can_change(self):
        pending = "experiments/sample-v2/README.md"
        base = self.base | {pending: ORIGINAL}
        additions = {pending: UPDATED, CHECKSUM.replace("sample-v1", "sample-v2"): UPDATED,
                     "release-requests/sample-v2.json": UPDATED}
        candidate = base | additions
        self.assertEqual(changed_archive_files(base, candidate), [])
        self.assertEqual(changed_archive_files(base, candidate | {SOURCE: UPDATED}), [SOURCE])


if __name__ == "__main__":
    unittest.main()
