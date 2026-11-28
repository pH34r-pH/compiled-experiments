# Experiment archive layout

Each directory under this folder is a stable public experiment identity.

A released experiment should contain:

- `README.md`: scientific result, replay instructions, and canonical artifact identity;
- `experiment-package-manifest.json`: canonical member inventory embedded in the compiled ZIP;
- `ro-crate-metadata.json`: RO-Crate 1.3 metadata;
- `dependency-closure.json`: embedded/external/runtime dependency classification;
- `source/`: the small human-readable replay entrypoint and provenance files useful for inspection;
- `receipts/`: Compiler integrity and scientific reproduction receipts;
- `artifacts/`: the canonical compiled ZIP and checksum inventory.

The compiled ZIP and its embedded package manifest are authoritative for the complete payload, including binary trace shards and large compressed evidence. The repository intentionally does not duplicate large decompressed evidence tables outside that ZIP.

Released experiment directories are append-only. Corrections that change the artifact bytes use a new release version and retain previous archival releases through GitHub/Zenodo history.

Archive CI protects files in any experiment directory that already has a
canonical `artifacts/SHA256SUMS` on the base revision. Prepare a scientific
revision in a new versioned directory. For a later verification, add a
receipt named `receipts/YYYY-MM-DD-description.json` and update the same
experiment's `README.md` to describe it. Keep the existing receipts,
source snapshot, ZIP, and checksum intact.
