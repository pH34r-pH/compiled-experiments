# Release process

This repository is the public archival surface for Compiled Experiments.

For each publication release:

1. Treat the experiment directory as immutable once released.
2. Verify the canonical compiled ZIP against its recorded SHA-256.
3. Verify Compiler integrity receipts and the scientific replay receipt.
4. Update root `.zenodo.json` and `CITATION.cff` for the release being archived.
5. Dispatch the owner-only release workflow with the reviewed archive and builder identities.
6. Let the connected Zenodo GitHub integration archive that release.
7. Record the Zenodo DOI and record URL back in the associated manuscript/publication metadata.

GitHub `main` is a staging surface. The archival identity is the reviewed GitHub Release plus its Zenodo DOI.

## Human publication boundary

Archival promotion is intentionally not an automatic continuation of Fleet execution or Experiment Compiler verification.

Before creating the GitHub Release, confirm that:

- the source owner approved the public disclosure/finalization;
- the artifact bytes are exactly the reviewed finalized package rather than a rebuild in this repository;
- package SHA-256, receipts, RO-Crate/dependency closure, release metadata, and redistribution/licensing agree;
- private runner paths, credentials, non-public evidence, and operational metadata have not crossed the disclosure boundary.

Creating the reviewed GitHub Release is the handoff to the owner-controlled Zenodo integration. This repository does not attempt to mint a DOI itself. After Zenodo archives the release, record the immutable DOI/record identity back in the associated manuscript and authoritative publication metadata; derived public surfaces may then display that relation.

### Check and publish a release candidate

Use the exact package SHA-256 and review the workflow and commit that built or
finalized it with Experiment Compiler. Use the following command in a clean
checkout of the reviewed archive commit.

```bash
python scripts/check_release_candidate.py \
  --experiment-id="$EXPERIMENT_ID" --sha256="$PACKAGE_SHA256" \
  --archive-commit="$ARCHIVE_COMMIT" --builder-workflow="$BUILDER_WORKFLOW" \
  --builder-commit="$BUILDER_COMMIT"
```

The check verifies the archive and asks GitHub CLI to verify a SLSA provenance
statement for the exact ZIP. It requires a reviewed `pH34r-pH` workflow on
`main`, the reviewed commit as both source and signer, and a GitHub-hosted
runner. The archive's replay records carry scientific evidence. If signed
provenance is unavailable, keep the candidate staged.

After reviewing the disclosure and the preflight result, the repository owner
runs **publish release** on `main` with the tag, experiment ID, ZIP digest,
archive commit, builder workflow, and builder commit. The workflow checks the
dispatched commit and signature again before obtaining write access. It pushes the tag without
force and creates a release only for an existing tag. A retry may reuse a tag
at the same commit if publication failed after tagging. If `main` moved, review
the new revision before dispatching again.

The existing #152 releases and the original release request remain historical
records.

## Versioning

Experiment directories use stable IDs. A scientific revision that changes evidence, protocol, code, or the canonical artifact must use a new release version and must not silently replace a previously archived artifact.

The first release is:

- experiment: `dsl-issue-152-heldout-radix-replay-v1`
- release version: `1.0.0`
- suggested tag: `dsl-issue-152-v1.0.0`

## Later verification records

Record a later verification as a dated entry beside the original receipts. Identify the released package and protocol by digest, the verification command and environment, the performer and scope of independence, and the input and output receipts. Record missing inputs and failed or incomplete stages explicitly. Provide the applicable comparison rule and record decision agreement and numerical agreement separately.

Preserve the original tag, ZIP, checksum, and historical receipts. A dated verification can link to earlier runs when their scope is known. Archive CI checks these records and their references using the retained files.
