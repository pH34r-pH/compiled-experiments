# Release process

This repository is the public archival surface for Compiled Experiments.

For each publication release:

1. Treat the experiment directory as immutable once released.
2. Verify the canonical compiled ZIP against its recorded SHA-256.
3. Verify Compiler integrity receipts and the scientific replay receipt.
4. Update root `.zenodo.json` and `CITATION.cff` for the release being archived.
5. Tag the reviewed commit and create a GitHub Release.
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

## Versioning

Experiment directories use stable IDs. A scientific revision that changes evidence, protocol, code, or the canonical artifact must use a new release version and must not silently replace a previously archived artifact.

The first release is:

- experiment: `dsl-issue-152-heldout-radix-replay-v1`
- release version: `1.0.0`
- suggested tag: `dsl-issue-152-v1.0.0`
