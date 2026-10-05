# DSL #152 — held-out radix-selection offline replay

**Compiled Experiment ID:** `dsl-issue-152-heldout-radix-replay-v1`

This is the public archival replay for the paper *Held-Out Validation of Product-Address-Aware Radix Selection for Closed Logarithmic Transformer Arithmetic*.

## Result

The preregistered held-out study tested whether empirical product-address behavior provides a useful radix-selection signal beyond numerical quality and CurveFP's theoretical phase count.

The reproduced disposition is:

- decision: `kill` / unsupported;
- frozen pairs validated: **0 of 4**;
- required to validate: **3 of 4**.

The public replay does not claim physical PPA results. Routing/write/tag quantities are explicit engineering proxies.

## Provenance

The corrected archival build is generated from Domain Scaling Lab source commit:

`26e3e40f8214c9b81740d9a315d4bf19cc9a5a47`

through the reviewed `issue-152 reproduction` workflow. The run performs a clean full-from-model capture, decision-stable comparison with the frozen historical result, Compiled Experiment assembly, Compiler integrity verification, and an offline replay from the retained operand traces.

Upstream model assets are not redistributed:

- `prajjwal1/bert-small@0ec5f86f27c1a77d704439db5e01c307ea11b9d4`;
- `google/bert_uncased_L-2_H-128_A-2@30b0a37ccaaa32f332884b96992754e246e48c5f`.

Those assets are needed only for full recapture. The archived replay is self-contained.

## Canonical artifact

The authoritative package is:

`artifact/issue152-replay.zip`

Its exact SHA-256 is recorded in `artifact/SHA256SUMS` and must match the Experiment Compiler verification receipt under `receipts/`.

The compiled ZIP contains the complete replay payload, including all eight retained operand-trace shards and compressed reference evidence. This Git directory intentionally exposes only the small human-readable source/provenance alongside the canonical ZIP rather than duplicating large decompressed evidence tables.

## Offline replay

1. Verify `artifact/SHA256SUMS`.
2. Extract `artifact/issue152-replay.zip`.
3. Create Python 3.12 with the NumPy version named by the extracted `experiment/requirements.txt`.
4. From the extracted `experiment/` directory run:

   ```sh
   python reproduce.py --output-dir reproduction
   ```

5. `reproduction/reproduction-receipt.json` must report:
   - `status: matched`;
   - `decision: kill`;
   - `validatedPairCount: 0`.

## Public release

Release version: **1.0.0**

Suggested Git tag: **`dsl-issue-152-v1.0.0`**

The GitHub Release is the trigger for the connected Zenodo archival record. Once Zenodo mints the DOI, that DOI is recorded back in the manuscript and publication metadata.
