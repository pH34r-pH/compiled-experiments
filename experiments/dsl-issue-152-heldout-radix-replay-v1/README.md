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

The corrected archival build is generated from Domain Scaling Lab commit `26e3e40f8214c9b81740d9a315d4bf19cc9a5a47`. The source reproduction job completed the full model capture, comparison with the frozen historical result, package assembly, and retained-trace replay. The [public reproduction summary](receipts/fresh-reproduction-summary.json) records the capture from that restricted run.

A [separate archive job](https://github.com/pH34r-pH/compiled-experiments/actions/runs/37379212103/job/111996928603) verified the same ZIP with the pinned Compiler and replayed it in a fresh Ubuntu environment with Python 3.12 and NumPy 2.5.2. That job's replay and archive checks succeeded. Its final branch write failed.

Upstream model assets are not redistributed:

- `prajjwal1/bert-small@0ec5f86f27c1a77d704439db5e01c307ea11b9d4`;
- `google/bert_uncased_L-2_H-128_A-2@30b0a37ccaaa32f332884b96992754e246e48c5f`.

Those assets are needed only for full recapture. The archived replay is self-contained.

## Canonical artifact

The authoritative package is:

`artifacts/issue152-replay.zip`

Its exact SHA-256 is recorded in `artifacts/SHA256SUMS` and must match the Experiment Compiler verification receipt under `receipts/`.

The compiled ZIP contains the complete replay payload, including all eight retained operand-trace shards and compressed reference evidence. This Git directory intentionally exposes only the small human-readable source/provenance alongside the canonical ZIP rather than duplicating large decompressed evidence tables.

## Reproduction scope

The [dated scope record](receipts/reproduction-scope-v1.json) binds the five stages to the original ZIP and the retained receipts.

| Stage | Evidence |
| --- | --- |
| Screening and selection | The frozen protocol lists the selection rule, four pairs, and two screening CSV checksums. The CSV bytes are unavailable in the released ZIP and the pinned producer Git tree, so the historical selection cannot be reconstructed from those sources. |
| Fresh measurement | The source reproduction job captured eight held-out traces using the pinned model and tokenizer. The release retains its summary, capture identities, and environment. |
| Analysis on retained traces | The archive job executed `reproduce.py` from the exact ZIP in a fresh runner. It reused the producer's traces and analysis code. Its [receipt](receipts/offline-replay-receipt.json) preserves the result and output checksums. The replay summary itself is represented by its checksum. |
| Decision agreement | The [source comparison](receipts/decision-stable-comparison.json) matched the frozen decision under the prespecified thresholds and margin rule. The archive replay also returned `kill`, with zero of four pairs validated and three required. |
| Numerical agreement | Unassessed. The source comparison reports nine differences that preserved the decision, including a routing win count of 519 versus 518 for one pair. The study supplied a decision rule and left numerical agreement without a separate criterion. |

The [November 27 clean replay record](receipts/2026-11-27-clean-replay.json) identifies the exact release ZIP, Compiler revision, runtime image, NumPy wheel, and verification command. The replay used the producer's frozen analysis code and retained traces in a fresh, network-disabled environment. The comparator recorded four tolerated differences and confirmed the frozen disposition of zero validated pairs. The replay summary has a different checksum from the earlier archive replay. Numerical agreement remains unassessed.

Two synthetic comparator checks changed only the first pair's operand error ratio. Changing 1.8832078664 to 18.8321 preserved the decision comparison and recorded the numerical discrepancy. Changing it to 1.10 crossed the 1.15 threshold and failed the comparison. Both are checks of the comparison rule using a copy of the summary.

## Offline replay

1. Verify `artifacts/SHA256SUMS`.
2. Extract `artifacts/issue152-replay.zip`.
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
