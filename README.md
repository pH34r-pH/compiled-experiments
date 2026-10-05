# Compiled Experiments

Public, versioned reproducibility artifacts produced by [Experiment Compiler](https://github.com/pH34r-pH/experiment-compiler).

This repository is the archival publication surface for reviewed Compiled Experiments. Source research may live elsewhere; releases here contain the portable, content-addressed material needed to inspect and replay the published result.

## Layout

```text
experiments/
  <experiment-id>/
    README.md
    experiment.json
    ro-crate-metadata.json
    dependency-closure.json
    artifact/
      <compiled-experiment>.zip
      SHA256SUMS
    receipts/
      ...
    source/
      ...
```

The canonical scientific artifact is the compiled ZIP named in each experiment README. Human-readable source/metadata are retained beside it for inspection. Large third-party model or dataset dependencies stay at their canonical immutable upstream locations when redistribution is unnecessary; the experiment records their exact identities.

## Release policy

Each GitHub Release represents one reviewed archival publication event.

1. Add or update the experiment under `experiments/<experiment-id>/`.
2. Update root `.zenodo.json` and `CITATION.cff` for that release.
3. Verify every declared digest and replay receipt.
4. Tag the reviewed commit and create a GitHub Release.
5. Zenodo archives that release and mints the DOI.
6. Record the DOI back in the associated manuscript/artifact metadata.

A release is not considered publication-ready merely because files are present on `main`; the content-addressed artifact, receipts, and release metadata must all agree.

## First artifact

The first publication candidate is:

[`dsl-issue-152-heldout-radix-replay-v1`](experiments/dsl-issue-152-heldout-radix-replay-v1/)

Held-out validation of product-address-aware radix selection for closed logarithmic transformer arithmetic. The reproduced disposition is `kill` / unsupported, with zero of four frozen pairs validated.

## License

Repository-authored code and metadata are Apache-2.0 unless a nested artifact states otherwise. Third-party upstream licenses are preserved in each experiment's provenance.
