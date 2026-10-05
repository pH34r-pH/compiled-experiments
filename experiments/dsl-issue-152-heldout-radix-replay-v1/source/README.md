# #152 offline evidence replay

This is the portable replay entrypoint for the held-out #152 result.

The final Compiled Experiment embeds:

- the exact frozen preregistration;
- eight retained held-out operand-trace shards with SHA-256 checksums;
- the small Domain Scaling Lab numerical/analysis modules required by #152;
- the fresh reproduction's complete reference analysis and comparison receipt;
- the exact upstream model/tokenizer identities used to create the traces.

The pretrained BERT model is **not required for this replay**. It is needed only
to regenerate the operand traces from the full-from-model reproduction path.

Create a Python 3.12 environment, install `requirements.txt`, then run:

```sh
python reproduce.py --output-dir reproduction
```

The command copies the retained trace shards into a clean output directory,
recomputes the frozen numerical/routing analysis from those operands, compares
the new summary against the retained fresh-reproduction summary, and exits
nonzero if the scientific result differs.
