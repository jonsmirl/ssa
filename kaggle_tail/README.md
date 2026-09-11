# RTX 6000 full-scale tail evaluation

**Measured result:** version 2's 0.25 influence cap improves full-test sparse perplexity through 32K
(154.77 versus 183.01 at 32K), but strict retrieval is 2/9 plus one tie, versus sparse 4/9 and dense 7/9.
It completed in 1750.94 s on the RTX PRO 6000 Blackwell (94.97 GiB GPU, 176.88 GiB visible host RAM).
Version 1's uncapped tail worsens 8K/32K perplexity and gets 1/9 retrieval; both runs remain archived.
See `runs/kaggle_tail_v2/ssa_tail_fullscale.json`, `comparison.json`, and `RESULTS.md`.
No new training was performed and no 10M tail test is claimed.

Private notebook `jonsmirl/ssa-tail-fullscale-rtx6000`, separate from the previous 10M notebook.
ARC3 is attached to request the RTX Pro 6000; internet is OFF. The existing private wheel dataset supplies
Transformers 5.12.1 and FAISS 1.14.1, installed with `--no-index --no-deps`. FAISS GPU search is not used.

Protocol (no new training or tuning): all 298,938 official WikiText-2 test tokens, nonoverlapping windows
at 512, 4096, 8192, and 32768; dense, sparse-only, and saved-gain tail modes. Include partial final windows
and aggregate by target count, not by window count. Context-boundary transitions are not scored.
Nine fixed semantic NIAH probes use lengths 8192, 32768, and 131072 and depths 0.1, 0.5, and 0.9.
Positions are not rescaled; this is a length-transfer test of gains trained at 512, not long-context training.

The router batches independent causal prefix forests using the same center/radius inequality as the
existing tree. It does not pool future queries or inspect a future-bearing node. Fixed-beam search remains
approximate. Counts/value sums advance in chunks; the full KV archive still grows with context. Vocabulary
projection is also chunked, avoiding a sequence-length-by-vocabulary allocation.

```bash
python kaggle_tail/build_notebook.py
kaggle kernels push -p kaggle_tail -t 10800
kaggle kernels status jonsmirl/ssa-tail-fullscale-rtx6000
python kaggle_tail/watch.py
kaggle kernels output jonsmirl/ssa-tail-fullscale-rtx6000 -p runs/kaggle_tail_v1
```

Version 2 freezes the validation-selected prototype tail with a maximum mixture share of 0.25:

```bash
python -m ssa.tail_revision_experiment
python kaggle_tail/build_notebook.py --tail-mode prototype --gain-mode saved \
  --max-tail-share 0.25 --selection runs/tail_revision_selection.json
kaggle kernels push -p kaggle_tail -t 10800
python kaggle_tail/watch.py
kaggle kernels output jonsmirl/ssa-tail-fullscale-rtx6000 -p runs/kaggle_tail_v2
```

The selection artifact records the validation hash, criterion, numerical implementation revision,
and reused-test limitation. It is included in the private payload and must match the requested flags.
Neither the influence limit nor the optional actual-mean Jensen estimator is a deterministic output
certificate. The Jensen estimator supplies a mass **lower** bound, not a certified omitted-mass upper bound.

All fourteen v2 manifest hashes verify; inputs/gains and every dense/sparse baseline result reproduce
v1. Recheck with `python runs/kaggle_tail_v2/audit.py`. Strict wins are counted separately because
the 8K/depth0.5 capped output ties `walnut` and `lantern` at 4.15625; candidate-order correctness is
not a strict retrieval success. Both the full corpus and probes were inspected in v1, so v2 is a
frozen regression test, not a new holdout.

The builder requires cached Qwen2.5-0.5B tokenizer files and the cached public WikiText-2 test Arrow file.
It embeds sources, tokenized test data, frozen gains, and a SHA256 manifest. Generated notebooks and token
bundles are local/private run inputs; rebuild them with the command above. Output `ssa_tail_fullscale.json`
is atomically checkpointed after every window/probe and records failures. The local one-window GPU
preflight is `runs/kaggle_tail_preflight.json`; it is not a remote full-corpus result.
