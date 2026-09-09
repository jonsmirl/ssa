# Kaggle >10M complete-transformer run

This notebook uses the ARC3-attached RTX Pro 6000 allocation, the official offline Qwen2.5-0.5B
model attachment, and the repository's streamed CCC/FlexAttention implementation.  It writes
kill-surviving telemetry to `ssa_10m_result.json` after every transformer layer.
Internet is disabled as required by the ARC3 accelerator; pinned public dependency wheels are mounted
from the private `jonsmirl/ssa-10m-offline-wheels` dataset and installed with `--no-index`.

```bash
python kaggle_10m/build_notebook.py
kaggle kernels push -p kaggle_10m
kaggle kernels status jonsmirl/ssa-complete-transformer-10m
kaggle kernels output jonsmirl/ssa-complete-transformer-10m -p runs/kaggle_10m
```

The main length is 10,000,128 tokens (strictly over 10M and block-aligned). The run first performs a
4K dense-equivalence gate and a 128K dense/streamed quality reference. Static YaRN extends positions;
the 10M row is capacity/mechanism evidence, not a claim of in-distribution language-model quality.

Measured version 5: all 24 layers completed in 713.185 s at 25.721 GB peak allocation and 14,021.8 token/s;
the analytic selected-fraction upper bound was 0.005062. Dense and streamed 128K semantic NIAH passed, and
the 10M semantic-candidate ranking passed (`walnut` 5.84375 versus best distractor 4.4375). Pre-RoPE content
Q/K drive the pure-PyTorch center-radius router, while post-RoPE Q/K still compute attention. A fixed 128-block
layer-1 head-consensus reservoir preserves high-confidence candidates through the stack. Version 1 established
that FAISS-GPU IVF crashes the Blackwell worker; versions 2–4 isolate the routing-quality failure corrected in
version 5. Full telemetry is stored in `runs/kaggle_10m_v5/ssa_10m_result.json`.
