"""
Model-agnostic eval harness for the SSA frozen-swap: needle-in-a-haystack (NIAH) retrieval and
LM loss, run *through a real HF causal LM* (the measurement the repo was missing). Toggle SSA with
`from ssa.gemma_ssa import install_ssa; install_ssa(model, budget_frac=...)` before calling.

Two signals, complementary:
  - NIAH accuracy: does the model recover a planted fact from depth `d` in an `n`-token context?
    This is where SSA selection is actually tested (raw LM loss is dominated by the local window).
  - LM loss: mean NLL on held-out real text — the cheap, dense quality signal.

The pure prompt/scoring functions are tokenizer-free and unit-tested; the model-driving functions
lazy-import torch so the mechanics test without a forward pass. Designed so swapping the model (small
dry-run model -> Gemma-4) is a one-line change.

Run a tiny dry-run:  python -m ssa.gemma_ssa_eval --model Qwen/Qwen2.5-0.5B --lengths 512,1024
"""
from __future__ import annotations
import re
import random

# A neutral filler sentence repeated to pad context (no incidental digits to confuse scoring).
FILLER = ("The garden path wound past the old stone wall where ivy climbed in the quiet afternoon "
          "light, and the air carried the scent of rain that had not yet arrived. ")
NEEDLE_TMPL = "Remember this: the secret access code is {val}."
QUESTION = " Question: what is the secret access code? Answer: the secret access code is"


def make_niah_text(val: int, depth_frac: float, n_filler_units: int):
    """Assemble (context, question) with the needle planted at `depth_frac` through the filler.
    Tokenizer-free and deterministic — the unit-testable core."""
    depth_frac = min(max(depth_frac, 0.0), 1.0)
    cut = int(round(n_filler_units * depth_frac))
    pre = FILLER * cut
    post = FILLER * (n_filler_units - cut)
    needle = NEEDLE_TMPL.format(val=val)
    return pre + needle + " " + post + QUESTION, str(val)


def score_continuation(generated: str, val: str) -> bool:
    """A hit iff the exact needle value appears as a standalone number in the model's continuation."""
    return re.search(rf"(?<!\d){re.escape(val)}(?!\d)", generated) is not None


def _rand_code(rng: random.Random) -> int:
    return rng.randint(10000, 99999)  # 5 digits: distinctive, single-token-ish, easy to score


def niah_accuracy(model, tokenizer, n_tokens, depths=(0.1, 0.5, 0.9),
                  trials=5, max_new_tokens=12, seed=0, device=None):
    """Fraction of (depth, trial) NIAH probes the model answers correctly at ~`n_tokens` context.
    Pads filler so the tokenized prompt lands near `n_tokens` (binary-free: estimate then trim)."""
    import torch
    rng = random.Random(seed)
    dev = device or (next(model.parameters()).device if hasattr(model, "parameters") else "cpu")
    # estimate filler units to hit n_tokens (cheap: tokens-per-unit from one sample)
    sample, _ = make_niah_text(_rand_code(rng), 0.5, 50)
    tpu = max(1.0, len(tokenizer(sample)["input_ids"]) / 50.0)
    units = max(1, int(n_tokens / tpu))
    hits = total = 0
    for d in depths:
        for _ in range(trials):
            val = _rand_code(rng)
            text, gold = make_niah_text(val, d, units)
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=n_tokens + 64)["input_ids"].to(dev)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=getattr(tokenizer, "eos_token_id", None))
            gen = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            hits += int(score_continuation(gen, gold))
            total += 1
    return hits / total if total else 0.0


def lm_loss(model, tokenizer, texts, max_len=2048, device=None):
    """Mean per-token NLL (nats) over `texts` — the dense quality signal. Lower is better."""
    import torch
    dev = device or (next(model.parameters()).device if hasattr(model, "parameters") else "cpu")
    tot_nll = tot_tok = 0.0
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"].to(dev)
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            logits = model(ids).logits[:, :-1].float()
            tgt = ids[:, 1:]
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum")
        tot_nll += float(nll); tot_tok += tgt.numel()
    return tot_nll / tot_tok if tot_tok else float("nan")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--lengths", default="512,1024")
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto").eval()
    print(f"{'n_tokens':>9} {'NIAH acc':>9}")
    for n in (int(x) for x in args.lengths.split(",")):
        acc = niah_accuracy(model, tok, n, trials=args.trials)
        print(f"  {n:>7} {acc:>9.3f}")


if __name__ == "__main__":
    main()
