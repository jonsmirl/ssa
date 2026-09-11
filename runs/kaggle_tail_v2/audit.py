"""Read-only provenance, baseline, quality, and strict-retrieval comparison.

Run from repo root: python runs/kaggle_tail_v2/audit.py
The downloaded token archives are private, ignored inputs needed for token checks.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
paths = {version: ROOT / "runs" / f"kaggle_tail_{version}" for version in ("v1", "v2")}
reports = {version: json.loads((path / "ssa_tail_fullscale.json").read_text()) for version, path in paths.items()}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


provenance = {}
for version, path in paths.items():
    manifest = json.loads((path / "tail_bundle/manifest.json").read_text())
    mismatches = {name: sha(path / name) if (path / name).exists() else "MISSING" for name, expected in manifest["sha256"].items()
                  if not (path / name).exists() or sha(path / name) != expected}
    provenance[version] = {
        "report_sha256": sha(path / "ssa_tail_fullscale.json"),
        "manifest_files": len(manifest["sha256"]),
        "hash_mismatches": mismatches,
        "report_manifest_matches_file": reports[version]["manifest"] == manifest,
        "status": reports[version]["status"],
        "driver_elapsed_seconds": reports[version]["finished"] - reports[version]["started"],
    }
token_arrays = {v: np.load(path / "tail_bundle/tokens.npz")["test"] for v, path in paths.items()}
provenance["identical_test_tokens"] = bool(np.array_equal(token_arrays["v1"], token_arrays["v2"]))
provenance["identical_saved_gains"] = (paths["v1"] / "tail_bundle/gains.json").read_bytes() == (paths["v2"] / "tail_bundle/gains.json").read_bytes()
selection = json.loads((paths["v2"] / "tail_bundle/selection.json").read_text())
provenance["embedded_selection_matches_frozen"] = (paths["v2"] / "tail_bundle/selection.json").read_bytes() == (ROOT / "runs/tail_revision_selection.json").read_bytes()
provenance["selection_validation_hash_matches"] = sha(ROOT / selection["validation_artifact"]) == selection["validation_sha256"]
provenance["selected_config_matches_protocol"] = all(reports["v2"]["protocol"][k] == v for k, v in selection["selected_config"].items())

corpus = {}
baseline = {}
fields = ("targets", "ce_nats", "perplexity", "token_top1_accuracy", "seconds", "peak_allocated_gb")
for context, modes in reports["v2"]["corpus"].items():
    previous = reports["v1"]["corpus"][context]
    corpus[context] = {mode: {k: row[k] for k in fields} for mode, row in modes.items()}
    corpus[context]["old_tail"] = {k: previous["tail"][k] for k in fields}
    tail = modes["tail"]
    pairs = {}
    for label, ref in (("sparse", modes["sparse"]), ("old_tail", previous["tail"])):
        tw, rw = tail["windows"], ref["windows"]
        assert len(tw) == len(rw)
        assert all((a["start"], a["targets"]) == (b["start"], b["targets"]) for a, b in zip(tw, rw))
        pairs[label] = {"windows": len(tw), "tail_lower_ce": sum(a["ce_nats"] < b["ce_nats"] for a, b in zip(tw, rw)),
                        "tail_equal_ce": sum(a["ce_nats"] == b["ce_nats"] for a, b in zip(tw, rw)),
                        "ce_delta": tail["ce_nats"] - ref["ce_nats"],
                        "ppl_relative_reduction": 1 - tail["perplexity"] / ref["perplexity"]}
    corpus[context]["paired"] = pairs
    baseline[context] = {}
    for mode in ("dense", "sparse"):
        left, right = modes[mode], previous[mode]
        baseline[context][mode] = {
            "ce_abs_delta": abs(left["ce_nats"] - right["ce_nats"]),
            "accuracy_abs_delta": abs(left["token_top1_accuracy"] - right["token_top1_accuracy"]),
            "max_window_ce_abs_delta": max(abs(a["ce_nats"] - b["ce_nats"]) for a, b in zip(left["windows"], right["windows"])),
            "target_layout_equal": [(x["start"], x["targets"]) for x in left["windows"]] == [(x["start"], x["targets"]) for x in right["windows"]],
        }

retrieval = {}
for version, report in reports.items():
    entries = []
    groups = {}
    for trial in report["probes"]:
        for mode, row in trial["modes"].items():
            r = row["retrieval"]
            scores = r["candidate_logits"]
            gold = scores[r["gold"]]
            strongest_other = max(v for k, v in scores.items() if k != r["gold"])
            entry = {"context": trial["context"], "depth": trial["depth"], "mode": mode,
                     "order_resolved_correct": r["correct"], "strict_correct": gold > strongest_other,
                     "gold_top_tie": gold == strongest_other,
                     "any_top_tie": sum(v == max(scores.values()) for v in scores.values()) > 1,
                     "gold_margin": gold - strongest_other, "seconds": row["seconds"],
                     "peak_allocated_gb": row["peak_allocated_gb"]}
            entries.append(entry)
            for key in (f"{trial['context']}/{mode}", f"all/{mode}"):
                group = groups.setdefault(key, {"n": 0, "order_resolved_correct": 0, "strict_correct": 0, "gold_top_tie": 0, "any_top_tie": 0})
                group["n"] += 1
                for field in ("order_resolved_correct", "strict_correct", "gold_top_tie", "any_top_tie"):
                    group[field] += entry[field]
    retrieval[version] = {"groups": groups, "entries": entries}

probe_baseline_equal = len(reports["v1"]["probes"]) == len(reports["v2"]["probes"]) and all(
    a["context"] == b["context"] and a["depth"] == b["depth"] and
    all(a["modes"][mode]["retrieval"] == b["modes"][mode]["retrieval"] for mode in ("dense", "sparse"))
    for a, b in zip(reports["v1"]["probes"], reports["v2"]["probes"]))
print(json.dumps({"provenance": provenance, "baseline": baseline,
                  "probe_baseline_retrieval_equal": probe_baseline_equal,
                  "corpus": corpus, "retrieval": retrieval,
                  "scope": "Frozen v2 regression on previously inspected test/probes; strict wins exclude order-resolved ties. Candidate logits are stored rounded to five decimals."}, indent=2))
