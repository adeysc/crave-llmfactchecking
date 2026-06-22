"""Binary veracity-accuracy evaluator.

Maps each sample's ground-truth `Label` and the model `judgement_verdict` into a binary
{real, fake} space and reports accuracy per dataset against the paper.

  verdict -> binary:  'True Claim'->real ; 'Misleading'/'Not enough Data'->fake ;
                      'judge_narratives fail'/missing/unknown -> SKIPPED.
  label   -> binary (per dataset; for Verite both OOC and miscaptioned are fake).

Used two ways:
  * imported by judge_clusters to score its judged outputs;
  * run directly to REPLAY the paper numbers from the shipped GPT-4o verdicts, e.g.
        python -m src.eval data/MMFakeBench_val_K4_wr.json
        python -m src.eval                      # all data/*_K4_wr.json (the paper gate)
"""
from __future__ import annotations

import os
import sys
import glob

from config.paths import DATA
from src.data_io import load_json

GATE_TOL = 0.02

# Paper CRAVE binary accuracy (Verite = T/F with OOC+miscaptioned fake).
PAPER = {
    "5PILS_OOC":       0.81,
    "MMFakeBench_val": 0.78,
    "dt_4k":           0.73,
    "verite":          0.82,
}
# Longest-first so "MMFakeBench_val" matches before any shorter prefix.
DATASET_KEYS = sorted(PAPER.keys(), key=len, reverse=True)

_VERDICT_FAKE = {"true claim": False, "misleading": True, "not enough data": True}
_LABEL_FAKE = {
    "5PILS_OOC":       {"true": False, "false": True},
    "MMFakeBench_val": {"true": False, "fake": True},
    "dt_4k":           {"true": False, "misleading": True,
                        "conflicted": True, "conficted": True},  # source typo kept
    "verite":          {"true": False, "out-of-context": True, "miscaptioned": True},
}


def dataset_of(path):
    base = os.path.basename(path)
    for key in DATASET_KEYS:
        if base.startswith(key):
            return key
    return None


def verdict_is_fake(verdict):
    if not isinstance(verdict, str):
        return None
    return _VERDICT_FAKE.get(verdict.strip().lower())


def label_is_fake(dataset, label):
    if not isinstance(label, str):
        return None
    return _LABEL_FAKE.get(dataset, {}).get(label.strip().lower())


def evaluate_file(path, verdict_field="judgement_verdict"):
    dataset = dataset_of(path)
    if dataset is None:
        raise ValueError(f"Cannot infer dataset from {os.path.basename(path)}")
    data = load_json(path)

    n_eval = n_correct = n_skip_verdict = n_skip_label = 0
    for sample in data.values():
        gt = label_is_fake(dataset, sample.get("Label"))
        if gt is None:
            n_skip_label += 1
            continue
        pred = verdict_is_fake(sample.get(verdict_field))
        if pred is None:
            n_skip_verdict += 1
            continue
        n_eval += 1
        n_correct += int(pred == gt)

    acc = (n_correct / n_eval) if n_eval else 0.0
    paper = PAPER[dataset]
    return {
        "file": os.path.basename(path), "dataset": dataset, "acc": acc, "paper": paper,
        "delta": acc - paper, "n_total": len(data), "n_eval": n_eval,
        "n_correct": n_correct, "n_skip_verdict": n_skip_verdict,
        "n_skip_label": n_skip_label, "within_tol": abs(acc - paper) <= GATE_TOL,
    }


def print_report(results, gate=True):
    print(f"{'file':34s} {'acc':>7} {'paper':>7} {'delta':>7} "
          f"{'n_eval':>7} {'skip_v':>7} {'skip_l':>7} {'ok':>4}")
    print("-" * 88)
    for r in results:
        print(f"{r['file']:34s} {r['acc']:7.3f} {r['paper']:7.2f} {r['delta']:+7.3f} "
              f"{r['n_eval']:7d} {r['n_skip_verdict']:7d} {r['n_skip_label']:7d} "
              f"{'PASS' if r['within_tol'] else 'FAIL':>4}")
    if gate:
        overall = bool(results) and all(r["within_tol"] for r in results)
        print("-" * 88)
        print(f"GATE: {'PASS' if overall else 'FAIL'} "
              f"(tolerance +/-{GATE_TOL}, {len(results)} file(s))")
        return overall
    return None


def main(argv):
    # Args are file paths/globs. No args -> replay the paper gate on data/*_K4_wr.json.
    if argv:
        paths = []
        for a in argv:
            paths.extend(sorted(glob.glob(a)) or [a])
    else:
        paths = sorted(glob.glob(str(DATA / "*_K4_wr.json")))
    results = [evaluate_file(p) for p in paths if dataset_of(p) is not None]
    if not results:
        print("No evaluable files.")
        return 2
    return 0 if print_report(results, gate=True) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
