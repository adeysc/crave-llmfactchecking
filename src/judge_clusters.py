"""Stage B -- judge every clustering variant with one judge and compare accuracy.

For each dataset it iterates the 6 configs (K2 K4 K8 D75 D80 D85), reads the matching
Stage-A clustering file (K4 == the baseevd in data/), runs the judge over each sample's
clusters, writes a judged copy, and scores it with src.eval. The judge model is loaded
ONCE and reused across all datasets/configs.

Full dataset by default; --limit N is a smoke shortcut (first N samples -- see README on
label-sort bias; use full for real numbers).

    python -m src.judge_clusters --dataset all --judge openai
    python -m src.judge_clusters --dataset MMFakeBench_val --judge openai --limit 10
"""
from __future__ import annotations

import argparse

from tqdm import tqdm

from config.clustering import DATASETS, CONFIGS, config_tag
from config.judges import JUDGES, DEFAULT_JUDGE
from src import judge as judge_mod
from src.backends import make_backend
from src.data_io import (load_json, save_json, config_input_path, judged_path,
                         summary_path)
from src import eval as eval_mod


def judge_file(in_path, out_path, limit):
    data = load_json(in_path)
    ids = list(data.keys())
    if limit is not None:
        ids = ids[:limit]
    n_fail = 0
    for sid in tqdm(ids, desc=out_path.name[:28], leave=False):
        s = data[sid]
        try:
            verdict, expl = judge_mod.judge_narratives(s.get("clusters", []),
                                                       s.get("Claim", ""))
        except Exception as e:
            verdict, expl = "judge_narratives fail", f"error: {e}"
        if verdict == "judge_narratives fail":
            n_fail += 1
        s["judgement_verdict"] = verdict
        s["judgement_explanation"] = expl
    out = {sid: data[sid] for sid in ids}
    save_json(out_path, out)
    avg_k = sum(len(data[sid].get("clusters", [])) for sid in ids) / max(len(ids), 1)
    return out, n_fail, avg_k


def run(dataset, judge_name, limit):
    results = []
    for algo, tag in CONFIGS:
        cfg = config_tag(algo, tag)
        in_path = config_input_path(dataset, cfg)
        if not in_path.exists():
            print(f"[skip] missing {in_path.name} (run generate_clusters first)")
            continue
        out_path = judged_path(dataset, cfg, judge_name)
        _, n_fail, avg_k = judge_file(in_path, out_path, limit)
        ev = eval_mod.evaluate_file(out_path)
        results.append({"config": cfg, "acc": ev["acc"], "n_eval": ev["n_eval"],
                        "fails": n_fail, "avg_clusters": round(avg_k, 2),
                        "file": out_path.name})
        print(f"  {cfg:4s} acc={ev['acc']:.3f} n_eval={ev['n_eval']} "
              f"fails={n_fail} avgK={avg_k:.2f}")

    paper = eval_mod.PAPER.get(dataset)
    print(f"\n=== {dataset} | judge={judge_name} | paper(gpt-4o)={paper} ===")
    print(f"{'cfg':>5} {'acc':>7} {'n_eval':>7} {'fails':>6} {'avgK':>6}")
    for r in results:
        print(f"{r['config']:>5} {r['acc']:7.3f} {r['n_eval']:7d} "
              f"{r['fails']:6d} {r['avg_clusters']:6.2f}")
    summary = {"dataset": dataset, "judge": judge_name, "limit": limit,
               "paper": paper, "results": results}
    save_json(summary_path(dataset, judge_name), summary)
    print(f"summary -> {summary_path(dataset, judge_name).name}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    ap.add_argument("--judge", default=DEFAULT_JUDGE, choices=sorted(JUDGES),
                    help=f"judge backend (default {DEFAULT_JUDGE})")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N samples per config (smoke); 0 = full (default)")
    args = ap.parse_args()

    limit = None if args.limit == 0 else args.limit
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    print(f"Judge backend: {args.judge} (loaded once for {len(datasets)} dataset(s))")
    judge_mod.set_backend(make_backend(args.judge))

    summaries = []
    for ds in datasets:
        print(f"\n================ DATASET: {ds} ================")
        summaries.append(run(ds, args.judge, limit))

    if len(summaries) > 1:
        print("\n######## ALL-DATASET OVERVIEW ########")
        for s in summaries:
            if not s["results"]:
                print(f"{s['dataset']:16s}: no results")
                continue
            best = max(s["results"], key=lambda r: r["acc"])
            print(f"{s['dataset']:16s} paper={s['paper']}  "
                  f"best={best['config']} acc={best['acc']:.3f}")


if __name__ == "__main__":
    main()
