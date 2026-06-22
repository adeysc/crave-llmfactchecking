"""Stage A -- re-cluster the base evidence (judge-independent).

Reads each baseevd `data/{ds}_K4_wo.json` and re-clusters its flat wo_refine evidence
(xv, xt) under every config except K4 (which already IS the baseevd): K2, K8, D75, D80,
D85. Writes `outputs/{ds}_{cfg}_wo.json` mirroring the baseevd schema but with `clusters`
replaced and the verdict/explanation blanked (Stage B fills those in).

Run ONCE; it is independent of the judge. Full dataset by default; --limit N is a
wiring/smoke shortcut (takes the first N samples -- see README on label-sort bias).

    python -m src.generate_clusters --dataset all
    python -m src.generate_clusters --dataset 5PILS_OOC --algos D --limit 10
"""
from __future__ import annotations

import copy
import argparse

from tqdm import tqdm

from config.clustering import DATASETS, CONFIGS, config_tag
from src.clustering import cluster_sample
from src.data_io import load_json, save_json, data_path, cluster_path


def run(dataset, algos, limit):
    data = load_json(data_path(dataset, "wo"))
    sample_ids = list(data.keys())
    if limit is not None:
        sample_ids = sample_ids[:limit]
        print(f"[{dataset}] limiting to first {len(sample_ids)} samples")

    for algo, tag in CONFIGS:
        if algo == "K" and tag == 4:
            continue                         # K4 == baseevd, already in data/
        if algo not in algos:
            continue
        cfg = config_tag(algo, tag)
        out = {}
        n_fail = 0
        for sid in tqdm(sample_ids, desc=cfg, leave=False):
            new = copy.deepcopy(data[sid])
            try:
                new["clusters"] = cluster_sample(data[sid], algo, tag)
            except Exception as e:
                n_fail += 1
                print(f"  cluster fail {sid[:14]} ({cfg}): {e}")
                new["clusters"] = []
            new["judgement_verdict"] = ""
            new["judgement_explanation"] = ""
            out[sid] = new
        save_json(cluster_path(dataset, cfg), out)
        avg_k = sum(len(s["clusters"]) for s in out.values()) / max(len(out), 1)
        note = f" ({n_fail} failed)" if n_fail else ""
        print(f"[OK] {dataset} {cfg} -> {cluster_path(dataset, cfg).name} "
              f"({len(out)} samples, avg clusters={avg_k:.2f}){note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    ap.add_argument("--algos", nargs="+", default=["K", "D"], choices=["K", "D"],
                    help="K = KMeans (k 2,8); D = Dynamic (thr 0.75,0.80,0.85, cap 8)")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N samples (smoke); 0 = full dataset (default)")
    args = ap.parse_args()

    limit = None if args.limit == 0 else args.limit
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        print(f"\n==== generate_clusters: {ds} ====")
        run(ds, args.algos, limit)


if __name__ == "__main__":
    main()
