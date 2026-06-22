
# Fact-Checking with Contextual Narratives: Leveraging Retrieval-Augmented LLMs for Social Media Analysis

**Authors:** [Arka Ujjal Dey](https://orcid.org/0000-0001-8392-1574), [Muhammad Junaid Awan](https://orcid.org/0000-0002-3857-1293), [Georgia Channing](https://orcid.org/0009-0001-6354-7527), [Christian Schroeder de Witt](https://orcid.org/0000-0003-4245-1179), and [John Collomosse](https://orcid.org/0000-0003-3580-4685)

**Venue:** *IEEE Transactions on Computational Social Systems (TCSS)*

---

### Abstract

Fact-checking systems have gained traction as scalable solutions, yet they often face challenges such as handling diverse evidence sources, integrating multimodal data, and presenting comprehensive narratives. In this work, we propose CRAVE (Cluster-based Retrieval Augmented Verification with Explanation), a novel framework that integrates retrieval-augmented Large Language Models (LLMs) with clustering techniques to address multimodal misinformation on social media. The framework is designed to process multi-modal inputs (text and images) and iteratively refine evidence through agent-based mechanisms. We validated the framework on multiple real-world and synthetic datasets, showing that breaking up evidence into narrative clusters improves both retrieval precision, clustering quality, and judgment accuracy, showcasing its potential as a robust decision-support tool for fact-checkers.

---

### Code and Data Release


Self-contained reproduction code for the clustering-and-LLM-judgment  of
**CRAVE** (Cluster-based Retrieval-Augmented Verification with Explanation)

1. **`generate_clusters`** — re-cluster the collected evidence under different schemes.
2. **`judge_clusters`** — judge each clustering with an LLM and score binary veracity
   accuracy.

The upstream evidence-collection + visual-feature stage is **out of scope**; this package
starts from the evidence already stored in `data/` (it is produced by `tcss_ad_gather_score`).

> **CPU-only release.** Everything runs with a plain `pip install`, no GPU: replay
> (`src.eval`), `generate_clusters`, and the GPT-4o judge (`judge_clusters --judge openai`,
> bring-your-own-key). See `RELEASE_NOTES.md`.

```
crave-llmfactchecking/
  config/      judges.py  clustering.py  paths.py     # all knobs live here
  src/         clustering.py  judge.py  backends.py    # logic
               eval.py  data_io.py
               generate_clusters.py  judge_clusters.py # the two entry points
  scripts/     env.sh  generate_clusters.sh  judge_clusters.sh
  data/        {dataset}_K4_wo.json  {dataset}_K4_wr.json   (8 files)
  outputs/     generated clusters, judged files, experiment summaries (flat)
```

---

## Data (the only inputs)

Four datasets: `5PILS_OOC`, `MMFakeBench_val`, `dt_4k` (the paper's DP set), `verite`.
Each ships **two** self-contained JSONs — nothing else is needed (no pickles, no images):

| file | what it is |
|---|---|
| `{ds}_K4_wo.json` | **canonical baseevd** — flat `wo_refine` evidence pool (`xv`, `xt`) + k=4 KMeans clusters + the `wo` verdicts. This is what `generate_clusters` re-clusters. |
| `{ds}_K4_wr.json` | `with_refine` evidence + the **stored GPT-4o verdicts**. The paper-faithful reference used for replay. |

Every evidence item uses a **minimal schema** (repo-wide, identical to `tcss_ad_gather_score`):
`xt` → `{summary, domain}`, `xv` → `{summary, domain, xv_s}`. Clustering and the judge run on
`summary` alone; `caption`/`title`/`image_path`/`date`/`page_link` are not included.

---

## Reproducing the paper numbers (replay — no API key, no GPU)

The shipped GPT-4o verdicts in the `_wr` files **are** the paper's results. Just score
them with the evaluator (this is the headline check and costs nothing):

```bash
python -m src.eval                                  # all data/*_K4_wr.json (the gate)
python -m src.eval data/MMFakeBench_val_K4_wr.json  # one dataset
```

Expected (binary accuracy, within ±0.02 of the paper):

| dataset | replay | paper |
|---|---|---|
| 5PILS_OOC | 0.816 | 0.81 |
| MMFakeBench_val | 0.778 | 0.78 |
| dt_4k | 0.738 | 0.73 |
| verite | 0.825 | 0.82 |

`src.eval` maps `True Claim`→real, `Misleading`/`Not enough Data`→fake,
`judge_narratives fail`→skipped; for Verite both out-of-context and miscaptioned are fake.

---

## The two scripts

Run from the repo root (or use the `scripts/*.sh` wrappers, which set the env for you).

### 1) `generate_clusters` — re-cluster the evidence

Reads `data/{ds}_K4_wo.json`, writes `outputs/{ds}_{cfg}_wo.json` (clusters replaced,
verdicts blank). Judge-independent; run once. No API key needed.

```bash
python -m src.generate_clusters --dataset all
python -m src.generate_clusters --dataset 5PILS_OOC --algos D --limit 10   # smoke
```

### 2) `judge_clusters` — judge each config and score it

Reads each config's clustering (K4 from `data/`, the rest from `outputs/`), runs the
judge, writes `outputs/{ds}_{cfg}_judged_{judge}.json` + `outputs/experiment_{ds}_{judge}.json`,
and prints a per-config accuracy table.

```bash
python -m src.judge_clusters --dataset all --judge openai                       # FULL
python -m src.judge_clusters --dataset 5PILS_OOC --judge openai --limit 10       # smoke
```

> **`--limit` defaults to `0` = the FULL dataset.** Omitting it runs everything (all configs,
> all samples). Pass an explicit `--limit N` for quick runs.
>
> **`--limit N` takes the first N samples in file order.** Some sets are **sorted by
> label** (e.g. `MMFakeBench_val` is all-real for its first ~300 rows), so a small
> `--limit` slice is one-class and its accuracy is meaningless. **Omit `--limit` (full)
> for real numbers**; use it only for quick wiring tests.

### End-to-end run (cluster → judge → predictions)

Run the two stages in order — e.g. a 10-sample smoke of `5PILS_OOC`:

```bash
export OPENAI_API_KEY=...        # the judge is GPT-4o, bring-your-own-key (paid)
                                 # (in the github_upload layout: `source ../secrets.local.sh`)

# 1) re-cluster (embeds `summary`; produces the K2/K8/D75/D80/D85 configs) — free
python -m src.generate_clusters --dataset 5PILS_OOC --limit 10

# 2) judge EVERY config -> per-config predictions + an accuracy summary
python -m src.judge_clusters   --dataset 5PILS_OOC --judge openai --limit 10

# (or via the wrappers, which source scripts/env.sh for the embedder cache + interpreter:)
#   ./scripts/generate_clusters.sh --dataset 5PILS_OOC --limit 10
#   ./scripts/judge_clusters.sh    --dataset 5PILS_OOC --judge openai --limit 10
```

This writes to `outputs/`:
- `5PILS_OOC_{K2,K4,K8,D75,D80,D85}_judged_openai.json` — the **final predictions** (each cluster
  gains `judgement_verdict` + `judgement_explanation`).
- `experiment_5PILS_OOC_openai.json` — the per-config accuracy summary.

> Only `judge_clusters --judge openai` makes paid calls; `generate_clusters` and `src.eval` are
> free. For a no-cost reproduction, use the replay gate above. A cost-controlled
> `--judge openai-mini` (gpt-4o-mini) is also available.

---

## Clustering schemes (the 6 configs)

Two schemes, each parameterised by its **own** natural knob:

- **`K` — KMeans**: cluster **count** `k ∈ {2, 4, 8}` → exactly k clusters.
- **`D` — Dynamic (Agglomerative)**: cosine-**similarity threshold** (paper Fig. 4). The
  cluster count *emerges* from the threshold (`distance = 1 − similarity`) and is **capped
  at 8**. Tags: `D75`/`D80`/`D85` = 0.75/0.80/0.85. Higher threshold → tighter → more
  clusters.

The 6 configs: `K2 K4 K8  D75 D80 D85`. **`K4` is the baseevd itself**, so
`generate_clusters` only produces the other 5. Text is embedded with a local
`SentenceTransformer("all-MiniLM-L6-v2")` (384-d, runs on CPU). All of this is configured in
`config/clustering.py`.

---

## The judge (`config/judges.py`)

- **`openai`** (default) — the paper's GPT-4o judge. Bring your own key: `export OPENAI_API_KEY=...`
  (or, in the github_upload layout, `source ../secrets.local.sh`). Reproduces the paper's accuracy.
- **`openai-mini`** — same path on `gpt-4o-mini`, input capped at 4000 tokens (cost-controlled).

The judge **prompt** (`src/judge.py`) is the paper's, verbatim, and identical for every backend.

---

## Environment & reproducibility

Dependencies are in `requirements.txt` 

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Sanity check (no API key, no GPU, seconds)** — confirms the install + data are good:

```bash
python -m src.eval        # should print 4 PASS rows reproducing the paper numbers
```

**Machine-specific paths** live only in `scripts/env.sh` (HuggingFace cache for the MiniLM
embedder + interpreter) and are all overridable. If you `pip install`-ed into an active env, you
don't need it — just run the modules directly:

```bash
python -m src.generate_clusters --dataset all
python -m src.judge_clusters    --dataset all --judge openai
```

The MiniLM embedder downloads once (~80 MB) on first use, then loads from the HuggingFace cache. To
point at an existing cache (and optionally force offline loads), export before running:

```bash
export HF_HUB_CACHE=/path/to/hf/cache
export HF_HOME=/path/to/hf/home
# export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # only if the embedder is already cached
```




 


For DP Dataset download and model weights refer to 
https://github.com/cejunaid/decade-ieee-crave-support-repo
---

### Citation

If you find this work useful for your research, please cite our preprint. This citation will be updated to reflect the full *IEEE Transactions on Computational Social Systems* (TCSS) version upon its final publication.
```bibtex
@article{dey2025fact,
  title={Fact-checking with contextual narratives: Leveraging retrieval-augmented llms for social media analysis},
  author={Dey, Arka Ujjal and Awan, Muhammad Junaid and Channing, Georgia and de Witt, Christian Schroeder and Collomosse, John},
  journal={arXiv preprint arXiv:2504.10166},
  year={2025}
}

```


