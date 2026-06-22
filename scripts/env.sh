#!/usr/bin/env bash
# Shared environment for the tcss_ad_crave runners. Sourced by the run scripts.
#
# It (1) cd's to the repo root so `python -m src.<x>` resolves `src/` and `config/`,
# (2) optionally points HuggingFace at a local cache for the MiniLM clustering embedder
#     (small; auto-downloads on first use if absent), and (3) picks an interpreter.
# Override any of these before sourcing if your paths differ. None of it is required if
# you `pip install -r requirements.txt` into an active env and run the modules directly.

# repo root = parent of this scripts/ dir
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

# Every value below is overridable: `export VAR=...` before calling a runner, and the
# `${VAR:-default}` form keeps your value. The defaults are THIS machine's paths — edit
# them (or override) for another box.

# --- HuggingFace cache for the MiniLM embedder (optional; only speeds up loads) ---
# Set HF_HUB_OFFLINE=1 yourself if the embedder is already cached and you want offline loads.
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data_hdd/rave_main/packages/.cache/huggingface}"
export HF_HOME="${HF_HOME:-/data_hdd/rave_main/packages/huggingface_cache}"

# --- interpreter / package path ---
# If you installed via `pip install -r requirements.txt` into an active venv/conda env,
# just set:  export PY=python   (and leave CRAVE_PYTHONPATH empty).
export PYTHONPATH="${CRAVE_PYTHONPATH:-/data_hdd/rave_main/packages/python_packages}:${PYTHONPATH}"
PY="${PY:-/data_hdd/rave_main/packages/miniconda3/envs/debate_env/bin/python}"
export PY
