#!/usr/bin/env bash
# Stage A: re-cluster the base evidence into the K2/K8/D75/D80/D85 configs.
# All args are forwarded, e.g.:  scripts/generate_clusters.sh --dataset all
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
exec "$PY" -m src.generate_clusters "$@"
