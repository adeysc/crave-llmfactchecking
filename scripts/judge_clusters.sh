#!/usr/bin/env bash
# Stage B: judge every config and score it. All args are forwarded, e.g.:
#   scripts/judge_clusters.sh --dataset all --judge openai
#   scripts/judge_clusters.sh --dataset MMFakeBench_val --judge openai --limit 10
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
exec "$PY" -m src.judge_clusters "$@"
