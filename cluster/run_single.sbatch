#!/usr/bin/env bash
#SBATCH --job-name=cbm-benchmark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

: "${BENCHMARK:?BENCHMARK is required}"
: "${METHOD:?METHOD is required}"
: "${SEED:?SEED is required}"

python train.py \
  --benchmark "$BENCHMARK" \
  --method "$METHOD" \
  --seed "$SEED" \
  --device auto \
  --output_dir "${OUTPUT_DIR:-results}" \
  --checkpoint_every_timesteps "${CHECKPOINT_EVERY:-50000}" \
  --eval_every_timesteps "${EVAL_EVERY:-50000}"
