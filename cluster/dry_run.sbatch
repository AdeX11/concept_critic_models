#!/usr/bin/env bash
#SBATCH --job-name=cbm-dry-run
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

python train.py \
  --benchmark "${BENCHMARK:-cartpole}" \
  --method "${METHOD:-no_concept}" \
  --seed "${SEED:-42}" \
  --total_timesteps 100 \
  --n_envs 1 \
  --n_steps 16 \
  --batch_size 16 \
  --output_dir "${OUTPUT_DIR:-dry_run_results}" \
  --checkpoint_every_timesteps 100 \
  --eval_every_timesteps 100
