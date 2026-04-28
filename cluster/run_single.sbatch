#!/usr/bin/env bash
#SBATCH --job-name=cbm-benchmark
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs
source .venv/bin/activate
export PYTHONUNBUFFERED=TRUE

: "${BENCHMARK:?BENCHMARK is required}"
: "${METHOD:?METHOD is required}"
: "${SEED:?SEED is required}"

extra_args=()
if [[ -n "${TOTAL_TIMESTEPS:-}" ]]; then
  extra_args+=(--total_timesteps "$TOTAL_TIMESTEPS")
fi
if [[ -n "${GVF_PAIRING:-}" ]]; then
  extra_args+=(--gvf_pairing "${GVF_PAIRING//:/,}")
fi

python train.py \
  --benchmark "$BENCHMARK" \
  --method "$METHOD" \
  --seed "$SEED" \
  --temporal_encoding "${TEMPORAL_ENCODING:-none}" \
  --training_mode "${TRAINING_MODE:-two_phase}" \
  --learning_rate "${LEARNING_RATE:-0.0003}" \
  --ent_coef "${ENT_COEF:-0.01}" \
  --lambda_v "${LAMBDA_V:-0.5}" \
  --lambda_s "${LAMBDA_S:-0.5}" \
  --num_labels "${NUM_LABELS:-500}" \
  --query_num_times "${QUERY_NUM_TIMES:-1}" \
  --device auto \
  --output_dir "${OUTPUT_DIR:-results}" \
  --checkpoint_every_timesteps "${CHECKPOINT_EVERY:-50000}" \
  --eval_every_timesteps "${EVAL_EVERY:-50000}" \
  --eval_episodes "${EVAL_EPISODES:-20}" \
  "${extra_args[@]}"
