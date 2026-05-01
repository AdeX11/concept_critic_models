#!/bin/bash
#SBATCH --job-name=single_run
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -e

# -------------------------------
# CPU safety (CRITICAL)
# -------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# -------------------------------
# Args (passed through)
# -------------------------------
METHOD="$1"
MODE="$2"
TEMPORAL="$3"
ENV="${4:-cartpole}"
TS="${5:-100000}"
SEED="${6:-42}"

echo "========================================"
echo "METHOD=$METHOD MODE=$MODE TEMPORAL=$TEMPORAL"
echo "ENV=$ENV TS=$TS SEED=$SEED"
echo "========================================"

# -------------------------------
# Run the actual job
# -------------------------------
bash run_single.sh "$METHOD" "$MODE" "$TEMPORAL" "$ENV" "$TS" "$SEED"