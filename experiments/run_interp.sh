#!/bin/bash
set -e

# Headless rendering setup
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1

# --- Arguments ---
# Usage: bash experiments/run_interp.sh <env_name> [seed] [method] [temporal_encoding]
ENV="${1:-cartpole}"
SEED="${2:-42}"
METHOD="${3:-concept_actor_critic}"
TEMPORAL="${4:-gru}"

# --- Paths ---
RESULTS_BASE="results"
INTERP_RESULTS_DIR="interp_results/${ENV}_seed${SEED}_${METHOD}"
INTERP_PLOTS_DIR="interp_plots/${ENV}_seed${SEED}_${METHOD}"

# Model path follows the run_env.sh naming convention
MODEL_FOLDER="${METHOD}_joint_${TEMPORAL}_${ENV}_seed${SEED}"
MODEL_PATH="${RESULTS_BASE}/${MODEL_FOLDER}/model.pt"

mkdir -p "$INTERP_RESULTS_DIR"
mkdir -p "$INTERP_PLOTS_DIR"

echo "=========================================================="
echo "INTERPRETABILITY SUITE: $ENV | Method: $METHOD | Seed: $SEED"
echo "=========================================================="
echo ""
echo "  Model path      : $MODEL_PATH"
echo "  JSON results    : $INTERP_RESULTS_DIR/"
echo "  Plots / GIFs    : $INTERP_PLOTS_DIR/"
echo ""

# Check model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Run experiments/run_env.sh $ENV first to train the model."
    exit 1
fi

# Run the full interpretability suite
python experiments/run_interp.py \
    --env "$ENV" \
    --method "$METHOD" \
    --model_path "$MODEL_PATH" \
    --temporal_encoding "$TEMPORAL" \
    --seed "$SEED" \
    --output_dir "$INTERP_RESULTS_DIR" \
    --plots_dir "$INTERP_PLOTS_DIR" \
    --device cpu

echo ""
echo "=========================================================="
echo "Done. Check outputs:"
echo "  $INTERP_RESULTS_DIR/"
echo "  $INTERP_PLOTS_DIR/"
echo "=========================================================="