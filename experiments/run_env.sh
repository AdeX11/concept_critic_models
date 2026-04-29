#!/bin/bash
set -e

# Headless rendering setup
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1

# --- Arguments ---
# Usage: bash run_env.sh <env_name> [timesteps] [seed]
ENV="${1:-cartpole}"
TS="${2:-20000}"
SEED="${3:-42}"

# --- Hyperparams ---
N_ENVS=4
N_STEPS=512
N_EPOCHS=10
BATCH_SIZE=256
DEVICE="cpu"

# --- Directory Logic ---
# RESULTS_BASE: where train.py dumps the raw experiment folders
# PLOT_DIR: where we consolidate the final analysis for this specific Env/Seed
RESULTS_BASE="results"
PLOT_DIR="plots/${ENV}_seed${SEED}"
REPLAY_DIR="$PLOT_DIR/replays"

mkdir -p "$RESULTS_BASE"
mkdir -p "$REPLAY_DIR"

echo "=========================================================="
echo "STARTING EXPERIMENT: $ENV | Seed: $SEED | TS: $TS"
echo "=========================================================="

# ---------------------------------------------------------------------------
# Helper: train + replay
# ---------------------------------------------------------------------------
train_and_replay() {
    local method="$1"
    local mode="$2"
    local temporal="${3:-gru}" # choices: gru, stacked, none
    
    # This string construction MUST match out_dir in train.py exactly
    local folder_name="${method}_${mode}_${temporal}_${ENV}_seed${SEED}"
    local full_out_dir="$RESULTS_BASE/$folder_name"
    local gif_path="$REPLAY_DIR/${method}_replay.gif"

    echo ""
    echo ">> [STEP 1/2] Training: $method ($mode | $temporal)"
    python train.py \
        --method "$method" \
        --training_mode "$mode" \
        --temporal_encoding "$temporal" \
        --env "$ENV" \
        --seed "$SEED" \
        --total_timesteps "$TS" \
        --n_envs "$N_ENVS" \
        --n_steps "$N_STEPS" \
        --n_epochs "$N_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --output_dir "$RESULTS_BASE"

    echo ">> [STEP 2/2] Replaying: $method"
    # Ensure the policy is built with the correct temporal encoding
    python replay.py \
        --env "$ENV" \
        --method "$method" \
        --temporal_encoding "$temporal" \
        --model_path "$full_out_dir/model.pt" \
        --output_gif "$gif_path" \
        --episodes 5 \
        --seed "$SEED" \
        --deterministic \
        ${4:+--show_concepts} # Pass show_concepts if 4th arg exists
}

# ---------------------------------------------------------------------------
# Run Methods
# ---------------------------------------------------------------------------

# 1) Standard Baseline
train_and_replay "no_concept" "two_phase" "gru"

# 2) Vanilla Concept Bottleneck
train_and_replay "vanilla_freeze" "joint" "gru"

#3) Concept-Actor-Critic
train_and_replay "concept_actor_critic" "joint" "gru" "show_concepts"

# ---------------------------------------------------------------------------
# Final Plotting
# ---------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "GENERATING COMPARISON PLOTS -> $PLOT_DIR"
echo "=========================================================="

python plot_results.py \
    --env "$ENV" \
    --results_dir "$RESULTS_BASE" \
    --output_dir "$PLOT_DIR" \
    --smooth_window 30

echo "Done. Comparison data and GIFs available in: $PLOT_DIR"