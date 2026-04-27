#!/bin/bash
# run_env.sh — Train and visualize any registered environment with all three methods.
# Usage: bash run_env.sh <env_name> [total_timesteps] [seed]
# Example: bash run_env.sh cartpole 50000 42
# Example: bash run_env.sh mountain_car 500000 42

set -e

# Prevent SDL/pygame from opening windows (headless rendering)
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1

ENV="${1:-cartpole}"
TS="${2:-50000}"
SEED="${3:-42}"

N_ENVS=4
N_STEPS=512
N_EPOCHS=10
BATCH_SIZE=256
DEVICE="cpu"
RESULTS_DIR="results/${ENV}_run"
REPLAY_DIR="$RESULTS_DIR/replays"
mkdir -p "$REPLAY_DIR"

echo "========================================"
echo "Training + Visualization: env=$ENV | ts=$TS | seed=$SEED"
echo "========================================"

# ---------------------------------------------------------------------------
# Helper: train + replay
# ---------------------------------------------------------------------------
train_and_replay() {
    local method="$1"
    local mode="$2"
    local temporal="${3:-none}"

    echo ""
    echo "[Training] $method (mode=$mode, temporal=$temporal) ..."
    python train.py \
        --method "$method" \
        --training_mode "$mode" \
        --env "$ENV" --seed "$SEED" \
        --total_timesteps "$TS" --n_envs "$N_ENVS" \
        --n_steps "$N_STEPS" --n_epochs "$N_EPOCHS" --batch_size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --output_dir "$RESULTS_DIR"

    local out_dir="$RESULTS_DIR/${method}_${mode}_${temporal}_${ENV}_seed${SEED}"
    local gif_path="$REPLAY_DIR/${method}_replay.gif"

    echo "[Replay] $method → $gif_path"
    python replay.py \
        --env "$ENV" --method "$method" \
        --model_path "$out_dir/model.pt" \
        --output_gif "$gif_path" \
        --seed "$SEED" --episodes 1 --max_steps 200 --fps 6 --deterministic \
        ${4:+--show-concepts}
}

# ---------------------------------------------------------------------------
# 1) no_concept (baseline)
# ---------------------------------------------------------------------------
train_and_replay "no_concept" "two_phase" "none"

# ---------------------------------------------------------------------------
# 2) vanilla_freeze (joint)
# ---------------------------------------------------------------------------
train_and_replay "vanilla_freeze" "joint" "none"

# ---------------------------------------------------------------------------
# 3) concept_actor_critic (joint)
# ---------------------------------------------------------------------------
train_and_replay "concept_actor_critic" "joint" "none" "show_concepts"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "All done! Results in $RESULTS_DIR"
echo "Replays in $REPLAY_DIR"
echo "========================================"
ls -lh "$REPLAY_DIR"
