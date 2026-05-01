#!/bin/bash
set -e

METHOD="$1"
MODE="$2"
TEMPORAL="$3"
ENV="$4"
TS="$5"
SEED="$6"

# --- Hyperparams ---
N_ENVS=4
N_STEPS=512
N_EPOCHS=10
BATCH_SIZE=256
DEVICE="cpu"

# --- Directories ---
RESULTS_BASE="results"
PLOT_DIR="plots/${ENV}_seed${SEED}"
REPLAY_DIR="$PLOT_DIR/replays"

mkdir -p "$RESULTS_BASE"
mkdir -p "$REPLAY_DIR"

# Match your naming convention exactly
folder_name="${METHOD}_${MODE}_${TEMPORAL}_${ENV}_seed${SEED}"
full_out_dir="$RESULTS_BASE/$folder_name"
gif_path="$REPLAY_DIR/${METHOD}_${MODE}_${TEMPORAL}.gif"

echo ""
echo ">> [TRAIN] $METHOD ($MODE | $TEMPORAL)"

python train.py \
    --method "$METHOD" \
    --training_mode "$MODE" \
    --temporal_encoding "$TEMPORAL" \
    --env "$ENV" \
    --seed "$SEED" \
    --total_timesteps "$TS" \
    --n_envs "$N_ENVS" \
    --n_steps "$N_STEPS" \
    --n_epochs "$N_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --output_dir "$RESULTS_BASE"

echo ">> [REPLAY] $METHOD ($MODE | $TEMPORAL)"

python replay.py \
    --env "$ENV" \
    --method "$METHOD" \
    --temporal_encoding "$TEMPORAL" \
    --model_path "$full_out_dir/model.pt" \
    --output_gif "$gif_path" \
    --episodes 5 \
    --seed "$SEED" \
    --deterministic