#!/bin/bash
set -e

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Headless rendering setup
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1

# --- Arguments ---
ENV="${1:-cartpole}"
TS="${2:-100000}"
SEED="${3:-42}"

# --- Hyperparams ---
N_ENVS=4
N_STEPS=512
N_EPOCHS=10
BATCH_SIZE=256
DEVICE="cpu"

# --- Directory Logic ---
RESULTS_BASE="results"
FINAL_PLOTS="final_plots"
FINAL_RESULTS="final_results"

mkdir -p "$RESULTS_BASE"
mkdir -p "$FINAL_PLOTS"
mkdir -p "$FINAL_RESULTS"

echo "=========================================================="
echo "SWEEP EXPERIMENT: $ENV | Seed: $SEED | TS: $TS"
echo "=========================================================="

# ---------------------------------------------------------------------------
# Helper: train only (no replay for speed)
# ---------------------------------------------------------------------------
train_only() {
    local method="$1"
    local mode="$2"
    local temporal="${3:-none}"

    local folder_name="${method}_${mode}_${temporal}_${ENV}_seed${SEED}"
    local full_out_dir="$RESULTS_BASE/$folder_name"

    echo ""
    echo ">> Training: $method ($mode | $temporal)"
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
        --output_dir "$RESULTS_BASE" &
    
    PIDS+=($!)
}

# ---------------------------------------------------------------------------
# Run all 9 combinations (NOW PARALLEL)
# ---------------------------------------------------------------------------
PIDS=()

# 1) no_concept baseline
train_only "no_concept" "two_phase" "none"

# 2-5) vanilla_freeze variants
train_only "vanilla_freeze" "two_phase" "none"
train_only "vanilla_freeze" "two_phase" "gru"
train_only "vanilla_freeze" "joint" "none"
train_only "vanilla_freeze" "joint" "gru"

# 6-9) concept_actor_critic variants
train_only "concept_actor_critic" "two_phase" "none"
train_only "concept_actor_critic" "two_phase" "gru"
train_only "concept_actor_critic" "joint" "none"
train_only "concept_actor_critic" "joint" "gru"

echo ""
echo "Running ${#PIDS[@]} jobs in parallel..."
echo ""

# WAIT FOR ALL TRAINING TO FINISH
wait "${PIDS[@]}"

# ---------------------------------------------------------------------------
# Final Comparison
# ---------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "GENERATING COMPARISON -> $FINAL_PLOTS & $FINAL_RESULTS"
echo "=========================================================="

python compare_sweep.py \
    --env "$ENV" \
    --results_dir "$RESULTS_BASE" \
    --output_plots "$FINAL_PLOTS" \
    --output_results "$FINAL_RESULTS"

echo ""
echo "Done. Final outputs:"
echo "  Plots:   $FINAL_PLOTS"
echo "  Results: $FINAL_RESULTS"

