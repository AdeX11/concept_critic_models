#!/bin/bash
# run_short.sh — Short parallel comparison of all 3 methods on dynamic_obstacles (8x8).
#
# Methods compared:
#   1. no_concept                        — plain PPO baseline
#   2. vanilla_freeze (two_phase)        — LICORICE supervised CBM
#   3. concept_actor_critic (gru)        — new method with temporal encoding
#   4. concept_actor_critic (none)       — ablation: no temporal encoding

set -e

ENV=pick_place
TS=200000
N_ENVS=4
N_STEPS=256
N_EPOCHS=5
BATCH=128
SEED=42
RESULTS_DIR=/results/short
PLOTS_DIR=/plots/short

# SMOKE mode: small local dry-runs (no_save)
SMOKE=true
if [ "$SMOKE" = "true" ]; then
    echo "[run_short] SMOKE mode: using small local runs (no_save)"
    TS=2000
    N_ENVS=1
    RESULTS_DIR=./results/short_smoke
    PLOTS_DIR=./plots/short_smoke
fi

echo "========================================"
echo "Starting 3 parallel training runs"
echo "env=$ENV  timesteps=$TS  n_envs=$N_ENVS  seed=$SEED"
echo "========================================"

# If running pick_place experiments, set STATE=true to use state-only variant
STATE=false
if [ "$STATE" = "true" ]; then
    STATE_ARG="--state"
else
    STATE_ARG=""
fi

python train.py \
    --method no_concept \
    --env $ENV --seed $SEED \
    $STATE_ARG \
    $( [ "$SMOKE" = "true" ] && echo "--no-save" ) \
    --total_timesteps $TS \
    --n_envs $N_ENVS \
    --n_steps $N_STEPS \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH \
    --device cuda \
    --output_dir $RESULTS_DIR &
PID0=$!

python train.py \
    --method vanilla_freeze \
    --training_mode two_phase \
    --env $ENV --seed $SEED \
    $STATE_ARG \
    $( [ "$SMOKE" = "true" ] && echo "--no-save" ) \
    --total_timesteps $TS \
    --n_envs $N_ENVS \
    --n_steps $N_STEPS \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH \
    --device cuda \
    --output_dir $RESULTS_DIR &
PID1=$!

python train.py \
    --method concept_actor_critic \
    --temporal_encoding gru \
    --training_mode two_phase \
    --env $ENV --seed $SEED \
    $STATE_ARG \
    $( [ "$SMOKE" = "true" ] && echo "--no-save" ) \
    --total_timesteps $TS \
    --n_envs $N_ENVS \
    --n_steps $N_STEPS \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH \
    --device cuda \
    --output_dir $RESULTS_DIR &
PID2=$!

python train.py \
    --method concept_actor_critic \
    --temporal_encoding none \
    --training_mode two_phase \
    --env $ENV --seed $SEED \
    $STATE_ARG \
    $( [ "$SMOKE" = "true" ] && echo "--no-save" ) \
    --total_timesteps $TS \
    --n_envs $N_ENVS \
    --n_steps $N_STEPS \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH \
    --device cuda \
    --output_dir $RESULTS_DIR &
PID3=$!

wait $PID0 $PID1 $PID2 $PID3
echo "All training runs done."

echo "========================================"
echo "Generating plots..."
echo "========================================"

python plot_results.py \
    --env $ENV \
    $STATE_ARG \
    --results_dir $RESULTS_DIR \
    --output_dir $PLOTS_DIR

echo "Complete. Plots saved to $PLOTS_DIR/"
