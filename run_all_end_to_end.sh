#!/bin/bash
# run_all.sh — launch 4 experiments in parallel sharing 1 GPU, then plot results.
#
# Methods:
#   0. no_concept                         — plain PPO baseline
#   1. vanilla_freeze                     — LICORICE supervised CBM
#   2. concept_actor_critic (stacked)         — new method with temporal encoding
#   3. concept_actor_critic (none)        — new method without temporal encoding (ablation)

set -e

ENV=highway
# For pick_place runs you can set STATE=true to use the state-only variant (no rendering/images)
STATE=true


# Build optional STATE_ARG passed to train.py
if [ "$STATE" = "true" ]; then
    STATE_ARG="--state"
else
    STATE_ARG=""
fi

# Defaults for full experiments
TS=75000
N_ENVS=2
SEED=42
RESULTS_DIR=results/run_all_hw_state_end
PLOTS_DIR=plots/run_all_hw_state_end


echo "========================================"
echo "Starting 4 parallel training runs (shared GPU)"
echo "env=$ENV  timesteps=$TS  n_envs=$N_ENVS  seed=$SEED"
echo "========================================"

python train.py \
    --method no_concept \
    --env $ENV --seed $SEED \
    $STATE_ARG \
    --total_timesteps $TS --n_envs $N_ENVS \
    --device cpu \
    --output_dir $RESULTS_DIR &
PID0=$!

python train.py \
    --method vanilla_freeze \
    --training_mode end_to_end \
    --temporal_encoding stacked \
    --env $ENV --seed $SEED \
    $STATE_ARG \
    --total_timesteps $TS --n_envs $N_ENVS \
    --device cpu \
    --output_dir $RESULTS_DIR &
PID1=$!

python train.py \
    --method concept_actor_critic \
    --temporal_encoding stacked \
    --training_mode end_to_end \
    --env $ENV --seed $SEED \
    $STATE_ARG \
    --total_timesteps $TS --n_envs $N_ENVS \
    --device cpu \
    --output_dir $RESULTS_DIR &
PID2=$!

python train.py \
    --method concept_actor_critic \
    --temporal_encoding none \
    --training_mode end_to_end \
    --env $ENV --seed $SEED \
    $STATE_ARG \
    --total_timesteps $TS --n_envs $N_ENVS \
    --device cpu \
    --output_dir $RESULTS_DIR &
PID3=$!

wait $PID0 $PID1 $PID2 $PID3
echo "Training done."

echo "========================================"
echo "Generating plots..."
echo "========================================"

python plot_results.py \
    --env $ENV \
    $STATE_ARG \
    --results_dir $RESULTS_DIR \
    --output_dir $PLOTS_DIR

echo "Complete. Plots saved to $PLOTS_DIR/"