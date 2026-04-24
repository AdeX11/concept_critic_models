#!/bin/bash
# Methods: PPO, Vanilla CBM, Concept Actor-Critic, GVF
# Config: End-to-End, No Temporal Encoding

set -e

ENV="pick_place" 
STATE=true    

if [ "$ENV" = "highway" ]; then ENV_SHORT="hw"; else ENV_SHORT="pp"; fi
if [ "$STATE" = "true" ]; then STATE_SHORT="state"; STATE_ARG="--state"; else STATE_SHORT="pixels"; STATE_ARG=""; fi

RUN_NAME="${ENV_SHORT}_${STATE_SHORT}"
RESULTS_DIR="results/none_joint/$RUN_NAME"
PLOTS_DIR="plots/none_joint/$RUN_NAME"

TS=100000
N_ENVS=6
SEED=42

echo "========================================"
echo "Starting 4 parallel training runs (CPU only)"
echo "Config: $RUN_NAME | env=$ENV | seed=$SEED"
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
    --training_mode joint \
    --temporal_encoding none \
    --env $ENV --seed $SEED \
    $STATE_ARG \
    --total_timesteps $TS --n_envs $N_ENVS \
    --device cpu \
    --output_dir $RESULTS_DIR &
PID1=$!

python train.py \
    --method concept_actor_critic \
    --training_mode joint \
    --temporal_encoding none \
    --env $ENV --seed $SEED \
    $STATE_ARG \
    --total_timesteps $TS --n_envs $N_ENVS \
    --device cpu \
    --output_dir $RESULTS_DIR &
PID2=$!

# python train.py \
#     --method concept_actor_critic \
#     --training_mode joint \
#     --temporal_encoding none \
#     --env $ENV --seed $SEED \
#     $STATE_ARG \
#     --total_timesteps $TS --n_envs $N_ENVS \
#     --device cpu \
#     --output_dir $RESULTS_DIR &
# PID3=$!

wait $PID0 $PID1 $PID2
echo "Training done."

python plot_results.py --env $ENV $STATE_ARG --results_dir $RESULTS_DIR --output_dir $PLOTS_DIR
echo "Complete. Plots saved to $PLOTS_DIR/"