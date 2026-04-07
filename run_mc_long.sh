#!/bin/bash
# run_mc_long.sh — long MountainCar comparison
#   Methods: no_concept | vanilla_freeze | concept_actor_critic (joint)
#   All trained for 2M timesteps, 8 envs, seed 42
#
# Usage: bash run_mc_long.sh

set -e

ENV=mountain_car
TS=2000000
N_ENVS=8
SEED=42
RESULTS_DIR=/glade/derecho/scratch/adadelek/results/mc_long
PLOTS_DIR=plots/mc_long

mkdir -p $RESULTS_DIR $PLOTS_DIR

echo "========================================"
echo "MountainCar (long): No Concept vs Vanilla Freeze vs Concept AC (joint)"
echo "env=$ENV  timesteps=$TS  n_envs=$N_ENVS  seed=$SEED"
echo "========================================"

python train.py \
    --method no_concept \
    --env $ENV --seed $SEED \
    --total_timesteps $TS --n_envs $N_ENVS \
    --device cuda \
    --output_dir $RESULTS_DIR &
PID0=$!
echo "no_concept PID=$PID0"

python train.py \
    --method vanilla_freeze \
    --training_mode two_phase \
    --query_num_times 1 \
    --env $ENV --seed $SEED \
    --total_timesteps $TS --n_envs $N_ENVS \
    --device cuda \
    --output_dir $RESULTS_DIR &
PID1=$!
echo "vanilla_freeze PID=$PID1"

python train.py \
    --method concept_actor_critic \
    --temporal_encoding none \
    --training_mode joint \
    --query_num_times 1 \
    --env $ENV --seed $SEED \
    --total_timesteps $TS --n_envs $N_ENVS \
    --device cuda \
    --output_dir $RESULTS_DIR &
PID2=$!
echo "concept_ac_joint PID=$PID2"

wait $PID0 $PID1 $PID2
echo ""
echo "========================================"
echo "Training done. Generating plots..."
echo "========================================"

python plot_results.py \
    --env $ENV \
    --results_dir $RESULTS_DIR \
    --output_dir $PLOTS_DIR

echo ""
echo "Done. Results in $RESULTS_DIR  Plots in $PLOTS_DIR"
