#!/bin/bash
# run_short.sh — Short parallel comparison of all 3 methods on dynamic_obstacles (8x8).
#
# Methods compared:
#   1. no_concept                        — plain PPO baseline
#   2. vanilla_freeze (two_phase)        — LICORICE supervised CBM
#   3. concept_actor_critic (gru)        — new method with temporal encoding
#   4. concept_actor_critic (none)       — ablation: no temporal encoding

set -e

ENV=dynamic_obstacles
TS=200000
N_ENVS=4
N_STEPS=256
N_EPOCHS=5
BATCH=128
SEED=42
RESULTS_DIR=/glade/derecho/scratch/adadelek/results
PLOTS_DIR=plots

echo "========================================"
echo "Starting 3 parallel training runs"
echo "env=$ENV  timesteps=$TS  n_envs=$N_ENVS  seed=$SEED"
echo "========================================"

python train.py \
    --method no_concept \
    --env $ENV --seed $SEED \
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
    --results_dir $RESULTS_DIR \
    --output_dir $PLOTS_DIR

echo "Complete. Plots saved to $PLOTS_DIR/"
