#!/bin/bash
# run_hidden_vel_comparison.sh — Compare all methods on HiddenVelocityEnv.
#
# This env is designed so temporal concepts (x_vel, y_vel, approaching_goal)
# are unrecoverable from a single frame — vanilla CBM must fail on them.
#
# Runs 4 methods in parallel:
#   0. no_concept                              — plain PPO baseline
#   1. vanilla_freeze  (two_phase)             — LICORICE: should nail static concepts, fail temporal
#   2. concept_actor_critic (end_to_end, none) — AC signal, no temporal encoding: should also fail temporal
#   3. concept_actor_critic (end_to_end, gru)  — AC signal + GRU: should predict temporal concepts well

set -e

ENV=hidden_velocity
TS=500000
N_ENVS=8
N_STEPS=512
N_EPOCHS=10
BATCH=256
SEED=42
RESULTS_DIR=/glade/derecho/scratch/adadelek/results
PLOTS_DIR=plots/hidden_velocity
VENV=/glade/derecho/scratch/adadelek/venv

source $VENV/bin/activate
mkdir -p $PLOTS_DIR

echo "========================================"
echo "HiddenVelocityEnv comparison"
echo "env=$ENV  timesteps=$TS  n_envs=$N_ENVS  seed=$SEED"
echo "========================================"

python train.py \
    --method no_concept \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cpu \
    --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/no_concept_${ENV}_seed${SEED}.log 2>&1 &
PID0=$!
echo "[PID $PID0] no_concept"

python train.py \
    --method vanilla_freeze \
    --training_mode two_phase \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cpu \
    --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/vanilla_freeze_${ENV}_seed${SEED}.log 2>&1 &
PID1=$!
echo "[PID $PID1] vanilla_freeze (two_phase)"

python train.py \
    --method concept_actor_critic \
    --temporal_encoding none \
    --training_mode two_phase \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cpu \
    --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/concept_ac_2p_none_${ENV}_seed${SEED}.log 2>&1 &
PID2=$!
echo "[PID $PID2] concept_actor_critic (end_to_end, none)"

python train.py \
    --method concept_actor_critic \
    --temporal_encoding gru \
    --training_mode two_phase \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cpu \
    --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/concept_ac_2p_gru_${ENV}_seed${SEED}.log 2>&1 &
PID3=$!
echo "[PID $PID3] concept_actor_critic (end_to_end, gru)"

echo "========================================"
echo "Waiting for all runs to finish..."
echo "Monitor with: tail -f $RESULTS_DIR/<name>.log"
echo "========================================"

wait $PID0 $PID1 $PID2 $PID3
echo "All training runs done."

python plot_results.py \
    --env $ENV \
    --results_dir $RESULTS_DIR \
    --output_dir $PLOTS_DIR

echo "Complete. Plots saved to $PLOTS_DIR/"
