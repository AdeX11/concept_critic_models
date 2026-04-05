#!/bin/bash
# run_all.sh — launch 4 experiments in parallel across 4 GPUs, then plot results.
#
# Runs:
#   GPU 0: no_concept                                   (baseline)
#   GPU 1: vanilla_freeze   two_phase                   (LICORICE baseline)
#   GPU 2: concept_actor_critic  gru  two_phase         (your method)
#   GPU 3: concept_actor_critic  none two_phase         (ablation: no temporal)
#
# After all finish, plot_results.py aggregates rewards.npy files into plots/.

set -e

ENV=lunar_lander
TS=1000000
N_ENVS=16
SEEDS=(42 123 456)
RESULTS_DIR=results
PLOTS_DIR=plots

echo "========================================"
echo "Starting 4-GPU training runs"
echo "env=$ENV  timesteps=$TS  n_envs=$N_ENVS"
echo "========================================"

# ---- Wave 1: seed 42 ----
echo "[wave 1] seed=42"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --method no_concept \
    --env $ENV --seed 42 \
    --total_timesteps $TS --n_envs $N_ENVS \
    --output_dir $RESULTS_DIR &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python train.py \
    --method vanilla_freeze \
    --training_mode two_phase \
    --env $ENV --seed 42 \
    --total_timesteps $TS --n_envs $N_ENVS \
    --output_dir $RESULTS_DIR &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python train.py \
    --method concept_actor_critic \
    --temporal_encoding gru \
    --training_mode two_phase \
    --env $ENV --seed 42 \
    --total_timesteps $TS --n_envs $N_ENVS \
    --output_dir $RESULTS_DIR &
PID2=$!

CUDA_VISIBLE_DEVICES=3 python train.py \
    --method concept_actor_critic \
    --temporal_encoding none \
    --training_mode two_phase \
    --env $ENV --seed 42 \
    --total_timesteps $TS --n_envs $N_ENVS \
    --output_dir $RESULTS_DIR &
PID3=$!

wait $PID0 $PID1 $PID2 $PID3
echo "[wave 1] done."

# ---- Wave 2: seed 123 ----
echo "[wave 2] seed=123"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --method no_concept \
    --env $ENV --seed 123 \
    --total_timesteps $TS --n_envs $N_ENVS \
    --output_dir $RESULTS_DIR &

CUDA_VISIBLE_DEVICES=1 python train.py \
    --method vanilla_freeze \
    --training_mode two_phase \
    --env $ENV --seed 123 \
    --total_timesteps $TS --n_envs $N_ENVS \
    --output_dir $RESULTS_DIR &

CUDA_VISIBLE_DEVICES=2 python train.py \
    --method concept_actor_critic \
    --temporal_encoding gru \
    --training_mode two_phase \
    --env $ENV --seed 123 \
    --total_timesteps $TS --n_envs $N_ENVS \
    --output_dir $RESULTS_DIR &

CUDA_VISIBLE_DEVICES=3 python train.py \
    --method concept_actor_critic \
    --temporal_encoding none \
    --training_mode two_phase \
    --env $ENV --seed 123 \
    --total_timesteps $TS --n_envs $N_ENVS \
    --output_dir $RESULTS_DIR &

wait
echo "[wave 2] done."

# ---- Wave 3: seed 456 ----
echo "[wave 3] seed=456"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --method no_concept \
    --env $ENV --seed 456 \
    --total_timesteps $TS --n_envs $N_ENVS \
    --output_dir $RESULTS_DIR &

CUDA_VISIBLE_DEVICES=1 python train.py \
    --method vanilla_freeze \
    --training_mode two_phase \
    --env $ENV --seed 456 \
    --total_timesteps $TS --n_envs $N_ENVS \
    --output_dir $RESULTS_DIR &

CUDA_VISIBLE_DEVICES=2 python train.py \
    --method concept_actor_critic \
    --temporal_encoding gru \
    --training_mode two_phase \
    --env $ENV --seed 456 \
    --total_timesteps $TS --n_envs $N_ENVS \
    --output_dir $RESULTS_DIR &

CUDA_VISIBLE_DEVICES=3 python train.py \
    --method concept_actor_critic \
    --temporal_encoding none \
    --training_mode two_phase \
    --env $ENV --seed 456 \
    --total_timesteps $TS --n_envs $N_ENVS \
    --output_dir $RESULTS_DIR &

wait
echo "[wave 3] done."

# ---- Aggregate and plot ----
echo ""
echo "========================================"
echo "All training done. Generating plots..."
echo "========================================"

python plot_results.py \
    --env $ENV \
    --results_dir $RESULTS_DIR \
    --output_dir $PLOTS_DIR

echo ""
echo "========================================"
echo "Complete. Plots saved to $PLOTS_DIR/"
echo "========================================"
