#!/bin/bash
# run_e2e_comparison.sh — Compare end_to_end concept_actor_critic against baselines.
#
# Runs 4 methods in parallel on lunar_lander_state:
#   0. no_concept                              — plain PPO baseline
#   1. vanilla_freeze  (two_phase)             — LICORICE supervised CBM
#   2. concept_actor_critic (end_to_end, none) — new method, no temporal encoding
#   3. concept_actor_critic (end_to_end, gru)  — new method, GRU temporal encoding

set -e

ENV=lunar_lander_state
TS=1000000
N_ENVS=8
N_STEPS=512
N_EPOCHS=10
BATCH=256
SEED=42
RESULTS_DIR=/glade/derecho/scratch/adadelek/results
PLOTS_DIR=plots
VENV=/glade/derecho/scratch/adadelek/venv

source $VENV/bin/activate

echo "========================================"
echo "Starting 4 parallel training runs (end_to_end comparison)"
echo "env=$ENV  timesteps=$TS  n_envs=$N_ENVS  seed=$SEED"
echo "========================================"

python train.py \
    --method no_concept \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cuda \
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
    --device cuda \
    --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/vanilla_freeze_${ENV}_seed${SEED}.log 2>&1 &
PID1=$!
echo "[PID $PID1] vanilla_freeze (two_phase)"

python train.py \
    --method concept_actor_critic \
    --temporal_encoding none \
    --training_mode end_to_end \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cuda \
    --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/concept_ac_e2e_none_${ENV}_seed${SEED}.log 2>&1 &
PID2=$!
echo "[PID $PID2] concept_actor_critic (end_to_end, none)"

python train.py \
    --method concept_actor_critic \
    --temporal_encoding gru \
    --training_mode end_to_end \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cuda \
    --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/concept_ac_e2e_gru_${ENV}_seed${SEED}.log 2>&1 &
PID3=$!
echo "[PID $PID3] concept_actor_critic (end_to_end, gru)"

echo "========================================"
echo "Waiting for all runs to finish..."
echo "Tail logs with: tail -f $RESULTS_DIR/<name>.log"
echo "========================================"

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
