#!/bin/bash
#PBS -A NAML0001
#PBS -q casper
#PBS -l select=1:ncpus=16:mem=64gb
#PBS -l walltime=05:00:00
#PBS -N run_tmaze
#PBS -j oe
#PBS -o /glade/derecho/scratch/adadelek/results/run_tmaze_pbs.log

cd /glade/u/home/adadelek/concept_critic_models
# run_tmaze.sh — Compare 9 approaches on TMazeEnv.
#
# T-Maze requires memory: a cue shown at positions 0-2 must be recalled
# at the junction (position 10).  With n_stack=4, the cue falls out of
# the frame window by position 6, so stacked envs cannot read the cue
# at decision time from the observation alone.
#
# Runs 9 methods in parallel:
#   0. no_concept                          — pure PPO; no concept net, no GRU
#   1. vanilla_freeze  two_phase  stacked  — LICORICE CBM; 2000 labels, 4 queries
#   2. vanilla_freeze  joint      stacked  — CBM, per-iteration supervision
#   3. vanilla_freeze  joint      gru      — GRU CBM + BPTT; fair arch match for 5
#   4. vanilla_freeze  two_phase  gru      — GRU CBM, concept net frozen during PPO
#   5. concept_actor_critic  joint  gru    — AC signal + GRU (hypothesis: best)
#   6. concept_actor_critic  joint  stacked  — AC signal, no GRU; ablation
#   7. concept_actor_critic  two_phase  gru  — GRU, concept net frozen during PPO
#   8. concept_actor_critic  two_phase  stacked — double ablation

set -e

ENV=tmaze
TS=2000000
N_ENVS=8
N_STEPS=16
N_EPOCHS=10
BATCH=256
SEED=42
RESULTS_DIR=/glade/derecho/scratch/adadelek/results
PLOTS_DIR=plots/tmaze
VENV=/glade/derecho/scratch/adadelek/venv

source $VENV/bin/activate
mkdir -p $PLOTS_DIR

echo "========================================"
echo "TMazeEnv comparison (9 runs in parallel)"
echo "env=$ENV  timesteps=$TS  n_envs=$N_ENVS  seed=$SEED"
echo "========================================"

# 0. no_concept — pure PPO; no concept bottleneck, no GRU, no memory
python train.py \
    --method no_concept \
    --training_mode two_phase \
    --temporal_encoding none \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --num_labels 0 --query_num_times 0 \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cpu \
    --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/no_concept_${ENV}_seed${SEED}.log 2>&1 &
PID0=$!
echo "[PID $PID0] no_concept  (pure PPO baseline)"

# 1. vanilla_freeze  two_phase  stacked
#    Dense supervision: 2000 labels across 4 queries.
python train.py \
    --method vanilla_freeze \
    --training_mode two_phase \
    --temporal_encoding stacked \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --num_labels 2000 --query_num_times 4 \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cpu \
    --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/vanilla_freeze_two_phase_stacked_${ENV}_seed${SEED}.log 2>&1 &
PID1=$!
echo "[PID $PID1] vanilla_freeze  two_phase  stacked  (2000 labels, 4 queries)"

# 2. vanilla_freeze  joint  stacked
#    Per-iteration rollout supervision, no GRU.
python train.py \
    --method vanilla_freeze \
    --training_mode joint \
    --temporal_encoding stacked \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --num_labels 500 --query_num_times 1 \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cpu \
    --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/vanilla_freeze_joint_stacked_${ENV}_seed${SEED}.log 2>&1 &
PID2=$!
echo "[PID $PID2] vanilla_freeze  joint      stacked"

# 3. vanilla_freeze  joint  gru
#    Same as 2 but with GRU + BPTT.  Architecturally identical to run 5.
#    Key comparison: 3 vs 5 isolates the training signal (supervised CE vs AC reward).
python train.py \
    --method vanilla_freeze \
    --training_mode joint \
    --temporal_encoding gru \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --num_labels 500 --query_num_times 1 \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cpu \
    --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/vanilla_freeze_joint_gru_${ENV}_seed${SEED}.log 2>&1 &
PID3=$!
echo "[PID $PID3] vanilla_freeze  joint      gru  (fair architecture baseline)"

# 4. vanilla_freeze  two_phase  gru
#    GRU CBM, concept net frozen during PPO update.
python train.py \
    --method vanilla_freeze \
    --training_mode two_phase \
    --temporal_encoding gru \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --num_labels 500 --query_num_times 1 \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cpu \
    --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/vanilla_freeze_two_phase_gru_${ENV}_seed${SEED}.log 2>&1 &
PID4=$!
echo "[PID $PID4] vanilla_freeze  two_phase  gru"

# 5. concept_actor_critic  joint  gru
#    AC reward signal + GRU + BPTT.  Hypothesis: best overall.
python train.py \
    --method concept_actor_critic \
    --training_mode joint \
    --temporal_encoding gru \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --num_labels 500 --query_num_times 1 \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cpu \
    --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/concept_actor_critic_joint_gru_${ENV}_seed${SEED}.log 2>&1 &
PID5=$!
echo "[PID $PID5] concept_actor_critic  joint  gru"

# 6. concept_actor_critic  joint  stacked
#    AC signal, no GRU; cue gone from window at junction.
python train.py \
    --method concept_actor_critic \
    --training_mode joint \
    --temporal_encoding stacked \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --num_labels 500 --query_num_times 1 \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cpu \
    --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/concept_actor_critic_joint_stacked_${ENV}_seed${SEED}.log 2>&1 &
PID6=$!
echo "[PID $PID6] concept_actor_critic  joint  stacked  (ablation)"

# 7. concept_actor_critic  two_phase  gru
#    GRU, concept net frozen during PPO update.
python train.py \
    --method concept_actor_critic \
    --training_mode two_phase \
    --temporal_encoding gru \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --num_labels 500 --query_num_times 1 \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cpu \
    --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/concept_actor_critic_two_phase_gru_${ENV}_seed${SEED}.log 2>&1 &
PID7=$!
echo "[PID $PID7] concept_actor_critic  two_phase  gru"

# 8. concept_actor_critic  two_phase  stacked
#    Double ablation: no GRU, concept net frozen during PPO.
python train.py \
    --method concept_actor_critic \
    --training_mode two_phase \
    --temporal_encoding stacked \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --num_labels 500 --query_num_times 1 \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cpu \
    --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/concept_actor_critic_two_phase_stacked_${ENV}_seed${SEED}.log 2>&1 &
PID8=$!
echo "[PID $PID8] concept_actor_critic  two_phase  stacked  (double ablation)"

echo "========================================"
echo "Waiting for all 9 runs to finish..."
echo "Monitor with:"
echo "  tail -f $RESULTS_DIR/no_concept_${ENV}_seed${SEED}.log"
echo "  tail -f $RESULTS_DIR/vanilla_freeze_two_phase_stacked_${ENV}_seed${SEED}.log"
echo "  tail -f $RESULTS_DIR/vanilla_freeze_joint_stacked_${ENV}_seed${SEED}.log"
echo "  tail -f $RESULTS_DIR/vanilla_freeze_joint_gru_${ENV}_seed${SEED}.log"
echo "  tail -f $RESULTS_DIR/vanilla_freeze_two_phase_gru_${ENV}_seed${SEED}.log"
echo "  tail -f $RESULTS_DIR/concept_actor_critic_joint_gru_${ENV}_seed${SEED}.log"
echo "  tail -f $RESULTS_DIR/concept_actor_critic_joint_stacked_${ENV}_seed${SEED}.log"
echo "  tail -f $RESULTS_DIR/concept_actor_critic_two_phase_gru_${ENV}_seed${SEED}.log"
echo "  tail -f $RESULTS_DIR/concept_actor_critic_two_phase_stacked_${ENV}_seed${SEED}.log"
echo "========================================"

wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8
echo "All training runs done."

python plot_results.py \
    --env $ENV \
    --results_dir $RESULTS_DIR \
    --output_dir $PLOTS_DIR

echo "Complete. Plots saved to $PLOTS_DIR/"
