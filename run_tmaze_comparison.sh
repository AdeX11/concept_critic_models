#!/bin/bash
# run_tmaze_comparison.sh — Compare 4 approaches on TMazeEnv.
#
# T-Maze requires memory: a cue shown at positions 0-2 must be recalled
# at the junction (position 10).  With n_stack=4, the cue falls out of
# the frame window by position 6, so stacked envs cannot read the cue
# at decision time from the observation alone.
#
# Runs 6 methods in parallel:
#   1. vanilla_freeze  two_phase  stacked — LICORICE CBM; dense supervised
#                                           labels (2000 over 4 queries);
#                                           cue gone from stack at junction
#                                           → expect ~50% cue accuracy
#   2. vanilla_freeze  joint      stacked — CBM with per-iteration rollout
#                                           supervision; same observational
#                                           limit as above
#   3. concept_actor_critic  joint  gru   — AC signal + GRU hidden state;
#                                           only temporal mechanism that can
#                                           carry cue to junction
#   4. concept_actor_critic  joint  stacked  — AC signal, no GRU; ablation
#   5. concept_actor_critic  two_phase  gru  — GRU but concept net frozen
#                                              during PPO update
#   6. concept_actor_critic  two_phase  stacked — double ablation: no GRU,
#                                                  concept net frozen

set -e

ENV=tmaze
TS=500000
N_ENVS=8
N_STEPS=512
N_EPOCHS=10
BATCH=256
SEED=42
RESULTS_DIR=/glade/derecho/scratch/adadelek/results
PLOTS_DIR=plots/tmaze
VENV=/glade/derecho/scratch/adadelek/venv

source $VENV/bin/activate
mkdir -p $PLOTS_DIR

echo "========================================"
echo "TMazeEnv comparison (6 runs in parallel)"
echo "env=$ENV  timesteps=$TS  n_envs=$N_ENVS  seed=$SEED"
echo "========================================"

# 1. vanilla_freeze  two_phase  stacked
#    Dense supervision: 2000 labels across 4 queries (one every 125k steps).
#    Concept net frozen during PPO update; only updated at query times.
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
#    Per-iteration rollout supervision — no need for large label budget.
#    Concept net updated every PPO iteration from rollout buffer concepts.
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

# 3. concept_actor_critic  joint  gru
#    AC reward signal drives GRU to maintain cue across 7 blank steps.
#    This is the only configuration that can carry the cue to the junction.
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
PID3=$!
echo "[PID $PID3] concept_actor_critic  joint  gru"

# 4. concept_actor_critic  joint  stacked
#    Ablation: AC signal + stacked obs, no GRU.
#    Cue is gone from the window at the junction — AC signal alone
#    cannot recover absent information.
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
PID4=$!
echo "[PID $PID4] concept_actor_critic  joint  stacked  (ablation)"

# 5. concept_actor_critic  two_phase  gru
#    Concept net frozen during PPO update; AC + supervised anchor update it
#    separately.  GRU can still carry the cue — tests whether two_phase
#    training hurts the GRU's ability to maintain temporal state vs joint.
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
PID5=$!
echo "[PID $PID5] concept_actor_critic  two_phase  gru"

# 6. concept_actor_critic  two_phase  stacked
#    Concept net frozen during PPO update; no GRU; cue gone from window.
#    Double ablation: no temporal mechanism and no joint gradient flow.
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
PID6=$!
echo "[PID $PID6] concept_actor_critic  two_phase  stacked  (double ablation)"

echo "========================================"
echo "Waiting for all 6 runs to finish..."
echo "Monitor with:"
echo "  tail -f $RESULTS_DIR/vanilla_freeze_two_phase_stacked_${ENV}_seed${SEED}.log"
echo "  tail -f $RESULTS_DIR/vanilla_freeze_joint_stacked_${ENV}_seed${SEED}.log"
echo "  tail -f $RESULTS_DIR/concept_actor_critic_joint_gru_${ENV}_seed${SEED}.log"
echo "  tail -f $RESULTS_DIR/concept_actor_critic_joint_stacked_${ENV}_seed${SEED}.log"
echo "  tail -f $RESULTS_DIR/concept_actor_critic_two_phase_gru_${ENV}_seed${SEED}.log"
echo "  tail -f $RESULTS_DIR/concept_actor_critic_two_phase_stacked_${ENV}_seed${SEED}.log"
echo "========================================"

wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6
echo "All training runs done."

python plot_results.py \
    --env $ENV \
    --results_dir $RESULTS_DIR \
    --output_dir $PLOTS_DIR

echo "Complete. Plots saved to $PLOTS_DIR/"
