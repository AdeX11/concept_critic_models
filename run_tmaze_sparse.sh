#!/bin/bash
#PBS -A NAML0001
#PBS -q casper
#PBS -l select=1:ncpus=36:mem=128gb
#PBS -l walltime=08:00:00
#PBS -N run_tmaze_sparse
#PBS -j oe
#PBS -o /glade/derecho/scratch/adadelek/results/tmaze_sparse/run_tmaze_sparse_pbs.log

cd /glade/u/home/adadelek/concept_critic_models

# run_tmaze_sparse.sh — Same 13-run sweep as run_tmaze_full.sh but with
# reward_mode=sparse: only +1.0 at the junction for the correct choice,
# 0.0 everywhere else (no step penalties, no premature penalty, no -1 for wrong).
#
# Purpose: test whether the concept AC reward provides useful shaping when
# task reward is maximally sparse.  Hypothesis: concept_ac+GRU+online should
# retain an advantage over CBM+GRU+online that was invisible under dense rewards.

set -e

ENV=tmaze
TS=2000000
N_ENVS=8
N_STEPS=16
N_EPOCHS=10
BATCH=256
SEED=42
RESULTS_DIR=/glade/derecho/scratch/adadelek/results/tmaze_sparse
PLOTS_DIR=plots/tmaze_sparse
VENV=/glade/derecho/scratch/adadelek/venv

source $VENV/bin/activate
mkdir -p $RESULTS_DIR $PLOTS_DIR

PIDS=()

echo "========================================"
echo "TMazeEnv sparse-reward sweep (13 runs)"
echo "env=$ENV  reward_mode=sparse  timesteps=$TS  n_envs=$N_ENVS  seed=$SEED"
echo "========================================"

# ---------------------------------------------------------------------------
# 0. none — pure PPO baseline
# ---------------------------------------------------------------------------
python train.py --concept_net none \
    --num_labels 0 --query_num_times 0 \
    --env $ENV --reward_mode sparse --seed $SEED --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cpu --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/none_${ENV}_seed${SEED}.log 2>&1 &
PIDS+=($!)
echo "[PID ${PIDS[-1]}] none"

# ---------------------------------------------------------------------------
# cbm × {gru, none} × {online} × {frozen, coupled}
# ---------------------------------------------------------------------------
for TEMPORAL in gru none; do
for FREEZE_FLAG in "--freeze_concept" ""; do

    FREEZE_STR=$([ "$FREEZE_FLAG" = "--freeze_concept" ] && echo "frozen" || echo "coupled")
    TAG="cbm_${TEMPORAL}_online_${FREEZE_STR}_${ENV}_seed${SEED}"

    python train.py \
        --concept_net cbm \
        --temporal $TEMPORAL --supervision online $FREEZE_FLAG \
        --num_labels 0 --query_num_times 0 \
        --env $ENV --reward_mode sparse --seed $SEED --total_timesteps $TS \
        --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
        --device cpu --output_dir $RESULTS_DIR \
        > $RESULTS_DIR/${TAG}.log 2>&1 &
    PIDS+=($!)
    echo "[PID ${PIDS[-1]}] cbm  $TEMPORAL  online  $FREEZE_STR"

done
done

# ---------------------------------------------------------------------------
# concept_ac × {gru, none} × {online, none} × {frozen, coupled}
# ---------------------------------------------------------------------------
for TEMPORAL in gru none; do
for SUPERVISION in online none; do
for FREEZE_FLAG in "--freeze_concept" ""; do

    FREEZE_STR=$([ "$FREEZE_FLAG" = "--freeze_concept" ] && echo "frozen" || echo "coupled")
    TAG="concept_ac_${TEMPORAL}_${SUPERVISION}_${FREEZE_STR}_${ENV}_seed${SEED}"

    python train.py \
        --concept_net concept_ac \
        --temporal $TEMPORAL --supervision $SUPERVISION $FREEZE_FLAG \
        --env $ENV --reward_mode sparse --seed $SEED --total_timesteps $TS \
        --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
        --device cpu --output_dir $RESULTS_DIR \
        > $RESULTS_DIR/${TAG}.log 2>&1 &
    PIDS+=($!)
    echo "[PID ${PIDS[-1]}] concept_ac  $TEMPORAL  $SUPERVISION  $FREEZE_STR"

done
done
done

echo "========================================"
echo "Waiting for all ${#PIDS[@]} runs to finish..."
echo "Monitor with: tail -f $RESULTS_DIR/<tag>.log"
echo "========================================"

wait "${PIDS[@]}"
echo "All training runs done."

python plot_results.py \
    --env $ENV \
    --results_dir $RESULTS_DIR \
    --output_dir $PLOTS_DIR

echo "Complete. Plots saved to $PLOTS_DIR/"
