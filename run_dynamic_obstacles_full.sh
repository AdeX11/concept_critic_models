#!/bin/bash
#PBS -A NAML0001
#PBS -q casper
#PBS -l select=1:ncpus=64:ngpus=1:mem=256gb
#PBS -l walltime=12:00:00
#PBS -N run_dynamic_obstacles_full
#PBS -j oe
#PBS -o /glade/derecho/scratch/adadelek/results/dynamic_obstacles_full/run_dynamic_obstacles_full_pbs.log

cd /glade/u/home/adadelek/concept_critic_models

# run_dynamic_obstacles_full.sh — Architecture + technique sweep on DynamicObstacles.
#
# Key question: does the concept AC reward provide an advantage over CBM
# when concept reward fires densely (every step, all 13 concepts)?
#
# 1  none          (pure PPO baseline)
# 4  cbm        × {gru, none} × {online}       × {frozen, coupled}
# 8  concept_ac × {gru, none} × {online, none} × {frozen, coupled}
# ── total: 13 runs

set -e

ENV=dynamic_obstacles
TS=5000000
N_ENVS=4
N_STEPS=512
N_EPOCHS=10
BATCH=256
SEED=42
RESULTS_DIR=/glade/derecho/scratch/adadelek/results/dynamic_obstacles_full
PLOTS_DIR=plots/dynamic_obstacles_full
VENV=/glade/derecho/scratch/adadelek/venv

source $VENV/bin/activate
mkdir -p $RESULTS_DIR $PLOTS_DIR

PIDS=()

echo "========================================"
echo "DynamicObstacles architecture sweep (13 runs)"
echo "env=$ENV  timesteps=$TS  n_envs=$N_ENVS  seed=$SEED"
echo "========================================"

# ---------------------------------------------------------------------------
# 0. none — pure PPO baseline
# ---------------------------------------------------------------------------
python train.py --concept_net none \
    --num_labels 0 --query_num_times 0 \
    --env $ENV --seed $SEED --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device auto --output_dir $RESULTS_DIR \
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
        --env $ENV --seed $SEED --total_timesteps $TS \
        --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
        --device auto --output_dir $RESULTS_DIR \
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
        --env $ENV --seed $SEED --total_timesteps $TS \
        --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
        --device auto --output_dir $RESULTS_DIR \
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
