#!/bin/bash

set -e

# -------------------------------
# CPU safety (important!)
# -------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# -------------------------------
# Config
# -------------------------------
ENV=highway_state
TS=${TS:-100000}
N_ENVS=1
N_STEPS=512
N_EPOCHS=10
BATCH=256
SEED=42

RESULTS_DIR=results/highway_state
LOG_DIR=logs/highway_state

mkdir -p $RESULTS_DIR $LOG_DIR

# -------------------------------
# Parallelism control
# -------------------------------
MAX_JOBS=${MAX_JOBS:-3}   # set to number of CPU cores / GPUs
echo "Running with max $MAX_JOBS parallel jobs"

# -------------------------------
# Define runs
# -------------------------------
CONCEPT_NET=(
none
cbm cbm cbm cbm
concept_ac concept_ac concept_ac concept_ac
concept_ac concept_ac concept_ac concept_ac
)

TEMPORAL=(
none
gru gru none none
gru gru none none
gru gru none none
)

SUPERVISION=(
none
online online online online
online none online none
online none online none
)

FREEZE=(
none
frozen coupled frozen coupled
frozen frozen frozen frozen
coupled coupled coupled coupled
)

# -------------------------------
# Function to run one job
# -------------------------------
run_job () {
    IDX=$1

    NET=${CONCEPT_NET[$IDX]}
    TEMP=${TEMPORAL[$IDX]}
    SUP=${SUPERVISION[$IDX]}
    FRZ=${FREEZE[$IDX]}

    FREEZE_FLAG=""
    [ "$FRZ" = "frozen" ] && FREEZE_FLAG="--freeze_concept"

    TAG="${NET}_${TEMP}_${SUP}_${FRZ}_${ENV}_seed${SEED}"
    LOG_FILE="$LOG_DIR/$TAG.log"

    echo "Starting $TAG"

    if [ "$NET" = "none" ]; then
        python train.py \
            --concept_net none \
            --num_labels 0 --query_num_times 0 \
            --env $ENV --seed $SEED --total_timesteps $TS \
            --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
            --device cpu --output_dir $RESULTS_DIR \
            > $LOG_FILE 2>&1
    else
        python train.py \
            --concept_net $NET \
            --temporal $TEMP \
            --supervision $SUP \
            $FREEZE_FLAG \
            --env $ENV --seed $SEED --total_timesteps $TS \
            --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
            --device cpu --output_dir $RESULTS_DIR \
            > $LOG_FILE 2>&1
    fi

    echo "Finished $TAG"
}


# -------------------------------
# Job launcher with throttling (portable)
# -------------------------------
PIDS=()

for IDX in "${!CONCEPT_NET[@]}"; do
    run_job $IDX &
    PIDS+=($!)

    # limit concurrency
    while [ "${#PIDS[@]}" -ge "$MAX_JOBS" ]; do
        sleep 1

        NEW_PIDS=()
        for PID in "${PIDS[@]}"; do
            if kill -0 $PID 2>/dev/null; then
                NEW_PIDS+=($PID)
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
    done
done

# wait for remaining jobs
wait

echo "========================================"
echo "All experiments completed"
echo "========================================"