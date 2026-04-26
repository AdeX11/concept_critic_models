#!/bin/bash
# Methods: PPO, Vanilla CBM, Concept Actor-Critic, GVF
# Config: End-to-End, No Temporal Encoding

set -e

ENV="pick_place" 
STATE=false  

if [ "$ENV" = "highway" ]; then ENV_SHORT="hw"; else ENV_SHORT="pp"; fi
if [ "$STATE" = "true" ]; then STATE_SHORT="state"; STATE_ARG="--state"; else STATE_SHORT="pixels"; STATE_ARG=""; fi

RUN_NAME="${ENV_SHORT}_${STATE_SHORT}"
RESULTS_DIR="results/none_joint/$RUN_NAME"
PLOTS_DIR="plots/none_joint/$RUN_NAME"

TS=75000
N_ENVS=6
SEED=42

echo "========================================"
echo "Starting 4 parallel training runs (CPU only)"
echo "Config: $RUN_NAME | env=$ENV | seed=$SEED"
echo "========================================"

python train.py \
    --method no_concept \
    --env $ENV --seed $SEED \
    $STATE_ARG \
    --total_timesteps $TS --n_envs $N_ENVS \
    --device cpu \
    --output_dir $RESULTS_DIR &
PID0=$!

python train.py \
    --method vanilla_freeze \
    --training_mode joint \
    --temporal_encoding none \
    --env $ENV --seed $SEED \
    $STATE_ARG \
    --total_timesteps $TS --n_envs $N_ENVS \
    --device cpu \
    --output_dir $RESULTS_DIR &
PID1=$!

python train.py \
    --method concept_actor_critic \
    --training_mode joint \
    --temporal_encoding none \
    --env $ENV --seed $SEED \
    $STATE_ARG \
    --total_timesteps $TS --n_envs $N_ENVS \
    --device cpu \
    --output_dir $RESULTS_DIR &
PID2=$!

python train.py \
    --method gvf \
    --training_mode joint \
    --temporal_encoding none \
    --env $ENV --seed $SEED \
    $STATE_ARG \
    --total_timesteps $TS --n_envs $N_ENVS \
    --device cpu \
    --output_dir $RESULTS_DIR &
PID3=$!

wait $PID0 $PID1 $PID2 $PID3
echo "Training done."

python plot_results.py --env $ENV $STATE_ARG --results_dir $RESULTS_DIR --output_dir $PLOTS_DIR
echo "Complete. Plots saved to $PLOTS_DIR/"

echo "========================================"
echo "Generating replay visualizations..."
echo "========================================"

REPLAY_DIR="$RESULTS_DIR/replays"
mkdir -p "$REPLAY_DIR"

# Directory names follow train.py convention:
# <method>_<training_mode>_<temporal_encoding>_<env_tag>_seed<seed>
NO_CONCEPT_DIR="$RESULTS_DIR/no_concept_two_phase_none_${ENV}_seed${SEED}"
VANILLA_DIR="$RESULTS_DIR/vanilla_freeze_joint_none_${ENV}_seed${SEED}"
CAC_DIR="$RESULTS_DIR/concept_actor_critic_joint_none_${ENV}_seed${SEED}"
GVF_DIR="$RESULTS_DIR/gvf_joint_none_${ENV}_seed${SEED}"

python replay.py \
    --env $ENV --method no_concept \
    --model_path "$NO_CONCEPT_DIR/model.pt" \
    --temporal_encoding none \
    --output_gif "$REPLAY_DIR/no_concept_replay.gif" \
    --seed $SEED --episodes 1 --max_steps 300 --fps 6 --deterministic &
R0=$!

python replay.py \
    --env $ENV --method vanilla_freeze \
    --model_path "$VANILLA_DIR/model.pt" \
    --temporal_encoding none \
    --output_gif "$REPLAY_DIR/vanilla_freeze_replay.gif" \
    --seed $SEED --episodes 1 --max_steps 300 --fps 6 --deterministic &
R1=$!

python replay.py \
    --env $ENV --method concept_actor_critic \
    --model_path "$CAC_DIR/model.pt" \
    --temporal_encoding none \
    --output_gif "$REPLAY_DIR/concept_actor_critic_replay.gif" \
    --seed $SEED --episodes 1 --max_steps 300 --fps 6 --deterministic &
R2=$!

python replay.py \
    --env $ENV --method gvf \
    --model_path "$GVF_DIR/model.pt" \
    --temporal_encoding none \
    --gvf_pairing 0 1 2 3 4 5 6 7 8 9 10 \
    --output_gif "$REPLAY_DIR/gvf_replay.gif" \
    --seed $SEED --episodes 1 --max_steps 300 --fps 6 --deterministic &
R3=$!

wait $R0 $R1 $R2 $R3
echo "Replay GIFs saved to $REPLAY_DIR/"
