#!/bin/bash
# run_both_envs_comparison.sh — Run all 4 methods on both environments in parallel.
#
# lunar_lander_state:  temporal concepts (velocity) are correlated with static ones
#                      → vanilla CBM can cheat by inferring velocity from position
#
# hidden_velocity:     temporal concepts are genuinely unobservable from a single frame
#                      → vanilla CBM must fail; only concept_actor_critic (gru) can succeed
#
# Running both side-by-side makes the contrast visible in the concept accuracy plots.
#
# 8 total runs launched in parallel (4 per env). All share the GPU.

set -e

TS=500000
N_ENVS=8
N_STEPS=512
N_EPOCHS=10
BATCH=256
SEED=42
RESULTS_DIR=/glade/derecho/scratch/adadelek/results
PLOTS_DIR_LL=plots/lunar_lander_state
PLOTS_DIR_HV=plots/hidden_velocity
VENV=/glade/derecho/scratch/adadelek/venv
LOG=$RESULTS_DIR/both_envs_comparison.log

source $VENV/bin/activate
mkdir -p $PLOTS_DIR_LL $PLOTS_DIR_HV

echo "========================================" | tee $LOG
echo "Launching 8 runs (4 per env) in parallel" | tee -a $LOG
echo "lunar_lander_state + hidden_velocity" | tee -a $LOG
echo "timesteps=$TS  n_envs=$N_ENVS  seed=$SEED" | tee -a $LOG
echo "========================================" | tee -a $LOG

# ---- lunar_lander_state ----

ENV=lunar_lander_state

python train.py --method no_concept \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cuda --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/no_concept_${ENV}_seed${SEED}.log 2>&1 &
PID0=$!; echo "[PID $PID0] no_concept           / $ENV" | tee -a $LOG

python train.py --method vanilla_freeze --training_mode two_phase \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cuda --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/vanilla_freeze_${ENV}_seed${SEED}.log 2>&1 &
PID1=$!; echo "[PID $PID1] vanilla_freeze        / $ENV" | tee -a $LOG

python train.py --method concept_actor_critic \
    --temporal_encoding none --training_mode end_to_end \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cuda --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/concept_ac_e2e_none_${ENV}_seed${SEED}.log 2>&1 &
PID2=$!; echo "[PID $PID2] concept_ac (e2e,none) / $ENV" | tee -a $LOG

python train.py --method concept_actor_critic \
    --temporal_encoding gru --training_mode end_to_end \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cuda --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/concept_ac_e2e_gru_${ENV}_seed${SEED}.log 2>&1 &
PID3=$!; echo "[PID $PID3] concept_ac (e2e,gru)  / $ENV" | tee -a $LOG

# ---- hidden_velocity ----

ENV=hidden_velocity

python train.py --method no_concept \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cuda --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/no_concept_${ENV}_seed${SEED}.log 2>&1 &
PID4=$!; echo "[PID $PID4] no_concept           / $ENV" | tee -a $LOG

python train.py --method vanilla_freeze --training_mode two_phase \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cuda --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/vanilla_freeze_${ENV}_seed${SEED}.log 2>&1 &
PID5=$!; echo "[PID $PID5] vanilla_freeze        / $ENV" | tee -a $LOG

python train.py --method concept_actor_critic \
    --temporal_encoding none --training_mode end_to_end \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cuda --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/concept_ac_e2e_none_${ENV}_seed${SEED}.log 2>&1 &
PID6=$!; echo "[PID $PID6] concept_ac (e2e,none) / $ENV" | tee -a $LOG

python train.py --method concept_actor_critic \
    --temporal_encoding gru --training_mode end_to_end \
    --env $ENV --seed $SEED \
    --total_timesteps $TS \
    --n_envs $N_ENVS --n_steps $N_STEPS --n_epochs $N_EPOCHS --batch_size $BATCH \
    --device cuda --output_dir $RESULTS_DIR \
    > $RESULTS_DIR/concept_ac_e2e_gru_${ENV}_seed${SEED}.log 2>&1 &
PID7=$!; echo "[PID $PID7] concept_ac (e2e,gru)  / $ENV" | tee -a $LOG

echo "" | tee -a $LOG
echo "Monitor progress:" | tee -a $LOG
echo "  tail -f $RESULTS_DIR/concept_ac_e2e_gru_lunar_lander_state_seed${SEED}.log" | tee -a $LOG
echo "  tail -f $RESULTS_DIR/concept_ac_e2e_gru_hidden_velocity_seed${SEED}.log" | tee -a $LOG
echo "" | tee -a $LOG

wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7
echo "All 8 runs done." | tee -a $LOG

# ---- Plots ----

echo "Generating plots..." | tee -a $LOG

python plot_results.py \
    --env lunar_lander_state \
    --results_dir $RESULTS_DIR \
    --output_dir $PLOTS_DIR_LL

python plot_results.py \
    --env hidden_velocity \
    --results_dir $RESULTS_DIR \
    --output_dir $PLOTS_DIR_HV

echo "========================================" | tee -a $LOG
echo "Done. Plots saved to:" | tee -a $LOG
echo "  $PLOTS_DIR_LL/" | tee -a $LOG
echo "  $PLOTS_DIR_HV/" | tee -a $LOG
echo "========================================" | tee -a $LOG
