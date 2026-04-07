#!/bin/bash
# run_mc_comparison.sh — compare Vanilla Freeze vs Concept AC (GRU) on MountainCar
# Single temporal concept: velocity only (position-only observation)
#
# Usage: bash run_mc_comparison.sh

set -e

ENV=mountain_car
TS=500000
N_ENVS=8
SEED=42
RESULTS_DIR=/glade/derecho/scratch/adadelek/results/mc_comparison
PLOTS_DIR=plots/mc_comparison

mkdir -p $RESULTS_DIR $PLOTS_DIR

echo "========================================"
echo "MountainCar: No Concept vs Vanilla Freeze vs Concept AC joint (none)"
echo "env=$ENV  timesteps=$TS  n_envs=$N_ENVS  seed=$SEED"
echo "========================================"

python train.py \
    --method no_concept \
    --env $ENV --seed $SEED \
    --total_timesteps $TS --n_envs $N_ENVS \
    --device cuda \
    --output_dir $RESULTS_DIR &
PID0=$!
echo "no_concept PID=$PID0"

python train.py \
    --method vanilla_freeze \
    --training_mode two_phase \
    --query_num_times 1 \
    --env $ENV --seed $SEED \
    --total_timesteps $TS --n_envs $N_ENVS \
    --device cuda \
    --output_dir $RESULTS_DIR &
PID1=$!
echo "vanilla_freeze PID=$PID1"

python train.py \
    --method concept_actor_critic \
    --temporal_encoding none \
    --training_mode joint \
    --query_num_times 1 \
    --env $ENV --seed $SEED \
    --total_timesteps $TS --n_envs $N_ENVS \
    --device cuda \
    --output_dir $RESULTS_DIR &
PID2=$!
echo "concept_ac_end_to_end_none PID=$PID2"

wait $PID0 $PID1 $PID2
echo ""
echo "========================================"
echo "Training done. Generating results..."
echo "========================================"

python - <<'EOF'
import numpy as np, os

results_dir = "/glade/derecho/scratch/adadelek/results/mc_comparison"
runs = {
    "No Concept":            "no_concept_two_phase_none_mountain_car_seed42",
    "Vanilla Freeze":        "vanilla_freeze_two_phase_none_mountain_car_seed42",
    "Concept AC joint":      "concept_actor_critic_joint_none_mountain_car_seed42",
}

print("\n=== Task Reward ===")
print(f"{'Method':<22}  {'mean':>8}  {'last100':>8}  {'eval':>8}")
print("-" * 55)
for name, run in runs.items():
    r = np.load(f"{results_dir}/{run}/rewards.npy")
    # eval reward from log
    import subprocess
    log = subprocess.check_output(
        f"grep 'eval:' {results_dir}/{run}/../../../*.log 2>/dev/null || echo ''",
        shell=True
    ).decode().strip()
    print(f"{name:<22}  {np.mean(r):>8.2f}  {np.mean(r[-100:]):>8.2f}")

print("\n=== Velocity Concept MSE (lower=better) ===")
print(f"{'Method':<22}  {'start':>10}  {'mid':>10}  {'end':>10}")
print("-" * 58)
for name, run in runs.items():
    path = f"{results_dir}/{run}/concept_acc.npz"
    if not os.path.exists(path):
        print(f"{name:<22}  no concept_acc.npz"); continue
    d = np.load(path, allow_pickle=True)
    ts, names, vals = d["timesteps"], d["names"].tolist(), d["values"]
    if "velocity" in names:
        idx = names.index("velocity")
        n = len(ts)
        start, mid, end = vals[0, idx], vals[n//2, idx], vals[-1, idx]
        print(f"{name:<22}  {start:>10.5f}  {mid:>10.5f}  {end:>10.5f}")

EOF

python plot_results.py \
    --env $ENV \
    --results_dir $RESULTS_DIR \
    --output_dir $PLOTS_DIR \
    2>/dev/null || echo "(plot_results.py skipped — check args)"

echo ""
echo "Done. Results in $RESULTS_DIR  Plots in $PLOTS_DIR"
