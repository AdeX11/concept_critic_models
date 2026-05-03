#!/usr/bin/env bash
# colab/setup.sh — Bootstrap a Colab runtime for Stage 0 work.
#
# Run via:  !bash colab/setup.sh
#
# Steps:
#   1. Install requirements.txt + highway_env (missing from requirements)
#   2. Print Python / GPU / torch versions
#   3. Smoke: 1k-step no_concept on armed_corridor_state (flat-vec, fastest)
#   4. Calibrate: 5k-step no_concept+gru on armed_corridor (pixel + memory) and
#      report measured steps/s, used to project suite wall-clock at session start
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== python ==="
python --version

echo "=== installing requirements ==="
pip install -q -r requirements.txt
pip install -q highway_env

echo "=== environment ==="
python - <<'PY'
import torch, sys
print(f"python         : {sys.version.split()[0]}")
print(f"torch          : {torch.__version__}")
print(f"cuda available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu            : {torch.cuda.get_device_name(0)}")
PY

SMOKE_DIR="/tmp/colab_smoke_$(date +%s)"
echo "=== smoke (1k steps, armed_corridor_state, no_concept) ==="
python -u train.py \
    --method no_concept \
    --benchmark armed_corridor_state \
    --seed 42 \
    --total_timesteps 1024 \
    --n_envs 4 --n_steps 256 --n_epochs 2 --batch_size 64 \
    --device auto \
    --output_dir "$SMOKE_DIR"

CAL_DIR="/tmp/colab_calibrate_$(date +%s)"
echo "=== calibration (5k steps, armed_corridor pixel+gru) ==="
START=$(date +%s)
python -u train.py \
    --method no_concept \
    --benchmark armed_corridor \
    --temporal_encoding gru \
    --seed 42 \
    --total_timesteps 5120 \
    --n_envs 4 --n_steps 512 --n_epochs 4 --batch_size 128 \
    --device auto \
    --output_dir "$CAL_DIR"
ELAPSED=$(($(date +%s) - START))

echo "=== throughput ==="
python - <<PY
elapsed = $ELAPSED
steps = 5120
sps = steps / max(elapsed, 1)
suite_steps = 31_600_000
print(f"calibration: {steps} steps in {elapsed}s -> {sps:.0f} steps/s")
print(f"projected single 300k run wall-clock: {300_000 / max(sps, 1) / 60:.0f} min")
print(f"projected single 1M run wall-clock:   {1_000_000 / max(sps, 1) / 60:.0f} min")
print(f"projected full suite (~31.6M steps):  {suite_steps / max(sps, 1) / 3600:.1f} h serial")
PY

echo "=== setup complete ==="
