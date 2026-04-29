#!/usr/bin/env zsh
# setup_env.sh — create a virtualenv or conda env and install project requirements.
# Usage:
#   ./setup_env.sh                # create .venv and install requirements.txt
#   ./setup_env.sh --conda myenv  # create a conda env `myenv` and install requirements there

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
PYTHON=${PYTHON:-python3}
PIP_ARGS=""

USE_CONDA=false
CONDA_ENV_NAME="cenv"  # default conda env name if --conda is passed without an env name

# Simple arg parser
while [[ ${#} -gt 0 ]]; do
  case $1 in
    --conda)
      USE_CONDA=true
      shift
      if [[ -n "${1:-}" && "${1:0:1}" != "-" ]]; then
        CONDA_ENV_NAME="$1"
        shift
      fi
      ;;
    -h|--help)
      echo "Usage: $0 [--conda <env_name>]"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: $0 [--conda <env_name>]"
      exit 1
      ;;
  esac
done

echo "[setup] repo root: $REPO_ROOT"

if $USE_CONDA ; then
  # Conda flow: create environment and run pip/commands inside it using `conda run`
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH. Install Miniconda/Anaconda and retry."
    exit 1
  fi

  echo "[setup] creating conda env: $CONDA_ENV_NAME (python=3.11)"
  conda create -n "$CONDA_ENV_NAME" python=3.11 -y

  echo "[setup] upgrading pip inside conda env"
  conda run -n "$CONDA_ENV_NAME" python -m pip install --upgrade pip setuptools wheel

  if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
    echo "[setup] installing base requirements into conda env $CONDA_ENV_NAME"
    conda run -n "$CONDA_ENV_NAME" python -m pip install -r "$REPO_ROOT/requirements.txt" $PIP_ARGS
  fi

  # Ensure common Gym-related packages are available (install via pip)
  if ! conda run -n "$CONDA_ENV_NAME" conda install -y -c conda-forge pybullet; then
    echo "[setup] conda install pybullet failed; falling back to pip install pybullet"
    conda run -n "$CONDA_ENV_NAME" python -m pip install pybullet || true
  fi

  echo "[setup] installing gym-related packages: panda-gym, gymnasium (pip) and pybullet (conda preferred)"
  conda run -n "$CONDA_ENV_NAME" python -m pip install pybullet panda-gym gymnasium highway-env|| true
  # Prefer conda-forge pybullet on conda envs (better binary support); fall back to pip

  # Verify Python files compile (does not import robosuite)
  echo "[setup] running python -m py_compile train.py inside conda env $CONDA_ENV_NAME"
  if ! conda run -n "$CONDA_ENV_NAME" python -m py_compile train.py; then
    echo "[setup] ERROR: compile check failed for train.py inside conda env $CONDA_ENV_NAME"
    echo "[setup] You can run: conda activate $CONDA_ENV_NAME; python -m py_compile train.py" \
         "to see the full traceback"
    exit 1
  fi

  echo "[setup] done. Activate with: conda activate $CONDA_ENV_NAME"
  exit 0
fi

# If user is inside an active conda env (and didn't pass --conda), install into it
if [[ -n "${CONDA_DEFAULT_ENV:-}" && "$USE_CONDA" = false ]]; then
  echo "[setup] detected active conda env: please re-run with --conda to install into this env, or deactivate conda env and re-run to use a venv"
  exit 1
fi
  

# Fallback: create a standard Python venv
if [[ ! -x "$(command -v $PYTHON)" ]]; then
  echo "ERROR: python3 not found. Install Python 3.10+ and re-run."
  exit 1
fi

# Create venv
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup] creating venv in $VENV_DIR"
  $PYTHON -m venv "$VENV_DIR"
else
  echo "[setup] using existing venv at $VENV_DIR"
fi

# Activate
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "[setup] upgrading pip, setuptools, wheel"
python -m pip install --upgrade pip setuptools wheel

# Install requirements
if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
  echo "[setup] installing base requirements from requirements.txt"
  pip install -r "$REPO_ROOT/requirements.txt" $PIP_ARGS
else
  echo "[setup] WARNING: requirements.txt not found in repo root"
fi

# Install gym-related packages into the venv (pip)
echo "[setup] installing gym-related packages into venv: panda-gym, gymnasium, pybullet"
pip install panda-gym gymnasium pybullet || true


# Add project root to PYTHONPATH in venv activation hook for convenience
ACTIVATE_HOOK="$VENV_DIR/bin/activate"
if ! grep -q "# added by setup_env.sh" "$ACTIVATE_HOOK" 2>/dev/null; then
  cat >> "$ACTIVATE_HOOK" <<'ACTV'
# added by setup_env.sh: add repo root to PYTHONPATH for local imports
export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTV
  echo "[setup] updated venv activate script to include project PYTHONPATH"
fi

echo "[setup] done. Activate with: source $VENV_DIR/bin/activate"

# Run compile check in the venv (does not import robosuite)
echo "[setup] running python -m py_compile train.py inside venv"
if ! python -m py_compile train.py; then
  echo "[setup] ERROR: compile check failed for train.py inside venv"
  echo "[setup] Activate the venv and run: python -m py_compile train.py to see the traceback"
  exit 1
fi

echo "[setup] compile check passed"