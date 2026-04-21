## Environment Setup

The project expects a local virtual environment managed with `python3.11` and `pip`. Follow these steps from the repository root:

1. **Check prerequisites**
   - Ensure the macOS command line tools / Homebrew are installed (needed for packages such as `box2d-py` and `swig`).
   - Confirm the interpreter is Python 3.10+ (repo tested with `python3.11 --version`).

2. **Create / refresh the virtual environment**
   ```bash
   python3.11 -m venv .venv
   ```

3. **Activate the environment**
   - macOS / Linux (bash/zsh): `source .venv/bin/activate`
   - fish: `source .venv/bin/activate.fish`
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
   - Windows cmd: `.venv\Scripts\activate.bat`
   - Leave the environment with `deactivate`.

4. **Bootstrap packaging tools (inside the env)**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

5. **Install project dependencies**
   ```bash
   python -m pip install -r requirements.txt
   ```
   This pulls PyTorch, Gymnasium `[box2d]`, Stable-Baselines3, MiniGrid, OpenCV, etc. A `.venv.freeze.txt` lock snapshot is generated via `python -m pip freeze > .venv.freeze.txt`.

6. **Sanity checks**
   - `python -m pip check` – verifies dependency consistency.
   - `python - <<'PY' ...` importing `torch`, `gymnasium`, `stable_baselines3`, `minigrid` (may require setting an OpenMP env var if macOS blocks SHM creation; see below).
   - `python train.py --help` – confirms CLI entry point loads.

### Notes / Known Issues

- If `box2d-py` build fails complaining about `swig`, install it via Homebrew (`brew install swig`) before running `pip install -r requirements.txt`.
- On some macOS sandboxed shells, importing PyTorch may print `OMP: Error #179: Function Can't open SHM2 failed`. Workaround: run outside the restricted environment or set up a local shell session with shared memory support (regular Terminal/iTerm). The rest of the setup can complete successfully; just rerun the import/test commands in a normal shell if needed.
