# Concept Critic Models — Codebase Overview

## Purpose

This project studies **concept bottleneck models for reinforcement learning** — agents that are forced to predict interpretable "concepts" (e.g. velocity, obstacle direction) as an intermediate representation before making decisions. The core question: does structuring the representation this way hurt or help task performance and interpretability?

---

## Entry Points

### `train.py`
The main training script. Parses arguments (`--method`, `--env`, `--training_mode`, `--temporal_encoding`, etc.), builds the environment and PPO model, runs `model.learn()`, then saves results to a directory named by the run configuration:

- `rewards.npy` — episode reward history
- `model.pt` — saved policy weights
- `concept_acc.npz` — concept MSE log over training
- `eval.txt` — final evaluation reward

Every experiment starts here.

### `plot_results.py`
Loads completed runs from disk and generates all comparison plots:

- **Learning curves** — smoothed episode reward vs. episode number, one line per method
- **Concept MSE over time** — velocity MSE at each checkpoint, split by static vs. temporal concepts
- **Per-concept bar chart** — final MSE per concept grouped by method
- **Summary table** — mean/std reward across seeds

Run after training finishes. Accepts `--env`, `--results_dir`, `--output_dir`.

### `compare.py`
An older, heavier version of `plot_results.py` that both trains and plots in one script. Predates the separation of training and plotting. Less used now.

### `correlation_test.py`
Validation tool. Fits a logistic regression (PCA-reduced pixels → class label) per concept to check whether concepts are genuinely invisible from a single frame (temporal) or not.

- **Lift < 0.10** → "TEMPORAL" — concept invisible from single frame, requires history
- **Lift 0.10–0.30** → partially temporal
- **Lift ≥ 0.30** → single-frame predictable

Currently hardcoded to `dynamic_obstacles` with classification concepts. Regression concepts (e.g. MountainCar velocity) would need binning before use.

---

## Shell Scripts

| Script | Purpose |
|---|---|
| `run_mc_comparison.sh` | 3-way MountainCar comparison: no\_concept vs vanilla\_freeze vs concept\_ac (joint), 500k steps |
| `run_mc_long.sh` | Same comparison at 2M steps, then generates all plots |
| `run_short.sh` | Quick smoke-test with reduced timesteps |
| `run_all.sh` | Original bulk experiment script across all envs/methods |

---

## `ppo/` — Core Algorithm

### `ppo.py`
The PPO training loop. Owns `learn()` and three training functions:

- **`train_policy()`** — Standard PPO clipped surrogate loss. Updates actor and value critic. In `two_phase` mode, concept net is frozen (excluded from optimizer).
- **`train_concept_actor_critic()`** — PPO-style update for the concept actor and concept critic. Runs every iteration. Uses `optimizer_concept_and_features` so gradients flow through concept net and feature extractor only (those layers are the only ones in the forward pass).
- **`train_concepts()`** — Supervised MSE/CE on labeled concept samples. Called at query times (periodic), or every iteration in `joint` mode using rollout buffer data.

Also logs concept MSE every 10 iterations via `_compute_concept_accuracy_from_buffer()`.

### `policy.py`
The `ActorCriticPolicy` nn.Module. Wires together:

```
obs → features_extractor → concept_net → mlp_extractor → action_net / value_net
```

Manages four optimizers with different parameter scopes:

| Optimizer | Parameters | Used by |
|---|---|---|
| `optimizer` | All params | PPO in `end_to_end`, concept AC in `end_to_end` |
| `optimizer_exclude_concept` | All except concept\_net | PPO in `two_phase` |
| `optimizer_concept_and_features` | concept\_net + features\_extractor | `train_concepts`, `train_concept_actor_critic` |
| `optimizer_concept_only` | concept\_net only | (reserved) |

### `networks.py`
Defines the two concept modules:

**`FlexibleMultiTaskNetwork`** — Simple linear heads from features to concept predictions. Used by `vanilla_freeze`. Pure supervised, no temporal state. For each concept: `Linear(512 → K)` for classification or `Linear(512 → 1)` for regression.

**`ConceptActorCritic`** — Concept actor (outputs a distribution over concept values) + concept critic (V\_c). Supports three temporal encodings:

- `gru` — GRUCell(512 → 256) carries hidden state h\_t across steps; heads read from h\_t
- `stacked` — no GRU; temporal info comes from frame-stacked observations at env level
- `none` — no temporal information (ablation baseline)

### `buffer.py`
Rollout buffer. Stores per-step: `(obs, concept, action, reward, value, log_prob, hidden_state, concept_value, concept_log_prob, concept_reward)`.

Computes GAE returns and advantages independently for both the task critic and the concept critic.

---

## `envs/` — Environment Wrappers

Each env wraps a Gymnasium environment to expose:

- A (possibly partial) observation space
- `get_concept()` — ground-truth concept labels at every step
- `task_types`, `concept_names`, `temporal_concepts` — metadata used by the policy to set up heads

### `mountain_car.py`
MountainCar with **position-only** observations `[B, 1]`. Velocity is the single hidden temporal concept — not in obs, not in reward. The only way to infer it is from position history (`corr(pos, vel) ≈ 0.06`). Reward is shaped with a position bonus to make the task tractable for PPO.

### `lunar_lander.py`
LunarLander with multiple variants: pixel obs, full state obs, position-only obs. Concepts include x/y velocity, angular velocity, and leg contacts — some static (visible from a single frame), some temporal (require history).

### `cartpole.py`
CartPole with pixel observations and stacked frames. Concepts include pole angle, cart velocity, etc.

### `dynamic_obstacles.py`
MiniGrid-based environment with moving obstacles. Pixel observations. Concepts include obstacle move directions — genuinely temporal since direction requires seeing consecutive frames. Used by `correlation_test.py`.

---

## Three Training Methods

| Method | Concept net training | PPO gradient → concept net |
|---|---|---|
| `no_concept` | No concept net | N/A |
| `vanilla_freeze` | Supervised MSE/CE on labeled samples | Blocked (frozen during PPO) |
| `concept_actor_critic` | Supervised anchor + PPO-clipped concept AC loss | Blocked in `two_phase`; flows through in `end_to_end`; supervised every iteration in `joint` |

### Training Modes

- **`two_phase`** — concept net is frozen during `train_policy()`. Only updated by concept-specific losses. Mirrors the LICORICE paper's vanilla\_freeze setup.
- **`end_to_end`** — policy gradient flows through concept net jointly. Concept net is shaped by both task reward and concept losses simultaneously.
- **`joint`** — concept net is frozen from PPO (same as `two_phase`), but supervised concept training runs every iteration using rollout buffer ground-truth labels — no periodic label queries needed.

---

## Gradient Flow Summary

```
Loss                    Forward pass touches              Updates
────────────────────────────────────────────────────────────────
Supervised anchor       feat_extractor → concept_net      feat_extractor, concept_net
Concept AC loss         feat_extractor → concept_net      feat_extractor, concept_net
PPO (two_phase)         feat_extractor → concept_net      feat_extractor, mlp_extractor,
                        → mlp_extractor → heads           action_net, value_net
                                                          (concept_net FROZEN)
PPO (end_to_end)        full forward pass                 all parameters
```

The computation graph is the real gating mechanism — the optimizer choice only matters for `two_phase` where PPO is explicitly blocked from touching concept\_net.

---

## Key Findings (MountainCar)

MountainCar with position-only observations is a controlled temporal concept test: velocity is the hidden concept, and `corr(pos, vel) ≈ 0.06` means it is genuinely invisible from a single frame.

| Method | Final velocity MSE | Task reward (last 100) |
|---|---|---|
| No Concept | — | ~72 |
| Vanilla Freeze (two\_phase, q=1) | 0.006 | ~65 |
| Concept AC (end\_to\_end) | 1.197 | ~73 |
| Concept AC (joint) | 0.009 | ~72 |

- **End-to-end** concept AC destroys concept interpretability — PPO reshapes the concept net away from velocity toward whatever representation maximizes reward.
- **Joint** training (frozen from PPO, supervised every iteration) matches no\_concept on task reward while learning velocity almost as accurately as vanilla freeze.
- **Vanilla freeze** learns velocity best but pays a task reward cost from the concept bottleneck constraint.
