# Stage 0 Pilot Plan — Concept Critic Models

**Status:** authoritative spec for Stage 0. Supersedes all prior plans.
**Branch:** `domingo-experimental` (commit `4d99952` and later)
**Date drafted:** 2026-04-28

---

## Bottom Line

The next step is **validation and default selection**, not env construction. Six temporal env families are implemented; what's missing is a defensible experimental protocol. Stage 0 produces canonical defaults before any main-matrix run.

## Hard Commitments (do not skip, even under pressure)

1. `vanilla_freeze` (CBM baseline) must be in Stage 0 — otherwise the comparison is unfair.
2. Defaults cannot be chosen from one seed. Coarse 1-seed sweep → 3-seed revalidation of top candidates.
3. Finalists must be confirmed at 1M timesteps before being declared canonical.
4. `gru` vs `none` must be tested on hidden temporal benchmarks early.
5. Admission criteria and tiebreak rules in this doc are written **before** any sweep results exist. They are not negotiable post-hoc.
6. If all configs fail hidden-variant admission after 1M, **halt and debug** — do not expand to the main matrix.
7. Sweep is staged, not full cross-product (see §Stage 0 Pilot).

---

## Current Branch State

### Implemented infrastructure
- `benchmark_registry.py` — benchmark IDs and canonical hyperparameters
- `envs/registry.py` — shared env factory
- `ppo/ppo.py` + `train.py` — checkpoint save/load, periodic eval, TensorBoard hooks
- `cluster/run_single.sbatch`, `cluster/run_sweep.py`, `cluster/aggregate.py` — basic launch and aggregation

### Implemented benchmarks
- Calibration: `cartpole`, `mountain_car`, `lunar_lander_state`, `lunar_lander_pos_only`, `lunar_lander`
- Legacy: `dynamic_obstacles`
- Event memory: `armed_corridor` triplet (state/visible/hidden)
- Phase inference: `phase_crossing` triplet (+ hard variant)
- Latent dynamics: `momentum_corridor` triplet (+ hard variant)
- Synchrony: `synchrony_window` triplet

### Engineering tasks before Stage 0 launch (in order)

1. **Define `training_mode="joint"` semantics in code/comments OR remove from sweep.** If the difference between `joint` and `end_to_end` cannot be articulated in writing, drop `joint` from the sweep.
2. **Add programmatic action histogram to `evaluate_detailed()`** in `ppo/ppo.py`. Track action counts during eval; return `action_histogram: List[int]` (length `n_actions`) and `dominant_action_fraction = max(hist)/sum(hist)` in `eval.json`.
3. **Extend `cluster/aggregate.py`** to emit one row per run: benchmark_id, method, seed, training_mode, temporal_encoding, learning_rate, ent_coef, lambda_v, lambda_s, num_labels, query_num_times, mean_reward, std_reward, success_rate, normalized_return, mean_episode_length, dominant_action_fraction, terminal-cause breakdown, per-concept metrics.
4. **Extend `cluster/run_single.sbatch`** to pass through: `TEMPORAL_ENCODING`, `TRAINING_MODE`, `LEARNING_RATE`, `ENT_COEF`, `LAMBDA_V`, `LAMBDA_S`, `NUM_LABELS`, `QUERY_NUM_TIMES`.
5. **Add `cluster/run_pilot.py`** — pilot sweep launcher implementing the staged sweep below.
6. **Add `tests/test_checkpoint_resume.py`** — train tiny budget, save, resume, confirm `num_timesteps`/reward history/eval log/optimizer state continuity.
7. **Local smoke test** before any OSCAR submission: one tiny job per method on `armed_corridor` and `phase_crossing`. Verify checkpoints, `eval.json`, `eval_checkpoints.json`, TensorBoard files; verify resume from `latest.pt`. **Measure throughput (steps/sec/GPU)** — this calibrates compute budget.

### Stage C prerequisite (verify, don't assume)
- Concept removal ablation requires env-side per-concept masking. **Status: unverified.** Scope as an engineering task before Stage C launch.

---

## Stage 0 Pilot

### Throughput assumption (verify at smoke test)
Working assumption: ~3000 steps/sec/GPU on small-grid CNN PPO with `n_envs=4`. At this rate, 300k ≈ 100s wall and 1M ≈ 333s wall per run. **If smoke test reveals throughput < 1500 steps/s, multiply all GPU-h figures by the ratio.** Re-derive budget rather than guess.

### Pilot benchmarks
- **Primary:** `armed_corridor` (event memory / countdown)
- **Secondary:** `phase_crossing` (phase inference)
- **Optional confirmation:** `momentum_corridor` (latent dynamics) — run only if Stages 0a–0d look anomalous

### Methods
`no_concept` (skyline), `vanilla_freeze` (baseline), `concept_actor_critic` (proposed)

### Staged sweep (avoids unmanageable cross-product)

Full CAC cross-product = 96 configs. Instead, exploit weak-interaction assumption between distant axes and stage the search:

**ROUND 1 — temporal architecture pilot.**
Lock `LR=3e-4, ent_coef=0.01, λ_v=λ_s=0.5`.
- CAC: sweep `training_mode × temporal_encoding` = 6 configs (drop `joint` if not semantically defined → 4 configs)
- `no_concept`: sweep `temporal_encoding ∈ {gru, none}` = 2 configs
- `vanilla_freeze` (training_mode locked at `two_phase`): sweep `temporal_encoding` = 2 configs

Total: ~10 configs × 2 envs × 1 seed × 300k = **20 runs ≈ 0.6 GPU-h** (assumed throughput).
Pick top 2 CAC `(training_mode, temporal_encoding)` by normalized return.

**ROUND 2 — optimization pilot.**
At winning `(training_mode, temporal_encoding)` per method, sweep `LR ∈ {3e-4, 1e-4} × ent_coef ∈ {0.01, 0.001}` = 4 configs/method.
Total: 4 × 3 methods × 2 envs × 1 seed × 300k = **24 runs ≈ 0.7 GPU-h.**

**ROUND 3 — concept loss pilot (CAC only).**
At winning LR/ent_coef from Round 2, sweep `λ_v ∈ {0.25, 0.5} × λ_s ∈ {0.25, 0.5}` = 4 configs.
Total: 4 × 2 envs × 1 seed × 300k = **8 runs ≈ 0.2 GPU-h.**

**ROUND 4 — vanilla_freeze label pilot.**
At winning LR/ent_coef from Round 2, sweep `num_labels ∈ {250, 500, 1000} × query_num_times ∈ {1, 2}` = 6 configs.
Total: 6 × 2 envs × 1 seed × 300k = **12 runs ≈ 0.4 GPU-h.**

### Sanity invariants — flag and halt if violated
- `gru` must beat `none` on hidden temporal benchmarks. If tied within seed noise, GRU integration is suspect — investigate before continuing.
- Visible variant must beat hidden by ≥ 0.10 normalized return on at least one method. Otherwise the env is not isolating temporal info.

### Per-round candidate filters (reject regardless of reward rank)
- `success_rate < 0.05`
- `dominant_action_fraction > 0.85` (action collapse)
- Single terminal cause > 0.85 of episodes (trivial failure mode)
- Replay shows trivial waiting/spinning/always-forward (diagnostic only — not the primary signal)

### 3-seed revalidation
Best 1 configuration per method at seeds `42, 123, 456`, 300k.
3 methods × 3 seeds × 2 envs = **18 runs ≈ 0.5 GPU-h.**

### Long-horizon confirmation
Best 1 candidate per method × 2 envs × 1 seed × 1M.
3 × 2 × 1 = **6 runs ≈ 0.6 GPU-h.**

### CAC architecture-fix ablation (one-shot)
At canonical CAC default, run on `armed_corridor` at 1M with architecture rolled back to pre-fix:
- state-only `V_c` (not action-conditional `Q_c`)
- scalar concept indices (not one-hot STE)

**1 run ≈ 0.1 GPU-h.** If pre-fix wins or ties, halt and investigate before main matrix.

### Stage 0 total
~3 GPU-h optimistic A100; ~9–18 GPU-h realistic V100. Re-derive at smoke test.

---

## Tiebreaker Rules (canonical, written before any results)

**Primary selector:** highest mean normalized return at fixed budget.

**Secondary (when within 10% relative normalized return):**
1. For CBM methods: prefer higher concept accuracy.
2. Prefer lower `dominant_action_fraction`.
3. Prefer `two_phase` over `end_to_end`/`joint` when reward is effectively tied AND concept metrics are materially better.

**For 3-seed revalidation:** the 3-seed mean must beat runner-up by a margin whose paired-seed bootstrap 95% CI excludes 0. If CI overlaps 0, the configurations are tied — fall through to secondary criteria.

---

## Admission Criteria for Main Matrix

A temporal family is admitted if EITHER path holds:

**Path A — standard:**
- `*_state` ≥ 0.70 normalized return
- `*_visible` ≥ 0.50 normalized return
- visible − hidden gap ≥ 0.10

**Path B — low ceiling, well-formed gradient:**
- visible − hidden gap ≥ 0.15
- state > visible (ordering preserved)

Path B admits benchmarks whose absolute ceiling is lower than expected but whose observability gradient is informative.

**Reject if:**
- Hidden fails to differentiate from random AND visible also fails.
- visible − hidden gap < 0.05 (env doesn't isolate temporal info).

**Halt protocol:** if no env passes admission after Stage 0 long-horizon confirmation, **halt and debug architecture or env design.** Do not expand to main matrix.

---

## Main Matrix (only after Stage 0)

- Methods: all three
- Seeds: 3 initially, expand to 5 only where spread appears
- Budget: registry canonical defaults unless Stage 0 shows insufficient
- Per-run outputs: `metadata.json`, checkpoints, TensorBoard, `eval.json`, `eval_checkpoints.json`

**Spread trigger for ablations:**
- `gap = best_mean_normalized_return − second_best_mean_normalized_return`
- Trigger only if `gap ≥ 0.10` AND bootstrap 95% CI for paired-seed gap excludes 0.

---

## Stage C Ablations (only on admitted benchmarks with detected spread)

- `temporal_encoding`: gru vs none, optionally stacked
- `training_mode`: canonical vs strongest alternative
- Sparse labels: 100% (env canonical), 25%, 10% of canonical `num_labels`
- Concept removal: temporal concepts removed one at a time **(requires env-side mask support — verify before scheduling)**
- For CAC: revisit `λ_v`, `λ_s` if spread axis warrants

---

## Compute Budget Summary

| Phase | Optimistic (A100) | Realistic (V100) |
|---|---|---|
| Stage 0 (4 rounds + revalidation + 1M + ablation) | ~3 GPU-h | ~9–18 GPU-h |
| Main matrix (assume 8 admitted, 3 methods, 3 seeds, 1M) | ~7 GPU-h | ~25 GPU-h |
| Targeted ablations | ~10 GPU-h | ~30 GPU-h |
| **Total planned envelope** | **~20 GPU-h** | **~75 GPU-h** |
| Upper bound with retries / 5-seed expansion | ~50 GPU-h | ~200 GPU-h |

Re-derive at smoke test once throughput is measured.

---

## Risks (preserved in writing)

1. A 1-seed winner can be noise. Never skip 3-seed revalidation.
2. A 300k winner can collapse by 1M. Never skip long-horizon confirmation.
3. Reward alone hides trivial policies. Never decide without terminal distributions, action histograms, and replay checks.
4. If `gru` does not beat `none` on hidden benchmarks, investigate before main matrix.
5. If visible does not beat hidden on any method, the benchmark is not isolating memory.
6. If no config passes hidden admission, the architecture or env is broken — debug before expanding.
7. Throughput assumptions may be wrong — re-derive budget at smoke test, not at OSCAR submission time.

---

## Open Questions (resolve before launch)

1. What is the precise semantic of `training_mode=joint` vs `end_to_end`? Document or drop from sweep.
2. Does the env code support per-concept masking for Stage C concept removal? If not, scope it as an engineering task.
3. Does smoke test confirm ≥ 1500 steps/sec/GPU throughput? Adjust budgets if not.

---

## Deferred Work (out of scope for Stage 0)

- **GVF default selection.** `GVFConceptNetwork` has been ported into this branch and can be launched deliberately with `METHOD=gvf`; `cluster/run_pilot.py --include_gvf` adds optional GVF Round 1 jobs without changing the default Stage 0 protocol. GVF is still **not** part of canonical Stage 0 default selection unless the protocol is explicitly expanded before launch.
- **Ade benchmark/evaluation additions.** `tmaze`, `hidden_velocity`, and `steerability_eval.py` are integrated into the current registry/CLI path for follow-on standardized experiments. They are not part of the first Stage 0 pilot benchmarks unless the launch command explicitly supplies them via `--benchmarks`.
