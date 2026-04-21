"""
correlation_test.py — Tests whether each concept can be predicted from a single pixel frame.

Fits a logistic regression (PCA-reduced pixels) per concept.
Reports accuracy vs. chance level for each concept.

  Low accuracy (≈ chance) → concept is truly temporal — invisible from single frame.
  High accuracy            → concept is single-frame predictable.

Usage:
  python correlation_test.py --n_samples 3000 --seed 42
"""

__test__ = False

import argparse
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.dynamic_obstacles import make_single_dynamic_obstacles_env


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_frames_and_concepts(env, n_samples: int, seed: int = 0):
    """
    Roll out env with random actions, collect (single_frame, concept) pairs.
    Always uses the first 3 channels (single RGB frame) regardless of n_stack.
    """
    rng = np.random.default_rng(seed)
    frames, concepts = [], []

    obs, _ = env.reset(seed=seed)
    for _ in range(n_samples):
        action = rng.integers(env.action_space.n)
        obs, _, done, truncated, _ = env.step(int(action))

        # First 3 channels = single RGB frame
        frame = obs[:3]
        concepts.append(env.get_concept())
        frames.append(frame)

        if done or truncated:
            obs, _ = env.reset()

    return np.array(frames, dtype=np.uint8), np.array(concepts, dtype=np.float32)


# ---------------------------------------------------------------------------
# Correlation test
# ---------------------------------------------------------------------------

def evaluate_predictability(frames, concepts, concept_names, n_pca: int = 64):
    """
    For each concept, fit logistic regression on PCA-reduced pixel features.
    Train on 80%, test on 20%.

    Returns dict: concept_name → {accuracy, chance, lift}
    """
    N = len(frames)
    # Downsample to 32x32 then flatten: [N, 3*32*32]
    import cv2
    X = np.stack([
        cv2.resize(f.transpose(1, 2, 0), (32, 32)).flatten()
        for f in frames
    ]).astype(np.float32) / 255.0

    split = int(0.8 * N)
    X_train, X_test = X[:split], X[split:]

    results = {}
    for i, name in enumerate(concept_names):
        y = concepts[:, i].astype(int)
        y_train, y_test = y[:split], y[split:]

        # Majority-class chance baseline
        vals, counts = np.unique(y_train, return_counts=True)
        chance = counts.max() / len(y_train)

        n_components = min(n_pca, X_train.shape[1], X_train.shape[0] - 1)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=n_components)),
            ("clf",    LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")),
        ])
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)

        results[name] = {
            "accuracy": acc,
            "chance":   chance,
            "lift":     acc - chance,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",  type=int, default=3000)
    parser.add_argument("--n_pca",      type=int, default=64)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--grid_size",  type=int, default=8)
    args = parser.parse_args()

    env = make_single_dynamic_obstacles_env(seed=args.seed, n_stack=1, grid_size=args.grid_size)

    print(f"Collecting {args.n_samples} (frame, concept) pairs with random actions...")
    frames, concepts = collect_frames_and_concepts(env, args.n_samples, args.seed)
    print(f"Done. frames: {frames.shape}  concepts: {concepts.shape}\n")

    print(f"Fitting logistic regression per concept (PCA n={args.n_pca}, 80/20 split)...")
    results = evaluate_predictability(
        frames, concepts, env.concept_names, n_pca=args.n_pca
    )

    # Print results
    header = f"{'Concept':<35} {'Acc':>6} {'Chance':>8} {'Lift':>7}  Verdict"
    print("\n" + header)
    print("-" * 75)
    for name, r in results.items():
        if r["lift"] < 0.10:
            verdict = "TEMPORAL ✓ (near chance)"
        elif r["lift"] < 0.30:
            verdict = "partially temporal"
        else:
            verdict = "single-frame predictable"
        print(
            f"{name:<35} {r['accuracy']:>6.3f} {r['chance']:>8.3f} "
            f"{r['lift']:>7.3f}  {verdict}"
        )

    print("\nLegend:")
    print("  TEMPORAL ✓        lift < 0.10 — concept invisible from single frame")
    print("  partially temporal lift < 0.30 — some single-frame correlation")
    print("  single-frame       lift >= 0.30 — concept visible from single frame")

    env.close()


if __name__ == "__main__":
    main()
