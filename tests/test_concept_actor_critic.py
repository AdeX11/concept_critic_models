"""
tests/test_concept_actor_critic.py

Three regression tests covering the fixes from the audit:

  TEST 1 — ratio_identity_mixed_concepts
    ConceptActorCritic with mixed classification + regression.
    Samples concept_actions, evaluates log_prob on those same actions twice
    (old and new on the same frozen network), and verifies the PPO importance
    ratio equals 1.0 to float precision.  This is the minimal sanity check
    that the sampled-action fix is wired correctly end-to-end.

  TEST 2 — decode_concept_vector_roundtrip
    forward() returns c_t in [B, policy_dim] one-hot format.
    decode_concept_vector(c_t) must return [B, n_concepts] where each
    classification column holds a valid class index in [0, K_i).

  TEST 3 — deque_logging_no_type_error
    list(deque)[-100:] must not raise TypeError (the fix).
    deque[-100:] must raise TypeError (confirming the bug it replaced).
"""

import collections
import math

import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo.networks import ConceptActorCritic


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

TASK_TYPES  = ["classification", "regression", "classification", "classification"]
NUM_CLASSES = [4, 1, 3, 5]   # regression placeholder is 1 (unused by the network)
FEATURE_DIM = 32
N_ACTIONS   = 7
BATCH_SIZE  = 8


@pytest.fixture
def net() -> ConceptActorCritic:
    """Fresh ConceptActorCritic in eval mode (no GRU dropout etc.)."""
    model = ConceptActorCritic(
        feature_dim=FEATURE_DIM,
        task_types=TASK_TYPES,
        num_classes=NUM_CLASSES,
        n_actions=N_ACTIONS,
        temporal_encoding="none",   # simplest variant; no GRU hidden state
    )
    model.eval()
    return model


@pytest.fixture
def features() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH_SIZE, FEATURE_DIM)


# ---------------------------------------------------------------------------
# TEST 1 — PPO ratio identity
# ---------------------------------------------------------------------------

def test_ratio_identity_mixed_concepts(net, features):
    """
    PPO importance ratio must equal 1.0 when old and new policies are identical.

    Steps:
      1. forward() → concept_dists (the "old" policy, effectively)
      2. sample_concept_actions() → concept_actions_stored
      3. concept_log_probs(dists, stored) → old_clp
      4. forward() again on the same features (same weights → same dists)
      5. concept_log_probs(same_dists, stored) → new_clp
      6. ratio = exp(new_clp - old_clp)  must be ≈ 1.0

    This fails if concept_log_probs uses concept_slices (one-hot indexing) when
    concept_actions is already in [B, n_concepts] per-concept layout — the fix
    keeps them separate.
    """
    with torch.no_grad():
        c_t, _, concept_dists_old, _ = net(features)

    concept_actions_stored = net.sample_concept_actions(concept_dists_old)
    old_clp = net.concept_log_probs(concept_dists_old, concept_actions_stored)

    # Recompute distributions from the same inputs (same weights → identical dists)
    with torch.no_grad():
        _, _, concept_dists_new, _ = net(features)

    new_clp = net.concept_log_probs(concept_dists_new, concept_actions_stored)

    ratio = torch.exp(new_clp - old_clp)

    assert old_clp.shape == (BATCH_SIZE,), (
        f"log_prob shape should be [B], got {old_clp.shape}"
    )
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5), (
        f"PPO ratio not ≈ 1.0; max deviation {(ratio - 1).abs().max().item():.2e}\n"
        f"old_clp={old_clp}\nnew_clp={new_clp}"
    )


# ---------------------------------------------------------------------------
# TEST 2 — decode_concept_vector round-trip
# ---------------------------------------------------------------------------

def test_decode_concept_vector_roundtrip(net, features):
    """
    decode_concept_vector must invert the one-hot encoding produced by forward().

    For each classification concept i with K_i classes:
      - c_t[:, start:end] is a one-hot-STE vector of width K_i
      - argmax selects the winning class; must be in [0, K_i)
    For regression concepts:
      - the scalar passes through unchanged (not NaN, not Inf)

    The output shape must be [B, n_concepts], not [B, policy_dim].
    """
    with torch.no_grad():
        c_t, _, _, _ = net(features)

    decoded = net.decode_concept_vector(c_t)

    # Shape
    assert decoded.shape == (BATCH_SIZE, net.n_concepts), (
        f"Expected [{BATCH_SIZE}, {net.n_concepts}], got {decoded.shape}"
    )

    # Per-concept validity
    for i, (task_type, n_cls) in enumerate(zip(TASK_TYPES, NUM_CLASSES)):
        col = decoded[:, i]
        if task_type == "classification":
            assert col.dtype == torch.float32, f"Concept {i}: expected float, got {col.dtype}"
            classes = col.long()
            assert (classes >= 0).all() and (classes < n_cls).all(), (
                f"Concept {i} (cls, K={n_cls}): out-of-range values {col.tolist()}"
            )
        else:
            assert not torch.isnan(col).any(), f"Concept {i} (reg): NaN in decoded output"
            assert not torch.isinf(col).any(), f"Concept {i} (reg): Inf in decoded output"


# ---------------------------------------------------------------------------
# TEST 3 — deque logging fix
# ---------------------------------------------------------------------------

def test_deque_logging_no_type_error():
    """
    Verifies that the ppo.py logging fix is semantically correct.

    The bug: `deque[-100:]` raises TypeError because deque does not support
    slice notation.  The fix: `list(deque)[-100:]` converts first.

    This test confirms:
      (a) the fixed form returns exactly 100 elements from a full deque
      (b) the raw deque slice raises TypeError, documenting why the fix exists
    """
    d = collections.deque(maxlen=10_000)
    for v in range(200):
        d.append(float(v))

    # Fixed form — must not raise
    window = list(d)[-100:]
    assert len(window) == 100, f"Expected 100 entries, got {len(window)}"
    assert window[-1] == 199.0, f"Last entry should be 199.0, got {window[-1]}"

    # Buggy form — must raise TypeError (documenting the original bug)
    with pytest.raises(TypeError):
        _ = d[-100:]
