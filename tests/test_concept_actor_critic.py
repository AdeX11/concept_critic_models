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

# Tests TEST 1 (PPO ratio identity via sample_concept_actions / concept_log_probs)
# and TEST 2 (decode_concept_vector round-trip) were specific to the
# domingo-experimental ConceptActorCritic API. The migrated angelic-new
# ConceptActorCritic uses a different design without those methods, so the
# tests no longer apply. The deque-logging fix below remains relevant.


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
