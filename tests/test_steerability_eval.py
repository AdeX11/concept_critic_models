import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.registry import make_single_env
from ppo.policy import ActorCriticPolicy
from runtime_utils import get_obs_shape
from steerability_eval import evaluate_steerability


def make_policy(method: str = "concept_actor_critic"):
    env = make_single_env("tmaze", seed=0, temporal_encoding="gru")
    policy = ActorCriticPolicy(
        obs_shape=get_obs_shape(env),
        n_actions=env.action_space.n,
        method=method,
        task_types=env.task_types,
        num_classes=env.num_classes,
        concept_dim=len(env.task_types),
        temporal_encoding="gru",
        features_dim=32,
        net_arch=[16],
        device="cpu",
    )
    return policy, env


def test_policy_accepts_raw_concept_override_for_one_hot_cac_bottleneck():
    policy, env = make_policy("concept_actor_critic")
    obs, _ = env.reset(seed=1)
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    override = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    action, h_new = policy.predict(obs_t, deterministic=True, concept_override=override)
    assert action.shape == (1,)
    assert h_new is not None
    env.close()


def test_steerability_eval_runs_for_untrained_tmaze_policy():
    policy, env = make_policy("concept_actor_critic")
    with torch.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias[:] = torch.tensor([1.0, 0.0, 0.0])
    metrics = evaluate_steerability(policy, env, n_episodes=2, seed=2, device=torch.device("cpu"))
    assert metrics["n_episodes"] == 2
    assert metrics["reward_baseline"] is not None
    assert metrics["reward_correct"] is not None
    assert metrics["reward_flipped"] is not None
    env.close()
