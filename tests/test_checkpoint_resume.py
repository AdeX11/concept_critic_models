import os
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.vector import SyncVectorEnv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo.ppo import PPO
from runtime_utils import NullSummaryWriter


class TinyConceptEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)
        self.task_types = ["classification"]
        self.num_classes = [2]
        self.concept_names = ["parity"]
        self.temporal_concepts = []
        self._rng = np.random.default_rng(seed)
        self.t = 0
        self.current_concept = np.zeros(1, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = 0
        self.current_concept = np.array([0.0], dtype=np.float32)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        self.current_concept = np.array([float(self.t % 2)], dtype=np.float32)
        obs = np.array([self.t, action, self.t % 2, 1.0], dtype=np.float32)
        reward = 1.0 if int(action) == self.t % self.action_space.n else 0.0
        terminated = self.t >= 3
        info = {
            "concept": self.current_concept.copy(),
            "success": reward > 0.0,
            "failure_reason": "none" if reward > 0.0 else "miss",
        }
        if terminated:
            info["terminal_observation"] = obs.copy()
        return obs, reward, terminated, False, info

    def get_concept(self):
        return self.current_concept.copy()


def make_model(seed: int = 7):
    vec_env = SyncVectorEnv([lambda: TinyConceptEnv(seed=seed)])
    eval_env = TinyConceptEnv(seed=seed + 100)
    policy_kwargs = {
        "obs_shape": eval_env.observation_space.shape,
        "n_actions": eval_env.action_space.n,
        "task_types": eval_env.task_types,
        "num_classes": eval_env.num_classes,
        "concept_dim": len(eval_env.task_types),
        "concept_names": eval_env.concept_names,
        "features_dim": 16,
        "net_arch": [16],
        "temporal_encoding": "none",
        "device": "cpu",
    }
    model = PPO(
        method="concept_actor_critic",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=4,
        n_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        training_mode="joint",
        seed=seed,
        device="cpu",
        verbose=0,
        eval_env=eval_env,
        writer=NullSummaryWriter(),
    )
    return model, vec_env, eval_env


def make_gvf_model(seed: int = 11):
    vec_env = SyncVectorEnv([lambda: TinyConceptEnv(seed=seed)])
    eval_env = TinyConceptEnv(seed=seed + 100)
    policy_kwargs = {
        "obs_shape": eval_env.observation_space.shape,
        "n_actions": eval_env.action_space.n,
        "task_types": eval_env.task_types,
        "num_classes": eval_env.num_classes,
        "concept_dim": len(eval_env.task_types),
        "concept_names": eval_env.concept_names,
        "features_dim": 16,
        "net_arch": [16],
        "temporal_encoding": "none",
        "gvf_pairing": [0],
        "device": "cpu",
    }
    model = PPO(
        method="gvf",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=4,
        n_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        training_mode="two_phase",
        seed=seed,
        device="cpu",
        verbose=0,
        eval_env=eval_env,
        writer=NullSummaryWriter(),
    )
    return model, vec_env, eval_env


def assert_optimizer_state_equal(left, right):
    assert left.keys() == right.keys()
    assert len(left["param_groups"]) == len(right["param_groups"])
    assert left["state"].keys() == right["state"].keys()
    for param_id, state in left["state"].items():
        other = right["state"][param_id]
        assert state.keys() == other.keys()
        for key, value in state.items():
            other_value = other[key]
            if torch.is_tensor(value):
                assert torch.equal(value, other_value), key
            else:
                assert value == other_value


def test_checkpoint_resume_preserves_training_state(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    model, vec_env, eval_env = make_model()
    try:
        model.learn(
            total_timesteps=8,
            query_num_times=0,
            eval_every_timesteps=4,
            eval_n_episodes=2,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_every_timesteps=4,
            run_metadata={"test": "checkpoint_resume"},
        )
        checkpoint = torch.load(checkpoint_dir / "latest.pt", map_location="cpu")
    finally:
        vec_env.close()
        eval_env.close()

    assert checkpoint["num_timesteps"] == 8
    assert checkpoint["episode_reward_history"]
    assert checkpoint["eval_log"]
    assert checkpoint["resume_state"]["next_eval_at"] == 12
    assert checkpoint["resume_state"]["next_checkpoint_at"] == 12
    assert checkpoint["rng_state"]
    assert checkpoint["optimizer_state_dict"]["state"]
    assert checkpoint["optimizer_exclude_concept_state_dict"] is not None
    assert checkpoint["optimizer_concept_only_state_dict"] is not None
    assert checkpoint["optimizer_concept_and_features_state_dict"]["state"]

    resumed, resumed_vec_env, resumed_eval_env = make_model()
    try:
        resumed.load_checkpoint_state(checkpoint)
        assert resumed.num_timesteps == checkpoint["num_timesteps"]
        assert list(resumed.episode_reward_history) == checkpoint["episode_reward_history"]
        assert resumed.eval_log == checkpoint["eval_log"]
        assert resumed._next_eval_at == checkpoint["resume_state"]["next_eval_at"]
        assert resumed._next_checkpoint_at == checkpoint["resume_state"]["next_checkpoint_at"]

        assert_optimizer_state_equal(
            checkpoint["optimizer_state_dict"],
            resumed.policy.optimizer.state_dict(),
        )
        assert_optimizer_state_equal(
            checkpoint["optimizer_exclude_concept_state_dict"],
            resumed.policy.optimizer_exclude_concept.state_dict(),
        )
        assert_optimizer_state_equal(
            checkpoint["optimizer_concept_only_state_dict"],
            resumed.policy.optimizer_concept_only.state_dict(),
        )
        assert_optimizer_state_equal(
            checkpoint["optimizer_concept_and_features_state_dict"],
            resumed.policy.optimizer_concept_and_features.state_dict(),
        )

        resumed.learn(
            total_timesteps=12,
            query_num_times=0,
            eval_every_timesteps=4,
            eval_n_episodes=2,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_every_timesteps=4,
            run_metadata={"test": "checkpoint_resume"},
        )
        assert resumed.num_timesteps == 12
        assert len(resumed.episode_reward_history) >= len(checkpoint["episode_reward_history"])
        assert len(resumed.eval_log) > len(checkpoint["eval_log"])
    finally:
        resumed_vec_env.close()
        resumed_eval_env.close()


def test_evaluate_detailed_reports_action_histogram():
    model, vec_env, eval_env = make_model()
    try:
        metrics = model.evaluate_detailed(n_episodes=3, deterministic=True)
    finally:
        vec_env.close()
        eval_env.close()

    histogram = metrics["action_histogram"]
    assert len(histogram) == 3
    assert sum(histogram) == sum(metrics["episode_lengths"])
    assert metrics["dominant_action_fraction"] == pytest.approx(max(histogram) / sum(histogram))
    assert 0.0 <= metrics["dominant_action_fraction"] <= 1.0


def test_evaluate_detailed_vector_fallback_reports_action_histogram():
    model, vec_env, eval_env = make_model()
    eval_env.close()
    model.eval_env = None
    try:
        metrics = model.evaluate_detailed(n_episodes=3, deterministic=True)
    finally:
        vec_env.close()

    histogram = metrics["action_histogram"]
    assert len(histogram) == 3
    assert sum(histogram) > 0
    assert 0.0 <= metrics["dominant_action_fraction"] <= 1.0


def test_gvf_training_smoke_produces_eval_and_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "gvf_checkpoints"
    model, vec_env, eval_env = make_gvf_model()
    try:
        model.learn(
            total_timesteps=8,
            query_num_times=1,
            query_labels_per_time=4,
            eval_every_timesteps=4,
            eval_n_episodes=2,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_every_timesteps=4,
            run_metadata={"test": "gvf_training_smoke"},
        )
        metrics = model.evaluate_detailed(n_episodes=2, deterministic=True)
    finally:
        vec_env.close()
        eval_env.close()

    assert model.num_timesteps == 8
    assert model.policy.concept_net.num_gvf == 1
    assert (checkpoint_dir / "latest.pt").exists()
    assert len(metrics["action_histogram"]) == 3
    assert metrics["dominant_action_fraction"] is not None
