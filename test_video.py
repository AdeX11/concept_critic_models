"""Quick integration test: train PPO for a few steps then record a video."""
import os
import numpy as np
import torch
from envs.pick_place import make_panda_pickplace_env, make_single_panda_env
from ppo.ppo import PPO
from envs.record import record_rollout_from_env

def main():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_envs = 2
    vec_env = make_panda_pickplace_env(n_envs=n_envs, seed=seed)
    single_env = make_single_panda_env(seed=seed)

    policy_kwargs = dict(
        obs_shape=single_env.observation_space.shape,
        n_actions=vec_env.single_action_space.n,
        task_types=single_env.task_types,
        num_classes=single_env.num_classes,
        concept_dim=len(single_env.task_types),
        concept_names=single_env.concept_names,
        features_dim=128,
        net_arch=[64, 64],
        temporal_encoding="none",
        gvf_pairing=list(range(len(single_env.task_types))),
        device="cpu",
    )

    model = PPO(
        method="no_concept",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=64,
        n_epochs=2,
        batch_size=32,
        device="cpu",
        verbose=0,
    )

    # Train for just a tiny bit so model has some weights
    print("[test] running mini training...")
    model.learn(total_timesteps=256)
    print("[test] mini training done.")

    # Record video
    os.makedirs("test_videos", exist_ok=True)
    video_path = "test_videos/test_rollout.gif"
    print(f"[test] recording video to {video_path}...")

    record_rollout_from_env(
        model,
        single_env,
        video_path=video_path,
        max_steps=50,
        deterministic=True,
        seed=seed + 100,
    )

    vec_env.close()
    single_env.close()

    # Verify file exists and has reasonable size
    if os.path.exists(video_path):
        size = os.path.getsize(video_path)
        print(f"[test] SUCCESS: video created, size={size} bytes")
    else:
        print("[test] FAILED: video not created")

if __name__ == "__main__":
    main()
