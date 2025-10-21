from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import List

import imageio
import numpy as np
import torch

from vla_fastvlm.device import get_best_device


@dataclass
class AlohaSimulationConfig:
    env_id: str = "gym_aloha/AlohaInsertion-v0"
    obs_type: str = "pixels_agent_pos"
    max_episode_steps: int = 1000
    seed: int = 42
    video_dir: str = "outputs/eval"
    task_instruction: str = "Insert the peg into the socket."
    device_preference: str | None = None
    num_episodes: int = 5
    record_video: bool = True


def run_aloha_simulation(policy, config: AlohaSimulationConfig) -> List[dict]:
    """Roll out policy inside the aloha simulator and optionally save rendering."""
    import gymnasium as gym

    try:
        import gym_aloha  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "gym-aloha is required for simulation. Install with `pip install gym-aloha`."
        ) from exc

    device = get_best_device(config.device_preference)
    policy.to(device)
    policy.eval()

    if "MUJOCO_GL" not in os.environ:
        system = platform.system().lower()
        if system == "darwin":
            os.environ["MUJOCO_GL"] = "glfw"
        elif system == "windows":
            os.environ["MUJOCO_GL"] = "d3d11"
        else:
            os.environ["MUJOCO_GL"] = "egl"
    results: List[dict] = []

    video_root = Path(config.video_dir)
    if config.record_video:
        video_root.mkdir(parents=True, exist_ok=True)

    for episode in range(config.num_episodes):
        env = gym.make(
            config.env_id,
            obs_type=config.obs_type,
            max_episode_steps=config.max_episode_steps,
            render_mode="rgb_array",
        )
        observation, info = env.reset(seed=config.seed + episode)
        frames = []
        done = False
        terminated = False
        total_reward = 0.0
        step = 0
        if config.record_video:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        while not done and step < config.max_episode_steps:
            state = torch.from_numpy(observation["agent_pos"]).float()
            image = torch.from_numpy(observation["pixels"]["top"]).permute(2, 0, 1).contiguous()
            image = image.float() / 255.0

            action = policy.select_action(image, state, config.task_instruction, device=device)
            action_np = action.detach().cpu().numpy()

            observation, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward

            if config.record_video:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            done = terminated or truncated
            step += 1

        env.close()

        success = bool(info.get("is_success", False) or terminated)
        episode_result = {
            "episode": episode,
            "total_reward": float(total_reward),
            "steps": step,
            "success": success,
        }
        results.append(episode_result)

        if config.record_video and frames:
            fps = env.metadata.get("render_fps", 30)
            video_path = video_root / f"episode_{episode:03d}.mp4"
            imageio.mimsave(str(video_path), np.stack(frames), fps=fps)

    return results
