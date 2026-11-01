from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vla_fastvlm.data import FASTVLA_TASKS
from vla_fastvlm.fastvla import MLXFastVLAPolicy
from vla_fastvlm.tracking import ExperimentTracker, mlx_memory_stats


class MetaWorldAdapter:
    """Thin Meta-World v3 adapter matching LeRobot's camera and observation contract."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task: str,
        *,
        camera_name: str = "corner2",
        image_width: int = 480,
        image_height: int = 480,
        max_steps: int = 500,
    ) -> None:
        import metaworld

        if task not in FASTVLA_TASKS:
            raise KeyError(f"Unknown FastVLA Meta-World task: {task}")
        self.task = task
        self.instruction = FASTVLA_TASKS[task]
        self.camera_name = camera_name
        self.image_width = image_width
        self.image_height = image_height
        self.max_steps = max_steps
        self._env = None
        self._metaworld = metaworld
        self._episode_steps = 0
        self._last_raw_obs: np.ndarray | None = None

    def _ensure(self) -> None:
        if self._env is not None:
            return
        mt1 = self._metaworld.MT1(self.task, seed=42)
        env = mt1.train_classes[self.task](
            render_mode="rgb_array", camera_name=self.camera_name
        )
        env.set_task(mt1.train_tasks[0])
        if self.camera_name == "corner2":
            env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        env.reset()
        env._freeze_rand_vec = False
        env.seeded_rand_vec = True
        self._env = env

    def reset(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        self._ensure()
        if self._env is None:
            raise RuntimeError("Meta-World environment failed to initialize")
        self._env.seed(seed)
        raw_obs, _ = self._env.reset(seed=seed)
        self._episode_steps = 0
        self._last_raw_obs = np.asarray(raw_obs, dtype=np.float64)
        return self._format_observation(raw_obs)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        self._ensure()
        if self._env is None:
            raise RuntimeError("Meta-World environment failed to initialize")
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (4,):
            raise ValueError(f"Meta-World adapter expects a 4-D action, got {action.shape}")
        raw_obs, reward, done, truncated, info = self._env.step(action)
        self._last_raw_obs = np.asarray(raw_obs, dtype=np.float64)
        self._episode_steps += 1
        success = bool(info.get("success", 0))
        timed_out = self._episode_steps >= self.max_steps
        image, state = self._format_observation(raw_obs)
        return image, state, success or bool(done), bool(truncated or timed_out), {
            "success": success,
            "reward": float(reward),
            "done": bool(done),
            "truncated": bool(truncated),
            "timed_out": timed_out,
            "manipulation_error": self._manipulation_error(),
        }

    def _format_observation(self, raw_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._env is None:
            raise RuntimeError("Meta-World environment failed to initialize")
        image = np.asarray(self._env.render())
        if self.camera_name == "corner2":
            image = np.flip(image, (0, 1))
        if image.shape[:2] != (self.image_height, self.image_width):
            raise ValueError(
                f"Rendered image {image.shape[:2]} does not match configured "
                f"{(self.image_height, self.image_width)}"
            )
        return image.copy(), np.asarray(raw_obs[:4], dtype=np.float32)

    def _manipulation_error(self) -> float:
        if self._env is None:
            return float("nan")
        try:
            if self._last_raw_obs is None:
                return float("nan")
            gripper_xyz = np.asarray(self._last_raw_obs[:3], dtype=np.float64)
            target = getattr(self._env, "_target_pos", None)
            if target is None:
                target = getattr(self._env, "goal", None)
            if target is not None:
                return float(
                    np.linalg.norm(gripper_xyz - np.asarray(target, dtype=np.float64)[:3])
                )
        except Exception:
            pass
        return float("nan")

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def _save_video(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=7)


def _counterfactual_instruction(task: str, rotated_tasks: tuple[str, ...]) -> str:
    others = [other for other in rotated_tasks if other != task]
    return FASTVLA_TASKS[others[0] if others else task]


def evaluate_policy(
    checkpoint_dir: str | Path,
    *,
    tasks: tuple[str, ...],
    episodes: int,
    replan_steps: int,
    max_steps: int,
    seed_start: int,
    output_dir: str | Path,
    tracker: ExperimentTracker,
    counterfactual: bool = False,
    save_videos: int = 3,
    image_resolution: int = 1024,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    policy = MLXFastVLAPolicy.load(checkpoint_dir, load_backbone=True)
    if policy.config.uses_vlm and hasattr(policy.backbone, "set_image_resolution"):
        policy.backbone.set_image_resolution(image_resolution)
    metadata = json.loads((Path(checkpoint_dir) / "metadata.json").read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episode_log = output / "episodes.jsonl"
    all_results: list[dict[str, Any]] = []
    with episode_log.open("w", encoding="utf-8") as log_file:
        for task_offset, task in enumerate(tasks):
            instruction = (
                _counterfactual_instruction(task, tasks)
                if counterfactual
                else FASTVLA_TASKS[task]
            )
            normalization = metadata["dataset"]["tasks"][task]["normalization"]
            action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
            action_std = np.asarray(normalization["action_std"], dtype=np.float32)
            state_mean = np.asarray(normalization["state_mean"], dtype=np.float32)
            state_std = np.asarray(normalization["state_std"], dtype=np.float32)
            env = MetaWorldAdapter(task, max_steps=max_steps)
            successes = 0
            failed = 0
            latencies: list[float] = []
            chunk_latencies: list[float] = []
            errors: list[float] = []
            stabilities: list[float] = []
            replan_jumps: list[float] = []
            failures: dict[str, int] = {}
            videos_saved = 0
            try:
                for episode in range(episodes):
                    seed = seed_start + task_offset * 100000 + episode
                    image, state = env.reset(seed)
                    frames: list[np.ndarray] = []
                    executed: list[np.ndarray] = []
                    success = False
                    reason = "timeout"
                    initial_error = float("nan")
                    final_error = float("nan")
                    for step in range(max_steps):
                        if step % replan_steps == 0:
                            chunk, timing = policy.predict_chunk(
                                image,
                                (state - state_mean) / state_std,
                                instruction,
                            )
                            chunk = chunk * action_std + action_mean
                            queue = list(chunk)
                            chunk_latencies.append(timing["end_to_end_seconds"])
                        action = np.asarray(queue.pop(0), dtype=np.float32)
                        if step > 0 and step % replan_steps == 0:
                            replan_jumps.append(
                                float(np.linalg.norm(action - executed[-1]))
                            )
                        image, state, terminated, truncated, info = env.step(action)
                        executed.append(action)
                        if save_videos and episode < save_videos:
                            frames.append(image)
                        step_latency = timing["end_to_end_seconds"] / max(1, min(replan_steps, len(chunk)))
                        latencies.append(step_latency)
                        error = float(info.get("manipulation_error", float("nan")))
                        if np.isfinite(error) and not np.isfinite(initial_error):
                            initial_error = error
                        if np.isfinite(error):
                            final_error = error
                        if terminated or truncated:
                            success = bool(info.get("success", False))
                            if success:
                                reason = "success"
                            elif truncated:
                                reason = "timeout"
                            else:
                                reason = "terminated_without_success"
                            break
                    if success:
                        successes += 1
                    else:
                        failed += 1
                        if np.isfinite(initial_error) and np.isfinite(final_error):
                            if final_error >= initial_error * 0.95:
                                reason = "stalled_or_worsened"
                        failures[reason] = failures.get(reason, 0) + 1
                    if len(executed) > 1:
                        action_array = np.stack(executed)
                        stabilities.append(
                            float(np.mean(np.linalg.norm(np.diff(action_array, axis=0), axis=-1)))
                        )
                    errors.append(final_error)
                    result = {
                        "task": task,
                        "instruction": instruction,
                        "episode": episode,
                        "seed": seed,
                        "success": success,
                        "failure_reason": None if success else reason,
                        "steps": len(executed),
                        "initial_manipulation_error": initial_error,
                        "final_manipulation_error": final_error,
                        "action_stability_l1": stabilities[-1] if stabilities else 0.0,
                        "counterfactual": counterfactual,
                    }
                    all_results.append(result)
                    log_file.write(json.dumps(result) + "\n")
                    log_file.flush()
                    if save_videos and episode < save_videos:
                        label = "success" if success else f"failure-{reason}"
                        _save_video(
                            output / "videos" / f"{task}-ep{episode}-{label}.mp4",
                            frames,
                        )
                        if videos_saved == 0:
                            tracker.log_video(
                                output / "videos" / f"{task}-ep{episode}-{label}.mp4",
                                key=f"rollout/{task}",
                                caption=f"{task}: {label}; seed={seed}; instruction={instruction}",
                            )
                        videos_saved += 1
                    if (episode + 1) % 10 == 0 or episode + 1 == episodes:
                        progress(
                            f"{task}: {episode + 1}/{episodes} episodes, "
                            f"success={successes}/{episode + 1}"
                        )
                        tracker.log(
                            {
                                f"eval/{task}/episodes": episode + 1,
                                f"eval/{task}/success_rate": successes / (episode + 1),
                                f"eval/{task}/successes": successes,
                                f"eval/{task}/failures": failed,
                                **{f"memory/{k}": v for k, v in mlx_memory_stats().items()},
                            }
                        )
            finally:
                env.close()
            task_summary = {
                "task": task,
                "instruction": instruction,
                "counterfactual": counterfactual,
                "episodes": episodes,
                "successes": successes,
                "failures": failed,
                "success_rate": successes / max(1, episodes),
                "failure_categories": failures,
                "mean_final_manipulation_error": float(np.nanmean(errors)) if errors else None,
                "mean_action_stability_l1": float(np.mean(stabilities)) if stabilities else None,
                "mean_replan_action_jump_l1": float(np.mean(replan_jumps))
                if replan_jumps
                else None,
                "latency_chunk_p50_seconds": float(np.percentile(chunk_latencies, 50)),
                "latency_chunk_p95_seconds": float(np.percentile(chunk_latencies, 95)),
                "latency_action_p50_seconds": float(np.percentile(latencies, 50)),
                "latency_action_p95_seconds": float(np.percentile(latencies, 95)),
                "effective_policy_update_frequency_hz": 1.0
                / max(1e-9, float(np.percentile(chunk_latencies, 50))),
                "replans_per_episode_estimate": max_steps / replan_steps,
                "videos": videos_saved,
            }
            tracker.log(
                {
                    f"eval/{task}/{key}": value
                    for key, value in task_summary.items()
                    if isinstance(value, (int, float)) and value is not None
                }
            )
    summary = {
        "checkpoint": str(checkpoint_dir),
        "tasks": tasks,
        "episodes_per_task": episodes,
        "total_episodes": len(all_results),
        "successes": sum(int(result["success"]) for result in all_results),
        "success_rate": sum(int(result["success"]) for result in all_results)
        / max(1, len(all_results)),
        "counterfactual": counterfactual,
        "image_resolution": image_resolution,
        "task_results": {
            task: [result for result in all_results if result["task"] == task]
            for task in tasks
        },
        **tracker.provenance(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
