#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from vla_fastvlm.data import FASTVLA_TASKS
from vla_fastvlm.evaluation.metaworld import MetaWorldAdapter
from vla_fastvlm.fastvla import MLXFastVLAPolicy
from vla_fastvlm.paths import runs_root


class LiveViewer:
    def __init__(self, task: str, checkpoint: Path) -> None:
        self.window = "FastVLA live inference"
        self.task = task
        self.checkpoint = str(checkpoint)
        self.closed = False
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 820, 720)
        cv2.moveWindow(self.window, 40, 40)
        cv2.setWindowProperty(self.window, cv2.WND_PROP_TOPMOST, 1)

    def show(self, frame: np.ndarray, message: str, step: int, max_steps: int) -> None:
        if self.closed:
            return
        image = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            (640, 640),
            interpolation=cv2.INTER_CUBIC,
        )
        canvas = np.zeros((720, 820, 3), dtype=np.uint8)
        canvas[:] = (18, 19, 24)
        canvas[60:700, 90:730] = image
        cv2.putText(
            canvas,
            "FastVLA / FastVLM-0.5B + LoRA",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 244, 248),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            self.task,
            (20, 760),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 180, 205),
            1,
            cv2.LINE_AA,
        )
        # Use a compact status band over the image instead of a second toolkit.
        overlay = canvas.copy()
        cv2.rectangle(overlay, (90, 610), (730, 700), (0, 0, 0), -1)
        canvas = cv2.addWeighted(overlay, 0.68, canvas, 0.32, 0)
        for index, line in enumerate(message.splitlines()):
            cv2.putText(
                canvas,
                line[:100],
                (105, 638 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (235, 239, 245),
                1,
                cv2.LINE_AA,
            )
        fraction = min(1.0, step / max(1, max_steps))
        cv2.rectangle(canvas, (90, 690), (730, 700), (55, 60, 70), -1)
        cv2.rectangle(canvas, (90, 690), (90 + int(640 * fraction), 700), (40, 180, 240), -1)
        cv2.imshow(self.window, canvas)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            self.close()

    def wait(self) -> None:
        while not self.closed:
            key = cv2.waitKey(100) & 0xFF
            if key in (27, ord("q")):
                break
        self.close()

    def close(self) -> None:
        self.closed = True
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(runs_root() / "fastvlm-lora-seed42" / "best"),
    )
    parser.add_argument("--task", default="button-press-topdown-v3")
    parser.add_argument("--seed", type=int, default=300001)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--display-fps", type=float, default=30.0)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    task_metadata = metadata["dataset"]["tasks"][args.task]
    norm = task_metadata["normalization"]
    action_mean = np.asarray(norm["action_mean"], dtype=np.float32)
    action_std = np.asarray(norm["action_std"], dtype=np.float32)
    state_mean = np.asarray(norm["state_mean"], dtype=np.float32)
    state_std = np.asarray(norm["state_std"], dtype=np.float32)

    viewer = LiveViewer(args.task, checkpoint)
    viewer.show(
        np.zeros((480, 480, 3), dtype=np.uint8),
        "Loading MLX checkpoint and simulator…",
        0,
        args.max_steps,
    )
    policy = MLXFastVLAPolicy.load(checkpoint, load_backbone=True)
    env = MetaWorldAdapter(args.task, max_steps=args.max_steps)
    frame, state = env.reset(args.seed)
    viewer.show(frame, "Ready — starting policy inference", 0, args.max_steps)

    queue: list[np.ndarray] = []
    timing = {}
    chunk_index = 0
    success = False
    frame_delay = 1.0 / args.display_fps
    try:
        for step in range(args.max_steps):
            if step % args.replan_steps == 0:
                chunk_index += 1
                viewer.show(
                    frame,
                    f"FastVLM inference at control step {step}…\n"
                    f"task: {FASTVLA_TASKS[args.task]}",
                    step,
                    args.max_steps,
                )
                chunk, timing = policy.predict_chunk(
                    frame,
                    (state - state_mean) / state_std,
                    FASTVLA_TASKS[args.task],
                )
                queue = list(chunk * action_std + action_mean)
                print(
                    f"chunk={chunk_index} step={step} "
                    f"first_action={np.round(queue[0], 4).tolist()} "
                    f"inference_s={timing['end_to_end_seconds']:.3f} "
                    f"frame_sha256={hashlib.sha256(frame.tobytes()).hexdigest()[:16]}",
                    flush=True,
                )
            action = queue.pop(0)
            frame, state, terminated, truncated, info = env.step(action, render=False)
            frame = env.render_observation()
            diagnostics = env.diagnostics()
            viewer.show(
                frame,
                f"LIVE MuJoCo | chunk {chunk_index} | step {step + 1:03d} | "
                f"inference {timing.get('end_to_end_seconds', 0):.3f}s\n"
                f"action=[{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}, {action[3]:.2f}]\n"
                f"object-to-target |Δz|={diagnostics.get('object_to_target_abs_z', float('nan')):.5f}\n"
                f"success={bool(info.get('success', False))}",
                step + 1,
                args.max_steps,
            )
            if (step + 1) % 10 == 0:
                print(
                    f"step={step + 1} success={bool(info.get('success', False))} "
                    f"inference={timing.get('end_to_end_seconds', 0):.3f}s",
                    flush=True,
                )
            time.sleep(frame_delay)
            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        env.close()

    viewer.show(
        frame,
        ("SUCCESS" if success else "FAILED / TIMEOUT")
        + f" | steps={step + 1} | seed={args.seed}\n"
        + "Close this window or press q when finished.",
        step + 1,
        args.max_steps,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "task": args.task,
                "seed": args.seed,
                "success": success,
                "steps": step + 1,
                "final_inference_seconds": timing.get("end_to_end_seconds"),
            }
        ),
        flush=True,
    )
    viewer.wait()


if __name__ == "__main__":
    main()
