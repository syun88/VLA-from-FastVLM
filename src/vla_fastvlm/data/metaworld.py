from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Definitions are transcribed from LeRobot's Apache-2.0 metaworld_config.json.
# Keeping only the selected tasks makes the local split manifest self-contained.
FASTVLA_TASKS: dict[str, str] = {
    "assembly-v3": "Pick up a nut and place it onto a peg",
    "button-press-topdown-v3": "Press a button from the top",
    "button-press-v3": "Press a button",
    "drawer-close-v3": "Push and close a drawer",
    "drawer-open-v3": "Open a drawer",
    "pick-place-v3": "Pick and place a puck to a goal",
    "push-v3": "Push the puck to a goal",
    "reach-v3": "Reach a goal position",
    "sweep-into-v3": "Sweep a puck into a hole",
    "window-open-v3": "Push and open a window",
    "window-close-v3": "Push and close a window",
}


@dataclass(frozen=True)
class FrameRecord:
    parquet_path: str
    row_index: int
    task_name: str
    task_text: str
    episode_index: int
    frame_index: int
    timestamp: float
    state: tuple[float, ...]
    action: tuple[float, ...]
    success: bool


@dataclass
class TaskData:
    task_name: str
    instruction: str
    frames: list[FrameRecord] = field(default_factory=list)

    @property
    def episode_indices(self) -> list[int]:
        return sorted({frame.episode_index for frame in self.frames})


@dataclass(frozen=True)
class EpisodeSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }


def _instruction_to_task(instruction: str) -> str:
    for task_name, task_text in FASTVLA_TASKS.items():
        if task_text == instruction:
            return task_name
    raise KeyError(
        f"Unknown Meta-World instruction {instruction!r}. Add its mapping to FASTVLA_TASKS."
    )


def scan_metaworld(
    dataset_root: str | Path,
    tasks: Sequence[str],
    *,
    image_key: str = "observation.image",
) -> dict[str, TaskData]:
    """Read frame metadata without loading the multi-gigabyte image column."""

    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.parquet"
    if not info_path.is_file() or not tasks_path.is_file():
        raise FileNotFoundError(f"Not a LeRobot v3 dataset root: {root}")

    task_table = pq.read_table(tasks_path)
    task_rows = task_table.to_pylist()
    instruction_by_index = {
        int(row["task_index"]): str(row["__index_level_0__"]) for row in task_rows
    }
    wanted_names = set(tasks)
    selected_indices = {
        index
        for index, text in instruction_by_index.items()
        if text in {FASTVLA_TASKS[name] for name in wanted_names}
    }
    task_name_by_index = {
        index: _instruction_to_task(instruction_by_index[index])
        for index in selected_indices
    }
    missing = wanted_names.difference(task_name_by_index.values())
    if missing:
        raise KeyError(f"Tasks not found in dataset: {sorted(missing)}")

    result = {
        task_name: TaskData(task_name, FASTVLA_TASKS[task_name])
        for task_name in tasks
    }
    parquet_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No LeRobot parquet files under {root / 'data'}")

    columns = [
        "task_index",
        "episode_index",
        "frame_index",
        "timestamp",
        "observation.state",
        "action",
        "next.success",
    ]
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path, columns=columns)
        task_indices = np.asarray(table["task_index"].to_pylist(), dtype=np.int16)
        for selected in np.unique(task_indices):
            selected = int(selected)
            if selected not in selected_indices:
                continue
            rows = np.flatnonzero(task_indices == selected)
            task_name = task_name_by_index[selected]
            task_data = result[task_name]
            for row_index in rows.tolist():
                task_data.frames.append(
                    FrameRecord(
                        parquet_path=str(parquet_path),
                        row_index=row_index,
                        task_name=task_name,
                        task_text=instruction_by_index[selected],
                        episode_index=int(table["episode_index"][row_index].as_py()),
                        frame_index=int(table["frame_index"][row_index].as_py()),
                        timestamp=float(table["timestamp"][row_index].as_py()),
                        state=tuple(
                            float(value)
                            for value in table["observation.state"][row_index].as_py()
                        ),
                        action=tuple(
                            float(value) for value in table["action"][row_index].as_py()
                        ),
                        success=bool(table["next.success"][row_index].as_py()),
                    )
                )
    for task_data in result.values():
        task_data.frames.sort(key=lambda frame: (frame.episode_index, frame.frame_index))
        if not task_data.frames:
            raise ValueError(f"No frames loaded for {task_data.task_name}")
    return result


def load_or_create_split(
    task_data: TaskData,
    split_path: str | Path,
    *,
    train_episodes: int,
    validation_episodes: int,
    test_episodes: int,
    seed: int,
) -> EpisodeSplit:
    path = Path(split_path)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("task_name") != task_data.task_name:
            raise ValueError(f"Split task mismatch in {path}")
        if payload.get("seed") != seed:
            raise ValueError(f"Split seed mismatch in {path}")
        return EpisodeSplit(
            train=tuple(payload["train"]),
            validation=tuple(payload["validation"]),
            test=tuple(payload["test"]),
        )

    episodes = np.asarray(task_data.episode_indices, dtype=np.int64)
    requested = train_episodes + validation_episodes + test_episodes
    if requested > len(episodes):
        raise ValueError(
            f"Episode split requests more episodes than are available: "
            f"requested={requested}, available={len(episodes)}"
        )
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episodes)
    split = EpisodeSplit(
        train=tuple(int(x) for x in shuffled[:train_episodes]),
        validation=tuple(
            int(x)
            for x in shuffled[train_episodes : train_episodes + validation_episodes]
        ),
        test=tuple(
            int(x)
            for x in shuffled[
                train_episodes + validation_episodes : requested
            ]
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task_name": task_data.task_name,
                "instruction": task_data.instruction,
                "seed": seed,
                "unused_episodes": [int(x) for x in shuffled[requested:]],
                **split.as_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return split


def build_action_chunks(
    task_data: TaskData,
    horizon: int,
    *,
    frame_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return actions, masks, and selected frame indices with strict episode boundaries."""

    by_episode: dict[int, list[FrameRecord]] = defaultdict(list)
    for frame in task_data.frames:
        by_episode[frame.episode_index].append(frame)
    action_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    frame_indices: list[int] = []
    frame_position = {
        (frame.episode_index, frame.frame_index): index
        for index, frame in enumerate(task_data.frames)
    }
    for episode in sorted(by_episode):
        frames = sorted(by_episode[episode], key=lambda frame: frame.frame_index)
        episode_actions = np.asarray([frame.action for frame in frames], dtype=np.float32)
        for local_index in range(0, len(frames), frame_stride):
            chunk = episode_actions[local_index : local_index + horizon]
            mask = np.zeros(horizon, dtype=np.float32)
            mask[: len(chunk)] = 1.0
            if len(chunk) < horizon:
                padded = np.zeros((horizon, episode_actions.shape[1]), dtype=np.float32)
                padded[: len(chunk)] = chunk
                chunk = padded
            action_rows.append(chunk)
            mask_rows.append(mask)
            frame_indices.append(
                frame_position[(episode, frames[local_index].frame_index)]
            )
    return np.stack(action_rows), np.stack(mask_rows), np.asarray(frame_indices, dtype=np.int64)


def compute_task_normalization(
    task_data: TaskData,
    train_episodes: Iterable[int],
    horizon: int,
) -> dict[str, list[float] | int]:
    train_set = set(train_episodes)
    states = np.asarray(
        [frame.state for frame in task_data.frames if frame.episode_index in train_set],
        dtype=np.float32,
    )
    actions, _, _ = build_action_chunks(
        TaskData(
            task_name=task_data.task_name,
            instruction=task_data.instruction,
            frames=[
                frame for frame in task_data.frames if frame.episode_index in train_set
            ],
        ),
        horizon=horizon,
    )
    flat_actions = actions.reshape(-1, actions.shape[-1])
    return {
        "state_mean": states.mean(axis=0).tolist(),
        "state_std": np.maximum(states.std(axis=0), 1e-6).tolist(),
        "action_mean": flat_actions.mean(axis=0).tolist(),
        "action_std": np.maximum(flat_actions.std(axis=0), 1e-6).tolist(),
        "state_dim": int(states.shape[-1]),
        "action_dim": int(flat_actions.shape[-1]),
        "train_episodes": int(len(train_set)),
    }


@dataclass
class CachedFeatureDataset:
    """Memory-mapped observation/action arrays emitted by cache_features.py."""

    task_dir: str | Path

    def __post_init__(self) -> None:
        self.root = Path(self.task_dir)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing feature-cache manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.episode_index = np.load(self.root / "episode_index.npy", mmap_mode="r")
        self.frame_index = np.load(self.root / "frame_index.npy", mmap_mode="r")
        self.states = np.load(self.root / "states.npy", mmap_mode="r")
        self.actions = np.load(self.root / "actions.npy", mmap_mode="r")
        self.action_masks = np.load(self.root / "action_masks.npy", mmap_mode="r")
        self.input_ids = np.load(self.root / "input_ids.npy", mmap_mode="r")
        self.attention_mask = np.load(self.root / "attention_mask.npy", mmap_mode="r")
        if self.manifest.get("has_visual", True):
            self.visual_tokens = np.load(
                self.root / "visual_tokens.npy", mmap_mode="r"
            )
            self.global_hidden = np.load(
                self.root / "global_hidden.npy", mmap_mode="r"
            )
        else:
            self.visual_tokens = None
            self.global_hidden = None
        normalization_path = self.root / "normalization.json"
        self.normalization = json.loads(normalization_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return int(self.episode_index.shape[0])

    def select_episodes(self, episodes: Iterable[int]) -> np.ndarray:
        allowed = set(int(episode) for episode in episodes)
        return np.asarray(
            [i for i, episode in enumerate(self.episode_index) if int(episode) in allowed],
            dtype=np.int64,
        )

    @property
    def instruction(self) -> str:
        return str(self.manifest["instruction"])


def iter_frames_with_images(
    task_data: TaskData,
    *,
    image_key: str = "observation.image",
) -> Iterator[tuple[FrameRecord, bytes]]:
    """Yield selected PNG bytes while reading each parquet file only once."""

    by_file: dict[str, list[FrameRecord]] = defaultdict(list)
    frame_lookup: dict[tuple[str, int], FrameRecord] = {}
    for frame in task_data.frames:
        by_file[frame.parquet_path].append(frame)
        frame_lookup[(frame.parquet_path, frame.row_index)] = frame
    for parquet_path in sorted(by_file):
        table = pq.read_table(parquet_path, columns=[image_key])
        image_values = table[image_key]
        for frame in sorted(by_file[parquet_path], key=lambda item: (item.episode_index, item.frame_index)):
            value = image_values[frame.row_index].as_py()
            if isinstance(value, dict):
                data = value.get("bytes")
            else:
                data = value
            if not data:
                raise ValueError(
                    f"Missing embedded image bytes at {parquet_path}:{frame.row_index}"
                )
            yield frame_lookup[(parquet_path, frame.row_index)], data
