from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, Optional

import torch
from datasets import IterableDataset as HFDIterableDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset

ImageTransform = Callable[[torch.Tensor], torch.Tensor]
StateTransform = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class AlohaSample:
    """Single record from the LeRobot aloha insertion dataset."""

    image: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    task: str
    metadata: Dict[str, torch.Tensor]


def default_aloha_transforms(image: torch.Tensor) -> torch.Tensor:
    """
    Convert the dataset image tensor to the format expected by FastVLM.

    The raw dataset stores images as (C, H, W) float32 in range [0, 255].
    We standardise to float32 in [0, 1].
    """
    if image.dtype != torch.float32:
        image = image.to(torch.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


class AlohaDataset(Dataset[AlohaSample]):
    """Finite dataset wrapper for local training."""

    def __init__(
        self,
        split: str = "train",
        repo_id: str = "lerobot/aloha_sim_insertion_human_image",
        cache_dir: Optional[str] = None,
        image_key: str = "observation.images.top",
        state_key: str = "observation.state",
        action_key: str = "action",
        task_key: str = "task",
        image_transform: ImageTransform = default_aloha_transforms,
        state_transform: StateTransform | None = None,
        limit_samples: int | None = None,
    ) -> None:
        dataset = load_dataset(
            repo_id,
            split=split,
            cache_dir=cache_dir,
        )
        dataset.set_format(type="torch")
        if limit_samples is not None:
            dataset = dataset.select(range(limit_samples))
        self._dataset = dataset
        self._image_key = image_key
        self._state_key = state_key
        self._action_key = action_key
        self._task_key = task_key
        self._image_transform = image_transform
        self._state_transform = state_transform

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._dataset)

    def __getitem__(self, index: int) -> AlohaSample:  # type: ignore[override]
        record = self._dataset[index]

        image = record[self._image_key]
        state = record[self._state_key]
        action = record[self._action_key]
        task = record[self._task_key]

        image = self._image_transform(image)
        if self._state_transform is not None:
            state = self._state_transform(state)

        metadata = {
            "episode_index": record.get("episode_index"),
            "frame_index": record.get("frame_index"),
            "timestamp": record.get("timestamp"),
            "index": record.get("index"),
            "task_index": record.get("task_index"),
        }

        return AlohaSample(
            image=image,
            state=state.to(torch.float32),
            action=action.to(torch.float32),
            task=task,
            metadata=metadata,
        )


class AlohaIterableDataset(IterableDataset[AlohaSample]):
    """
    Streaming dataset wrapper to avoid downloading the full LeRobot dataset.
    """

    def __init__(
        self,
        split: str = "train",
        repo_id: str = "lerobot/aloha_sim_insertion_human_image",
        cache_dir: Optional[str] = None,
        image_key: str = "observation.images.top",
        state_key: str = "observation.state",
        action_key: str = "action",
        task_key: str = "task",
        image_transform: ImageTransform = default_aloha_transforms,
        state_transform: StateTransform | None = None,
    ) -> None:
        dataset = load_dataset(
            repo_id,
            split=split,
            cache_dir=cache_dir,
            streaming=True,
        )
        if not isinstance(dataset, HFDIterableDataset):
            raise RuntimeError("Expected iterable dataset when streaming=True.")
        self._dataset = dataset
        self._image_key = image_key
        self._state_key = state_key
        self._action_key = action_key
        self._task_key = task_key
        self._image_transform = image_transform
        self._state_transform = state_transform

    def __iter__(self) -> Iterator[AlohaSample]:
        for record in self._dataset:
            image = record[self._image_key]
            if isinstance(image, torch.Tensor):
                pass
            else:
                image = torch.tensor(image)

            state = record[self._state_key]
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state)

            action = record[self._action_key]
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action)

            task = record[self._task_key]

            image = self._image_transform(image)
            if self._state_transform is not None:
                state = self._state_transform(state)

            metadata = {
                "episode_index": torch.tensor(record.get("episode_index"))
                if record.get("episode_index") is not None
                else None,
                "frame_index": torch.tensor(record.get("frame_index"))
                if record.get("frame_index") is not None
                else None,
                "timestamp": torch.tensor(record.get("timestamp"))
                if record.get("timestamp") is not None
                else None,
                "index": torch.tensor(record.get("index"))
                if record.get("index") is not None
                else None,
                "task_index": torch.tensor(record.get("task_index"))
                if record.get("task_index") is not None
                else None,
            }
            yield AlohaSample(
                image=image.to(torch.float32),
                state=state.to(torch.float32),
                action=action.to(torch.float32),
                task=task,
                metadata=metadata,
            )


def create_aloha_dataloader(
    dataset: Dataset[AlohaSample] | IterableDataset[AlohaSample],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Construct a PyTorch dataloader that yields dictionaries ready for FastVLM training.
    """

    def collate_fn(batch: Iterable[AlohaSample]) -> Dict[str, torch.Tensor | list]:
        batch_list = list(batch)
        images = torch.stack([sample.image for sample in batch_list])
        states = torch.stack([sample.state for sample in batch_list])
        actions = torch.stack([sample.action for sample in batch_list])
        tasks = [sample.task for sample in batch_list]
        metadata = [sample.metadata for sample in batch_list]
        return {
            "images": images,
            "states": states,
            "actions": actions,
            "tasks": tasks,
            "metadata": metadata,
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if isinstance(dataset, Dataset) else False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
