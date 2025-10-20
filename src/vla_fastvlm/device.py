import os
from typing import Literal

import torch

DeviceLiteral = Literal["cuda", "mps", "cpu"]


def is_cuda_available() -> bool:
    """Return True if CUDA is available and not disabled."""
    disabled = os.environ.get("FASTVLM_FORCE_DEVICE", "").lower() == "cpu"
    return torch.cuda.is_available() and not disabled


def is_mps_available() -> bool:
    """Return True if Apple's Metal Performance Shaders backend can be used."""
    disabled = os.environ.get("FASTVLM_FORCE_DEVICE", "").lower() == "cpu"
    return torch.backends.mps.is_available() and not disabled


def get_best_device(preferred: DeviceLiteral | None = None) -> torch.device:
    """
    Pick the most capable runtime device.

    Priority:
    1. User-provided preferred argument if available.
    2. CUDA GPU.
    3. Apple MPS.
    4. CPU.
    """
    if preferred:
        preferred = preferred.lower()  # type: ignore[assignment]
    if preferred == "cuda" and is_cuda_available():
        return torch.device("cuda")
    if preferred == "mps" and is_mps_available():
        return torch.device("mps")

    if is_cuda_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")

    return torch.device("cpu")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Recursively move tensors in the batch to the target device."""
    result: dict = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, dict):
            result[key] = move_batch_to_device(value, device)
        else:
            result[key] = value
    return result
