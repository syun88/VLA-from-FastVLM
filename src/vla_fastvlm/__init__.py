"""
VLA-from-FastVLM
================

Python package that adapts Apple's FastVLM vision-language model with an action head for
robotic manipulation tasks trained on LeRobot datasets.
"""

from .device import get_best_device, is_cuda_available, is_mps_available
from .model.policy import FastVLMPolicy

__all__ = [
    "get_best_device",
    "is_cuda_available",
    "is_mps_available",
    "FastVLMPolicy",
]
