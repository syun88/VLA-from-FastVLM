"""
VLA-from-FastVLM
================

MLX-native research package for adapting Apple's FastVLM to robotic action chunking.
"""

from .fastvla.configuration_fastvla import FastVLAConfig
from .fastvla.modeling_fastvla import ActionPolicy, MLXFastVLAPolicy
from .schema import CameraSpec, RobotSchema

__all__ = [
    "FastVLAConfig",
    "ActionPolicy",
    "MLXFastVLAPolicy",
    "CameraSpec",
    "RobotSchema",
]
