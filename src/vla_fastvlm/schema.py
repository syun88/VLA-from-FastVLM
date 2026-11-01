from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CameraSpec:
    name: str
    source_key: str
    width: int
    height: int
    color_space: str = "RGB"


@dataclass(frozen=True)
class RobotSchema:
    """Robot-specific observation/action contract outside the model implementation.

    A Dual Scorpion adapter can declare two arms, several cameras, joint/gripper names, and
    its control frequency without changing FastVLM or the action-head math.
    """

    robot_type: str
    cameras: tuple[CameraSpec, ...]
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    control_frequency_hz: float
    action_horizon: int
    action_space: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(camera.name for camera in self.cameras)

    @property
    def state_dim(self) -> int:
        return len(self.state_names)

    @property
    def action_dim(self) -> int:
        return len(self.action_names)

    def validate(self) -> None:
        if not self.cameras:
            raise ValueError("RobotSchema requires at least one camera")
        if not self.state_names or not self.action_names:
            raise ValueError("RobotSchema requires state and action names")
        if len(set(self.camera_names)) != len(self.cameras):
            raise ValueError("Camera names must be unique")
        if self.control_frequency_hz <= 0 or self.action_horizon < 1:
            raise ValueError("Invalid control frequency or action horizon")

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_type": self.robot_type,
            "cameras": [camera.__dict__ for camera in self.cameras],
            "state_names": list(self.state_names),
            "action_names": list(self.action_names),
            "control_frequency_hz": self.control_frequency_hz,
            "action_horizon": self.action_horizon,
            "action_space": self.action_space,
            "metadata": self.metadata,
        }
