from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from vla_fastvlm.fastvla import ActionPolicy, FastVLAConfig, MLXFastVLAPolicy
from vla_fastvlm.schema import CameraSpec, RobotSchema


def _config() -> FastVLAConfig:
    return FastVLAConfig(
        backbone_mode="frozen",
        state_dim=4,
        action_dim=4,
        action_horizon=8,
        head_hidden_dim=16,
        visual_hidden_size=8,
        spatial_grid=2,
        spatial_token_dim=4,
    )


def test_action_policy_output_and_gradient_update() -> None:
    config = _config()
    policy = ActionPolicy(config, visual_hidden_size=8)
    batch = {
        "visual_tokens": mx.array(np.random.default_rng(1).normal(size=(2, 4, 8)).astype(np.float32)),
        "global_hidden": mx.array(np.random.default_rng(2).normal(size=(2, 8)).astype(np.float32)),
        "state": mx.array(np.random.default_rng(3).normal(size=(2, 4)).astype(np.float32)),
        "actions": mx.array(np.random.default_rng(4).normal(size=(2, 8, 4)).astype(np.float32)),
        "action_mask": mx.ones((2, 8)),
    }

    def loss_fn(_batch):
        predictions = policy(
            visual_tokens=_batch["visual_tokens"],
            global_hidden=_batch["global_hidden"],
            state=_batch["state"],
        )
        return policy.loss(predictions, _batch["actions"], _batch["action_mask"])

    value_and_grad = nn.value_and_grad(policy, loss_fn)
    (loss, _), gradients = value_and_grad(batch)
    optimizer = optim.AdamW(learning_rate=1e-2)
    before = np.asarray(policy.action_projection.weight)
    optimizer.update(policy, gradients)
    mx.eval(policy.parameters(), loss)
    after = np.asarray(policy.action_projection.weight)

    assert np.isfinite(float(loss.item()))
    assert not np.array_equal(before, after)


def test_checkpoint_roundtrip(tmp_path) -> None:
    config = _config()
    policy = MLXFastVLAPolicy(config, load_backbone=False)
    policy.save(tmp_path, {"test": True})

    restored = MLXFastVLAPolicy.load(tmp_path, load_backbone=False)
    assert restored.action_policy is not None
    assert restored.config.to_dict() == config.to_dict()


def test_multi_camera_and_non_metaworld_action_dimensions() -> None:
    config = FastVLAConfig(
        backbone_mode="frozen",
        camera_names=("wrist", "overhead"),
        state_dim=14,
        action_dim=7,
        action_horizon=16,
        head_hidden_dim=16,
        visual_hidden_size=8,
        spatial_grid=2,
        spatial_token_dim=4,
    )
    policy = ActionPolicy(config, visual_hidden_size=8)
    output = policy(
        visual_tokens=mx.zeros((2, 2, 4, 8)),
        global_hidden=mx.zeros((2, 2, 8)),
        state=mx.zeros((2, 14)),
    )

    assert output.shape == (2, 16, 7)


def test_dual_arm_robot_schema_is_serializable() -> None:
    schema = RobotSchema(
        robot_type="dual-scorpion",
        cameras=(CameraSpec("left", "cam.left", 640, 480), CameraSpec("right", "cam.right", 640, 480)),
        state_names=tuple(f"joint_{i}" for i in range(12)),
        action_names=tuple(f"joint_cmd_{i}" for i in range(12)),
        control_frequency_hz=30,
        action_horizon=16,
        action_space="joint_position",
    )
    schema.validate()
    assert schema.to_dict()["cameras"][0]["name"] == "left"
    assert schema.state_dim == schema.action_dim == 12
