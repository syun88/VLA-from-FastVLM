
import json
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from vla_fastvlm.device import get_best_device
from vla_fastvlm.fastvla import FastVLAConfig, FastVLAPolicy
from vla_fastvlm.model.policy import FastVLMPolicy, FastVLMPolicyConfig
from vla_fastvlm.model.fastvlm_adapter import FastVLMBackboneConfig


def load_policy_from_checkpoint(
    checkpoint_dir: str | Path,
    device_preference: Optional[str] = None,
    strict: bool = True,
) -> Tuple[Union[FastVLMPolicy, FastVLAPolicy], torch.device]:
    """Load FastVLM policy weights from a checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    config_path = checkpoint_path / "policy_config.json"
    weights_path = checkpoint_path / "policy_state_dict.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing policy_config.json in {checkpoint_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing policy_state_dict.pt in {checkpoint_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    # Detect which policy to instantiate based on stored config keys.
    if "vlm_model_name" in config_dict:
        policy_cfg = FastVLAConfig(**config_dict)
        policy = FastVLAPolicy(policy_cfg)
    else:
        backbone_cfg = FastVLMBackboneConfig(**config_dict.pop("backbone"))
        policy_cfg = FastVLMPolicyConfig(backbone=backbone_cfg, **config_dict)
        policy = FastVLMPolicy(policy_cfg)

    state_dict = torch.load(weights_path, map_location="cpu")
    policy.load_state_dict(state_dict, strict=strict)

    device = get_best_device(device_preference)
    policy.to(device)
    policy.eval()
    return policy, device
