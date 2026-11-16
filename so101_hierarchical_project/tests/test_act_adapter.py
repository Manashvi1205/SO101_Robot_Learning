import numpy as np
import pytest

from so101_hierarchical.act_adapter import (
    ACTPolicyAdapter, ObservationError, inspect_checkpoint, validate_saved_contract,
)
from so101_hierarchical.registry import build_default_registry


def test_complete_checkpoints_pass_structural_validation():
    for skill in build_default_registry():
        if skill.implementation_type != "act_policy":
            continue
        result = inspect_checkpoint(skill.checkpoint_path, validate_weights=True)
        assert result.complete, result.to_dict()
        assert result.safetensors_valid is True
        validate_saved_contract(skill)


def test_observation_validation_requires_exact_camera_keys():
    skill = build_default_registry().get("pick_place_variable")
    adapter = ACTPolicyAdapter(skill)
    observation = {
        "observation.state": np.zeros(6, dtype=np.float32),
        "observation.images.wrist": np.zeros((3, 720, 1280), dtype=np.float32),
        "observation.images.front": np.zeros((3, 720, 1280), dtype=np.float32),
    }
    with pytest.raises(ObservationError, match="phone"):
        adapter.validate_observation(observation)


def test_observation_validation_accepts_hwc_images():
    skill = build_default_registry().get("pick_place_fixed")
    adapter = ACTPolicyAdapter(skill)
    observation = {
        "observation.state": np.zeros(6, dtype=np.float32),
        "observation.images.wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation.images.front": np.zeros((720, 1280, 3), dtype=np.uint8),
    }
    adapter.validate_observation(observation)

