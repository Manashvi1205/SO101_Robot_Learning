from pathlib import Path

from so101_hierarchical.registry import build_default_registry


def test_default_registry_contains_only_verified_skills():
    registry = build_default_registry()
    assert registry.names() == [
        "detect_blocks", "go_home", "go_inspection", "identify_color",
        "pick_place_fixed", "pick_place_variable", "verify_placement",
    ]
    assert registry.validate_assets() == {}


def test_variable_policy_preserves_phone_key_and_resolution():
    skill = build_default_registry().get("pick_place_variable")
    cameras = {camera.key: camera.chw for camera in skill.cameras}
    assert cameras == {
        "observation.images.wrist": (3, 720, 1280),
        "observation.images.phone": (3, 720, 1280),
    }
    assert skill.camera_aliases == {"observation.images.front": "observation.images.phone"}


def test_registry_does_not_reference_corrupted_100k_checkpoint():
    for skill in build_default_registry():
        if skill.implementation_type != "act_policy":
            continue
        assert "100000" not in skill.checkpoint_path
        assert Path(skill.checkpoint_path).name == "pretrained_model"


def test_pose_skills_are_not_act_checkpoints():
    registry = build_default_registry()
    assert registry.get("go_home").implementation_type == "joint_trajectory"
    assert registry.get("go_inspection").supported_targets == ("inspection",)


def test_every_skill_exposes_execution_contract():
    for skill in build_default_registry():
        assert skill.required_inputs
        assert skill.preconditions
        assert skill.execution_backend
        assert skill.availability_status
        assert skill.success_criteria
        assert skill.error_handling

