from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .models import ActionSpec, CameraRequirement, SkillSpec


FIXED_CHECKPOINT = Path(
    "/home/manashvi/so101_outputs/fixed_21episodes_2cam_act/checkpoints/050000/pretrained_model"
)
VARIABLE_CHECKPOINT = Path(
    "/home/manashvi/so101_outputs/variable_box_40episodes_act/checkpoints/050000/pretrained_model"
)
DATASET_ROOT = Path("/home/manashvi/.cache/huggingface/lerobot")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSE_CONFIG = PROJECT_ROOT / "config" / "poses.yaml"
COLOR_CONFIG = PROJECT_ROOT / "config" / "color_detection.yaml"

JOINT_ACTION = ActionSpec(
    names=(
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )
)


class SkillRegistry:
    def __init__(self, skills: list[SkillSpec] | tuple[SkillSpec, ...] = ()) -> None:
        self._skills: dict[str, SkillSpec] = {}
        for skill in skills:
            self.register(skill)

    def register(self, skill: SkillSpec) -> None:
        if skill.name in self._skills:
            raise ValueError(f"Skill already registered: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillSpec:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"Unknown skill: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._skills)

    def to_dict(self) -> dict[str, dict]:
        return {name: self._skills[name].to_dict() for name in self.names()}

    def __iter__(self) -> Iterator[SkillSpec]:
        for name in self.names():
            yield self._skills[name]

    def validate_assets(self) -> dict[str, list[str]]:
        required = (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        )
        problems: dict[str, list[str]] = {}
        for skill in self:
            if skill.implementation_type != "act_policy":
                if skill.config_path and not Path(skill.config_path).is_file():
                    problems[skill.name] = ["configuration file"]
                continue
            checkpoint = Path(skill.checkpoint_path)
            missing = [name for name in required if not (checkpoint / name).is_file()]
            if not Path(skill.dataset_path).is_dir():
                missing.append("dataset directory")
            if missing:
                problems[skill.name] = missing
        return problems


def build_default_registry() -> SkillRegistry:
    fixed = SkillSpec(
        name="pick_place_fixed",
        description=(
            "Pick up the trained small stapler-pin box from its fixed start pose and "
            "place it at the trained fixed destination."
        ),
        checkpoint_path=str(FIXED_CHECKPOINT),
        dataset_path=str(DATASET_ROOT / "so101_fixed_21episodes_2cam"),
        training_steps=50_000,
        cameras=(
            CameraRequirement("observation.images.wrist", 3, 480, 640),
            CameraRequirement("observation.images.front", 3, 720, 1280),
        ),
        action=JOINT_ACTION,
        start_mode="fixed_start",
        supported_objects=("stapler_pin_box",),
        supported_targets=("fixed_target",),
        default_arguments={"object": "stapler_pin_box", "target": "fixed_target"},
        required_inputs=("observation.state", "observation.images.wrist", "observation.images.front"),
        preconditions=(
            "trained stapler-pin box is at the demonstrated fixed start pose",
            "workspace and fixed destination match the training setup",
            "robot starts in the demonstrated rollout state",
        ),
        execution_backend="act_fixed_50k",
        availability_status="dry_run_ready_hardware_locked",
        success_criteria=("policy rollout completes", "object is at the trained fixed target"),
        error_handling=("stop immediately", "do not retry without operator inspection"),
    )
    variable = SkillSpec(
        name="pick_place_variable",
        description=(
            "Pick up the trained box from a variable position in the demonstrated "
            "workspace and place it at the trained fixed destination."
        ),
        checkpoint_path=str(VARIABLE_CHECKPOINT),
        dataset_path=str(DATASET_ROOT / "so101_variable_box_40episodes"),
        training_steps=50_000,
        cameras=(
            CameraRequirement("observation.images.wrist", 3, 720, 1280),
            CameraRequirement("observation.images.phone", 3, 720, 1280),
        ),
        action=JOINT_ACTION,
        start_mode="variable_start",
        supported_objects=("box",),
        supported_targets=("fixed_target",),
        default_arguments={"object": "box", "target": "fixed_target"},
        camera_aliases={"observation.images.front": "observation.images.phone"},
        required_inputs=("observation.state", "observation.images.wrist", "observation.images.phone"),
        preconditions=(
            "trained physical box is inside the demonstrated variable-start workspace",
            "fixed destination and camera geometry match training",
            "robot starts in the demonstrated rollout state",
        ),
        execution_backend="act_variable_50k",
        availability_status="dry_run_ready_hardware_locked",
        success_criteria=("policy rollout completes", "trained box is at the trained fixed target"),
        error_handling=("stop immediately", "request operator inspection or replanning"),
    )
    go_home = SkillSpec(
        name="go_home",
        description="Move through a validated, velocity-limited joint trajectory to the provisional home pose.",
        checkpoint_path="",
        dataset_path="",
        training_steps=0,
        cameras=(),
        action=JOINT_ACTION,
        start_mode="guarded_measured_joint_state",
        supported_objects=(),
        supported_targets=("home",),
        default_arguments={"pose": "home"},
        implementation_type="joint_trajectory",
        config_path=str(POSE_CONFIG),
        required_inputs=("measured_joint_state",),
        preconditions=("measured state passes named-pose proximity and joint-limit guards",),
        execution_backend="guarded_pose_controller",
        availability_status="dry_run_ready_hardware_locked",
        success_criteria=("measured joints are within the configured home tolerance",),
        error_handling=("stop trajectory", "do not retry automatically"),
    )
    go_inspection = SkillSpec(
        name="go_inspection",
        description=(
            "Move through a validated, velocity-limited joint trajectory to the provisional inspection pose."
        ),
        checkpoint_path="",
        dataset_path="",
        training_steps=0,
        cameras=(),
        action=JOINT_ACTION,
        start_mode="guarded_measured_joint_state",
        supported_objects=(),
        supported_targets=("inspection",),
        default_arguments={"pose": "inspection"},
        implementation_type="joint_trajectory",
        config_path=str(POSE_CONFIG),
        required_inputs=("measured_joint_state",),
        preconditions=("measured state passes named-pose proximity and joint-limit guards",),
        execution_backend="guarded_pose_controller",
        availability_status="dry_run_ready_hardware_locked",
        success_criteria=("measured joints are within the configured inspection tolerance",),
        error_handling=("stop trajectory", "do not retry automatically"),
    )
    detect_blocks = SkillSpec(
        name="detect_blocks",
        description="Detect red, green, and blue boxes and report image-coordinate geometry.",
        checkpoint_path="", dataset_path="", training_steps=0,
        cameras=(CameraRequirement("wrist", 3, 720, 1280),), action=JOINT_ACTION,
        start_mode="inspection_view", supported_objects=("red_box", "green_box", "blue_box", "all_boxes"),
        supported_targets=(), default_arguments={"camera": "wrist"}, implementation_type="perception",
        config_path=str(COLOR_CONFIG), required_inputs=("camera_frame", "camera_name"),
        preconditions=("camera pose and image characteristics match detector configuration",),
        execution_backend="color_detector", availability_status="available",
        success_criteria=("structured detections are returned without converting pixels to robot coordinates",),
        error_handling=("report missing colors", "fail on unreadable or empty frame"),
    )
    identify_color = SkillSpec(
        name="identify_color",
        description="Select one requested color from structured block detections.",
        checkpoint_path="", dataset_path="", training_steps=0, cameras=(), action=JOINT_ACTION,
        start_mode="from_detection_result", supported_objects=("red_box", "green_box", "blue_box"),
        supported_targets=(), implementation_type="perception_query",
        required_inputs=("detections", "color"), preconditions=("detect_blocks has produced detections",),
        execution_backend="color_selector", availability_status="available",
        success_criteria=("exactly one best matching color detection is returned",),
        error_handling=("return object_not_found without inventing a position",),
    )
    verify_placement = SkillSpec(
        name="verify_placement",
        description=("Visually verify a colored box against an explicitly supplied image-space target region; "
                     "it does not infer robot-frame geometry."),
        checkpoint_path="", dataset_path="", training_steps=0,
        cameras=(CameraRequirement("wrist", 3, 720, 1280),), action=JOINT_ACTION,
        start_mode="post_action_view", supported_objects=("red_box", "green_box", "blue_box"),
        supported_targets=("configured_image_region",), implementation_type="verification",
        config_path=str(COLOR_CONFIG), required_inputs=("post_action_frame", "color", "target_region"),
        preconditions=("an image-space target region is explicitly configured",),
        execution_backend="visual_placement_verifier", availability_status="conditional",
        success_criteria=("selected box center lies inside the configured image-space region",),
        error_handling=("return verification_failed", "request replanning or operator review"),
    )
    return SkillRegistry((fixed, variable, go_home, go_inspection, detect_blocks, identify_color, verify_placement))

