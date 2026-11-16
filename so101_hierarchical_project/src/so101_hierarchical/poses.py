from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


JOINT_ORDER = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
ARM_JOINTS = JOINT_ORDER[:-1]
GRIPPER = JOINT_ORDER[-1]
FEETECH_RESOLUTION = 4096


class PoseConfigError(ValueError):
    pass


@dataclass(frozen=True)
class JointLimit:
    minimum: float
    maximum: float
    unit: str

    def to_dict(self) -> dict[str, Any]:
        return {"minimum": self.minimum, "maximum": self.maximum, "unit": self.unit}


@dataclass(frozen=True)
class NamedPose:
    name: str
    joints: Mapping[str, float]
    verified_from_follower_measurement: bool


@dataclass(frozen=True)
class TransitionSpec:
    name: str
    target: str
    allowed_start_poses: tuple[str, ...]


@dataclass(frozen=True)
class PoseConfig:
    path: Path
    robot_type: str
    calibration_file: Path
    provisional: bool
    rate_hz: float
    minimum_duration_s: float
    screenshot_rounding_tolerance: float
    arm_start_tolerance: float
    gripper_start_tolerance: float
    max_velocity: Mapping[str, float]
    poses: Mapping[str, NamedPose]
    transitions: Mapping[str, TransitionSpec]


@dataclass(frozen=True)
class TrajectoryWaypoint:
    index: int
    time_s: float
    joints: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "time_s": self.time_s, "joints": dict(self.joints)}


@dataclass(frozen=True)
class TrajectoryPlan:
    transition: str
    start_source: str
    target_pose: str
    duration_s: float
    rate_hz: float
    max_velocity: Mapping[str, float]
    limits: Mapping[str, JointLimit]
    waypoints: tuple[TrajectoryWaypoint, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    target_verified: bool
    hardware_accessed: bool = False
    execution_authorized: bool = False

    @property
    def trajectory_valid(self) -> bool:
        return not self.errors

    def to_dict(self, include_waypoints: bool = True) -> dict[str, Any]:
        joint_changes = {
            joint: self.waypoints[-1].joints[joint] - self.waypoints[0].joints[joint]
            for joint in JOINT_ORDER
        }
        result = {
            "transition": self.transition,
            "start_source": self.start_source,
            "target_pose": self.target_pose,
            "duration_s": self.duration_s,
            "rate_hz": self.rate_hz,
            "waypoint_count": len(self.waypoints),
            "joint_changes": joint_changes,
            "max_velocity": dict(self.max_velocity),
            "limits": {key: value.to_dict() for key, value in self.limits.items()},
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "target_verified": self.target_verified,
            "trajectory_valid": self.trajectory_valid,
            "hardware_accessed": self.hardware_accessed,
            "execution_authorized": self.execution_authorized,
        }
        if include_waypoints:
            result["waypoints"] = [waypoint.to_dict() for waypoint in self.waypoints]
        return result


def _require_keys(mapping: Mapping[str, Any], keys: set[str], context: str) -> None:
    missing = keys - set(mapping)
    if missing:
        raise PoseConfigError(f"{context} missing keys: {sorted(missing)}")


def _joint_values(raw: Mapping[str, Any], context: str) -> dict[str, float]:
    _require_keys(raw, set(JOINT_ORDER), context)
    unknown = set(raw) - set(JOINT_ORDER)
    if unknown:
        raise PoseConfigError(f"{context} has unknown joints: {sorted(unknown)}")
    result = {name: float(raw[name]) for name in JOINT_ORDER}
    if not all(math.isfinite(value) for value in result.values()):
        raise PoseConfigError(f"{context} contains a non-finite value")
    return result


def load_pose_config(path: str | Path) -> PoseConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise PoseConfigError("Pose configuration must be a YAML object")
    _require_keys(raw, {"robot_type", "calibration_file", "control", "poses", "transitions"}, "config")
    control = raw["control"]
    poses = {
        name: NamedPose(
            name=name,
            joints=_joint_values(value["joints"], f"pose {name!r}"),
            verified_from_follower_measurement=bool(value.get("verified_from_follower_measurement", False)),
        )
        for name, value in raw["poses"].items()
    }
    transitions = {
        name: TransitionSpec(
            name=name,
            target=str(value["target"]),
            allowed_start_poses=tuple(str(item) for item in value["allowed_start_poses"]),
        )
        for name, value in raw["transitions"].items()
    }
    for transition in transitions.values():
        if transition.target not in poses:
            raise PoseConfigError(f"Transition {transition.name!r} has unknown target {transition.target!r}")
        unknown_starts = set(transition.allowed_start_poses) - set(poses)
        if unknown_starts:
            raise PoseConfigError(
                f"Transition {transition.name!r} has unknown start poses: {sorted(unknown_starts)}"
            )
    velocity = _joint_values(control["max_velocity"], "control.max_velocity")
    if any(value <= 0 for value in velocity.values()):
        raise PoseConfigError("All velocity limits must be positive")
    return PoseConfig(
        path=config_path,
        robot_type=str(raw["robot_type"]),
        calibration_file=Path(raw["calibration_file"]),
        provisional=bool(raw.get("provisional", True)),
        rate_hz=float(control["rate_hz"]),
        minimum_duration_s=float(control["minimum_duration_s"]),
        screenshot_rounding_tolerance=float(control.get("screenshot_rounding_tolerance", 0.0)),
        arm_start_tolerance=float(control["start_tolerance"]["arm_degrees"]),
        gripper_start_tolerance=float(control["start_tolerance"]["gripper_units"]),
        max_velocity=velocity,
        poses=poses,
        transitions=transitions,
    )


def calibrated_joint_limits(calibration_file: str | Path) -> dict[str, JointLimit]:
    with Path(calibration_file).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    _require_keys(raw, {name.removesuffix(".pos") for name in JOINT_ORDER}, "calibration")
    limits: dict[str, JointLimit] = {}
    for joint in ARM_JOINTS:
        motor = joint.removesuffix(".pos")
        minimum_raw = float(raw[motor]["range_min"])
        maximum_raw = float(raw[motor]["range_max"])
        midpoint = (minimum_raw + maximum_raw) / 2.0
        limits[joint] = JointLimit(
            minimum=(minimum_raw - midpoint) * 360.0 / (FEETECH_RESOLUTION - 1),
            maximum=(maximum_raw - midpoint) * 360.0 / (FEETECH_RESOLUTION - 1),
            unit="degrees",
        )
    limits[GRIPPER] = JointLimit(0.0, 100.0, "calibrated_0_100")
    return limits


def _limit_errors(
    state: Mapping[str, float],
    limits: Mapping[str, JointLimit],
    *,
    label: str,
    tolerance: float = 0.0,
) -> list[str]:
    errors = []
    for joint in JOINT_ORDER:
        value = state[joint]
        limit = limits[joint]
        epsilon = tolerance if joint in ARM_JOINTS else 0.0
        if value < limit.minimum - epsilon or value > limit.maximum + epsilon:
            errors.append(
                f"{label} {joint}={value:.4f} is outside calibrated "
                f"[{limit.minimum:.4f}, {limit.maximum:.4f}] {limit.unit}"
            )
    return errors


def _start_guard_errors(
    current: Mapping[str, float],
    expected: NamedPose,
    config: PoseConfig,
    *,
    tolerance_scale: float = 1.0,
) -> list[str]:
    errors = []
    for joint in JOINT_ORDER:
        base_tolerance = (
            config.gripper_start_tolerance if joint == GRIPPER else config.arm_start_tolerance
        )
        tolerance = base_tolerance * tolerance_scale
        delta = abs(current[joint] - expected.joints[joint])
        if delta > tolerance:
            errors.append(
                f"current {joint} differs from allowed start pose {expected.name!r} "
                f"by {delta:.3f}, exceeding tolerance {tolerance:.3f}"
            )
    return errors


def plan_pose_trajectory(
    config: PoseConfig,
    transition_name: str,
    current_state: Mapping[str, Any],
    *,
    expected_start_pose: str,
    state_source: str = "measured_follower_state",
    allow_home_approach: bool = False,
) -> TrajectoryPlan:
    if transition_name not in config.transitions:
        raise PoseConfigError(f"Unknown transition: {transition_name}")
    transition = config.transitions[transition_name]
    if allow_home_approach and (
        state_source != "measured_follower_state"
        or transition_name != "go_home"
        or expected_start_pose != "home"
    ):
        raise PoseConfigError(
            "home-approach tolerance is allowed only for live measured go_home motion"
        )
    if expected_start_pose not in transition.allowed_start_poses:
        raise PoseConfigError(
            f"{transition_name} does not allow expected start pose {expected_start_pose!r}; "
            f"allowed: {list(transition.allowed_start_poses)}"
        )
    current = _joint_values(current_state, "current_state")
    expected = config.poses[expected_start_pose]
    target = config.poses[transition.target]
    limits = calibrated_joint_limits(config.calibration_file)
    errors = _limit_errors(
        current,
        limits,
        label="start",
        tolerance=config.screenshot_rounding_tolerance,
    )
    errors.extend(
        _limit_errors(
            target.joints,
            limits,
            label=f"target pose {target.name!r}",
            tolerance=config.screenshot_rounding_tolerance,
        )
    )
    errors.extend(
        _start_guard_errors(
            current,
            expected,
            config,
            tolerance_scale=2.0 if allow_home_approach else 1.0,
        )
    )

    delta = {joint: target.joints[joint] - current[joint] for joint in JOINT_ORDER}
    # Cubic smoothstep has max ds/dt = 1.5 / duration. Choose duration so
    # every joint's peak velocity is at or below its configured limit.
    duration = max(
        config.minimum_duration_s,
        max(1.5 * abs(delta[joint]) / config.max_velocity[joint] for joint in JOINT_ORDER),
    )
    intervals = max(1, math.ceil(duration * config.rate_hz))
    duration = intervals / config.rate_hz
    waypoints: list[TrajectoryWaypoint] = []
    for index in range(intervals + 1):
        phase = index / intervals
        blend = 3.0 * phase**2 - 2.0 * phase**3
        joints = {joint: current[joint] + delta[joint] * blend for joint in JOINT_ORDER}
        waypoints.append(TrajectoryWaypoint(index, index / config.rate_hz, joints))

    waypoint_limit_joints: set[str] = set()
    for waypoint in waypoints:
        for joint in JOINT_ORDER:
            if joint in waypoint_limit_joints:
                continue
            value = waypoint.joints[joint]
            limit = limits[joint]
            epsilon = config.screenshot_rounding_tolerance if joint in ARM_JOINTS else 0.0
            if value < limit.minimum - epsilon or value > limit.maximum + epsilon:
                errors.append(
                    f"waypoint {waypoint.index} {joint}={value:.4f} is outside calibrated "
                    f"[{limit.minimum:.4f}, {limit.maximum:.4f}] {limit.unit}"
                )
                waypoint_limit_joints.add(joint)
    for previous, current_waypoint in zip(waypoints, waypoints[1:], strict=False):
        dt = current_waypoint.time_s - previous.time_s
        for joint in JOINT_ORDER:
            velocity = abs(current_waypoint.joints[joint] - previous.joints[joint]) / dt
            if velocity > config.max_velocity[joint] + 1e-9:
                errors.append(
                    f"waypoint {current_waypoint.index} {joint} velocity {velocity:.4f} exceeds "
                    f"limit {config.max_velocity[joint]:.4f}"
                )

    warnings = []
    if allow_home_approach:
        warnings.append(
            "using guarded HOME approach tolerance (10 degrees arm, 10 gripper units); "
            "the normal 5-unit guard is rechecked before continuing"
        )
    if config.provisional or not target.verified_from_follower_measurement:
        warnings.append(
            f"target pose {target.name!r} is provisional and must be verified from measured follower state"
        )
    # Preserve order while removing repeated waypoint limit errors.
    unique_errors = tuple(dict.fromkeys(errors))
    return TrajectoryPlan(
        transition=transition_name,
        start_source=(
            f"{state_source}_home_approach:{expected_start_pose}"
            if allow_home_approach
            else f"{state_source}_near:{expected_start_pose}"
        ),
        target_pose=target.name,
        duration_s=duration,
        rate_hz=config.rate_hz,
        max_velocity=config.max_velocity,
        limits=limits,
        waypoints=tuple(waypoints),
        errors=unique_errors,
        warnings=tuple(warnings),
        target_verified=target.verified_from_follower_measurement,
        execution_authorized=(
            not unique_errors
            and target.verified_from_follower_measurement
            and state_source == "measured_follower_state"
        ),
    )

