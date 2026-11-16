from __future__ import annotations

import glob
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .poses import GRIPPER, JOINT_ORDER, PoseConfig, TrajectoryPlan, calibrated_joint_limits, plan_pose_trajectory


class RobotSafetyError(RuntimeError):
    pass


def discover_serial_ports() -> list[str]:
    return sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))


def validate_explicit_port(port: str, discovered: list[str] | None = None) -> None:
    ports = discovered if discovered is not None else discover_serial_ports()
    if port not in ports:
        raise RobotSafetyError(f"Requested robot port {port!r} not present; discovered {ports}")


def connect_follower(port: str, calibration_dir: str | Path, robot_id: str | None = None):
    """Connect to only the follower, without calibration or cameras."""
    validate_explicit_port(port)
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    config = SOFollowerRobotConfig(
        port=port,
        # Match the user's recent ACM1 teleoperation/recording commands, which
        # omit --robot.id and therefore load calibration/robots/so_follower/None.json.
        id=robot_id,
        calibration_dir=Path(calibration_dir),
        cameras={},
        use_degrees=True,
        max_relative_target=None,
    )
    robot = SOFollower(config)
    robot.connect(calibrate=False)
    return robot


def joint_state_from_observation(observation: Mapping[str, Any]) -> dict[str, float]:
    missing = set(JOINT_ORDER) - set(observation)
    if missing:
        raise RobotSafetyError(f"Follower observation missing joints: {sorted(missing)}")
    return {joint: float(observation[joint]) for joint in JOINT_ORDER}


def plan_from_measured_state(
    robot: Any,
    pose_config: PoseConfig,
    transition: str,
    expected_start_pose: str,
    *,
    allow_home_approach: bool = False,
) -> tuple[dict[str, float], TrajectoryPlan]:
    measured = joint_state_from_observation(robot.get_observation())
    plan = plan_pose_trajectory(
        pose_config,
        transition,
        measured,
        expected_start_pose=expected_start_pose,
        state_source="measured_follower_state",
        allow_home_approach=allow_home_approach,
    )
    return measured, replace(plan, hardware_accessed=True)


def require_motion_approval(response: str, required_phrase: str = "MOVE") -> None:
    if response != required_phrase:
        raise RobotSafetyError(
            f"Movement not approved; exact confirmation {required_phrase!r} was not entered"
        )


@dataclass(frozen=True)
class MotionResult:
    completed: bool
    reason: str
    final_state: Mapping[str, float]
    final_error: Mapping[str, float]


@dataclass(frozen=True)
class PoseValidationResult:
    valid: bool
    required_pose: str
    measured_joints: Mapping[str, float]
    errors: tuple[str, ...]
    joint_errors: Mapping[str, float]
    joint_units: Mapping[str, str]
    tolerances: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "required_pose": self.required_pose,
            "measured_joints": dict(self.measured_joints),
            "joint_errors": dict(self.joint_errors),
            "joint_units": dict(self.joint_units),
            "tolerances": dict(self.tolerances),
            "errors": list(self.errors),
        }


def validate_measured_named_pose(
    robot: Any,
    pose_config: PoseConfig,
    pose_name: str,
) -> PoseValidationResult:
    if pose_name not in pose_config.poses:
        raise RobotSafetyError(f"Unknown required capture pose: {pose_name!r}")
    measured = joint_state_from_observation(robot.get_observation())
    target = pose_config.poses[pose_name].joints
    limits = calibrated_joint_limits(pose_config.calibration_file)
    errors: list[str] = []
    joint_errors: dict[str, float] = {}
    tolerances: dict[str, float] = {}
    units: dict[str, str] = {}
    for joint in JOINT_ORDER:
        tolerance = (
            pose_config.gripper_start_tolerance if joint == GRIPPER else pose_config.arm_start_tolerance
        )
        delta = measured[joint] - target[joint]
        joint_errors[joint] = delta
        tolerances[joint] = tolerance
        units[joint] = limits[joint].unit
        if not limits[joint].minimum <= measured[joint] <= limits[joint].maximum:
            errors.append(
                f"measured {joint}={measured[joint]:.4f} outside calibrated "
                f"[{limits[joint].minimum:.4f}, {limits[joint].maximum:.4f}]"
            )
        if abs(delta) > tolerance:
            errors.append(
                f"measured {joint} differs from {pose_name!r} by {abs(delta):.3f}, "
                f"exceeding tolerance {tolerance:.3f}"
            )
    return PoseValidationResult(
        valid=not errors,
        required_pose=pose_name,
        measured_joints=measured,
        errors=tuple(errors),
        joint_errors=joint_errors,
        joint_units=units,
        tolerances=tolerances,
    )


def execute_validated_trajectory(
    robot: Any,
    plan: TrajectoryPlan,
    pose_config: PoseConfig,
    stop_event: threading.Event,
    *,
    max_tracking_error_arm: float = 15.0,
    max_tracking_error_gripper: float = 20.0,
    allow_provisional_target: bool = False,
) -> MotionResult:
    if not plan.trajectory_valid:
        raise RobotSafetyError(f"Trajectory is invalid: {list(plan.errors)}")
    provisional_authorized = (
        allow_provisional_target
        and plan.start_source.startswith("measured_follower_state_")
    )
    if not plan.target_verified and not provisional_authorized:
        raise RobotSafetyError("Target pose is provisional or execution is not authorized")
    if not plan.execution_authorized and not provisional_authorized:
        raise RobotSafetyError("Execution is not authorized")
    limits = calibrated_joint_limits(pose_config.calibration_file)
    started = time.monotonic()
    last_state: dict[str, float] = {}
    for waypoint in plan.waypoints:
        if stop_event.is_set():
            return MotionResult(False, "emergency stop requested", last_state, {})
        try:
            measured = joint_state_from_observation(robot.get_observation())
        except Exception as exc:
            return MotionResult(False, f"joint read failed: {exc}", last_state, {})
        last_state = measured
        for joint, value in measured.items():
            limit = limits[joint]
            if not (limit.minimum <= value <= limit.maximum):
                return MotionResult(False, f"measured {joint} outside calibrated limits", measured, {})
            tolerance = max_tracking_error_gripper if joint == GRIPPER else max_tracking_error_arm
            if abs(value - waypoint.joints[joint]) > tolerance:
                return MotionResult(False, f"excessive tracking error on {joint}", measured, {})
        try:
            robot.send_action(dict(waypoint.joints))
        except Exception as exc:
            return MotionResult(False, f"command failed: {exc}", measured, {})
        deadline = started + waypoint.time_s
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    final_state = joint_state_from_observation(robot.get_observation())
    target = plan.waypoints[-1].joints
    final_error = {joint: final_state[joint] - target[joint] for joint in JOINT_ORDER}
    return MotionResult(True, "completed", final_state, final_error)

