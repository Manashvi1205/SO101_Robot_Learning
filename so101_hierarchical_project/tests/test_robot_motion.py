from dataclasses import replace
from pathlib import Path
import threading

import pytest

from so101_hierarchical.poses import load_pose_config, plan_pose_trajectory
from so101_hierarchical.robot_motion import (
    RobotSafetyError,
    execute_validated_trajectory,
    plan_from_measured_state,
    require_motion_approval,
    validate_measured_named_pose,
    validate_explicit_port,
)


CONFIG = Path(__file__).resolve().parents[1] / "config" / "poses.yaml"


def test_exact_move_confirmation_is_required():
    require_motion_approval("MOVE")
    for value in ("move", " MOVE", "yes", ""):
        with pytest.raises(RobotSafetyError):
            require_motion_approval(value)


def test_provisional_move_requires_full_exact_confirmation():
    phrase = "MOVE TO PROVISIONAL INSPECTION"
    require_motion_approval(phrase, phrase)
    for value in ("MOVE", phrase.lower(), f" {phrase}", ""):
        with pytest.raises(RobotSafetyError):
            require_motion_approval(value, phrase)


def test_explicit_follower_port_must_exist():
    validate_explicit_port("/dev/ttyACM0", ["/dev/ttyACM0", "/dev/ttyACM1"])
    with pytest.raises(RobotSafetyError):
        validate_explicit_port("/dev/ttyACM9", ["/dev/ttyACM0", "/dev/ttyACM1"])


class FakeRobot:
    def __init__(self, state):
        self.state = dict(state)
        self.commands = []

    def get_observation(self):
        return dict(self.state)

    def send_action(self, action):
        self.commands.append(dict(action))
        self.state = dict(action)


def test_capture_pose_validation_uses_measured_inspection_state_and_units():
    config = load_pose_config(CONFIG)
    robot = FakeRobot(config.poses["inspection"].joints)
    result = validate_measured_named_pose(robot, config, "inspection")
    assert result.valid
    assert result.joint_units["shoulder_pan.pos"] == "degrees"
    assert result.joint_units["gripper.pos"] == "calibrated_0_100"


def test_capture_pose_validation_rejects_out_of_tolerance_state():
    config = load_pose_config(CONFIG)
    state = dict(config.poses["inspection"].joints)
    state["elbow_flex.pos"] += 6.0
    result = validate_measured_named_pose(FakeRobot(state), config, "inspection")
    assert not result.valid
    assert any("elbow_flex.pos" in error for error in result.errors)


def test_measured_plan_records_hardware_access():
    config = load_pose_config(CONFIG)
    robot = FakeRobot(config.poses["home"].joints)
    measured, plan = plan_from_measured_state(robot, config, "go_inspection", "home")
    assert measured == config.poses["home"].joints
    assert plan.hardware_accessed is True


def test_unverified_pose_cannot_execute_even_with_valid_trajectory():
    config = load_pose_config(CONFIG)
    plan = plan_pose_trajectory(
        config,
        "go_home",
        config.poses["home"].joints,
        expected_start_pose="home",
        state_source="measured_follower_state",
    )
    robot = FakeRobot(config.poses["home"].joints)
    with pytest.raises(RobotSafetyError, match="provisional"):
        execute_validated_trajectory(robot, plan, config, threading.Event())
    assert robot.commands == []


def test_explicit_provisional_target_can_execute_only_from_measured_start(monkeypatch):
    config = load_pose_config(CONFIG)
    plan = plan_pose_trajectory(
        config,
        "go_inspection",
        config.poses["home"].joints,
        expected_start_pose="home",
        state_source="measured_follower_state",
    )
    robot = FakeRobot(config.poses["home"].joints)
    monkeypatch.setattr("so101_hierarchical.robot_motion.time.sleep", lambda _seconds: None)
    result = execute_validated_trajectory(
        robot,
        plan,
        config,
        threading.Event(),
        allow_provisional_target=True,
    )
    assert result.completed
    assert robot.commands[-1] == pytest.approx(config.poses["inspection"].joints)


def test_provisional_target_override_rejects_simulated_start():
    config = load_pose_config(CONFIG)
    plan = plan_pose_trajectory(
        config,
        "go_inspection",
        config.poses["home"].joints,
        expected_start_pose="home",
        state_source="configured_pose_simulation",
    )
    robot = FakeRobot(config.poses["home"].joints)
    with pytest.raises(RobotSafetyError, match="provisional"):
        execute_validated_trajectory(
            robot,
            plan,
            config,
            threading.Event(),
            allow_provisional_target=True,
        )
    assert robot.commands == []


def test_guarded_measured_state_can_execute_home_then_inspection(monkeypatch):
    config = load_pose_config(CONFIG)
    measured = {
        "shoulder_pan.pos": -3.648351648351648,
        "shoulder_lift.pos": -103.25274725274726,
        "elbow_flex.pos": 97.71428571428571,
        "wrist_flex.pos": 76.52747252747252,
        "wrist_roll.pos": -12.351648351648352,
        "gripper.pos": 1.8288700195950358,
    }
    robot = FakeRobot(measured)
    monkeypatch.setattr("so101_hierarchical.robot_motion.time.sleep", lambda _seconds: None)

    _, home_plan = plan_from_measured_state(
        robot,
        config,
        "go_home",
        "home",
        allow_home_approach=True,
    )
    home_result = execute_validated_trajectory(
        robot,
        home_plan,
        config,
        threading.Event(),
        allow_provisional_target=True,
    )
    assert home_result.completed
    assert robot.state == pytest.approx(config.poses["home"].joints)

    _, inspection_plan = plan_from_measured_state(robot, config, "go_inspection", "home")
    assert inspection_plan.trajectory_valid
    inspection_result = execute_validated_trajectory(
        robot,
        inspection_plan,
        config,
        threading.Event(),
        allow_provisional_target=True,
    )
    assert inspection_result.completed
    assert robot.state == pytest.approx(config.poses["inspection"].joints)


def test_emergency_stop_prevents_first_command_on_authorized_test_plan():
    config = load_pose_config(CONFIG)
    plan = plan_pose_trajectory(
        config,
        "go_home",
        config.poses["home"].joints,
        expected_start_pose="home",
        state_source="measured_follower_state",
    )
    plan = replace(plan, target_verified=True, execution_authorized=True)
    robot = FakeRobot(config.poses["home"].joints)
    stop = threading.Event()
    stop.set()
    result = execute_validated_trajectory(robot, plan, config, stop)
    assert not result.completed
    assert robot.commands == []

