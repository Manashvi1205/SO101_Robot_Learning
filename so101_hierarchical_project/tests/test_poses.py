from pathlib import Path

import pytest

from so101_hierarchical.poses import (
    JOINT_ORDER,
    calibrated_joint_limits,
    load_pose_config,
    plan_pose_trajectory,
)


CONFIG = Path(__file__).resolve().parents[1] / "config" / "poses.yaml"


def test_pose_config_loads_exact_transcribed_values():
    config = load_pose_config(CONFIG)
    assert config.provisional is True
    assert config.calibration_file.name == "None.json"
    assert config.poses["home"].joints == {
        "shoulder_pan.pos": -5.54,
        "shoulder_lift.pos": -102.24,
        "elbow_flex.pos": 89.93,
        "wrist_flex.pos": 75.74,
        "wrist_roll.pos": -9.89,
        "gripper.pos": 2.25,
    }
    assert config.poses["inspection"].joints == {
        "shoulder_pan.pos": -4.66,
        "shoulder_lift.pos": -41.41,
        "elbow_flex.pos": 5.89,
        "wrist_flex.pos": 100.18,
        "wrist_roll.pos": -7.43,
        "gripper.pos": 2.25,
    }
    assert not config.poses["home"].verified_from_follower_measurement


def test_calibrated_limits_match_driver_representations():
    config = load_pose_config(CONFIG)
    limits = calibrated_joint_limits(config.calibration_file)
    assert limits["shoulder_pan.pos"].unit == "degrees"
    assert limits["gripper.pos"].unit == "calibrated_0_100"
    assert limits["gripper.pos"].minimum == 0.0
    assert limits["gripper.pos"].maximum == 100.0


def test_home_to_home_dry_run_is_stationary_and_valid():
    config = load_pose_config(CONFIG)
    plan = plan_pose_trajectory(
        config,
        "go_home",
        config.poses["home"].joints,
        expected_start_pose="home",
        state_source="configured_pose_simulation",
    )
    assert plan.trajectory_valid, plan.errors
    assert plan.execution_authorized is False
    assert plan.hardware_accessed is False
    assert plan.waypoints[0].joints == plan.waypoints[-1].joints


def test_home_to_inspection_is_within_calibrated_limits():
    config = load_pose_config(CONFIG)
    plan = plan_pose_trajectory(
        config,
        "go_inspection",
        config.poses["home"].joints,
        expected_start_pose="home",
    )
    assert plan.trajectory_valid, plan.errors
    assert plan.execution_authorized is False
    assert all(waypoint.joints.keys() == config.poses["home"].joints.keys() for waypoint in plan.waypoints)


def test_cubic_trajectory_respects_velocity_limits_for_valid_reverse_motion():
    config = load_pose_config(CONFIG)
    start = dict(config.poses["inspection"].joints)
    plan = plan_pose_trajectory(
        config,
        "go_home",
        start,
        expected_start_pose="inspection",
    )
    assert plan.trajectory_valid, plan.errors
    for before, after in zip(plan.waypoints, plan.waypoints[1:], strict=False):
        dt = after.time_s - before.time_s
        for joint in JOINT_ORDER:
            velocity = abs(after.joints[joint] - before.joints[joint]) / dt
            assert velocity <= config.max_velocity[joint] + 1e-9


def test_arbitrary_start_is_rejected_by_near_pose_guard():
    config = load_pose_config(CONFIG)
    arbitrary = {joint: 0.0 for joint in JOINT_ORDER}
    plan = plan_pose_trajectory(
        config,
        "go_home",
        arbitrary,
        expected_start_pose="home",
    )
    assert not plan.trajectory_valid
    assert any("differs from allowed start pose" in error for error in plan.errors)


def test_wrong_named_start_is_rejected_before_planning():
    config = load_pose_config(CONFIG)
    with pytest.raises(Exception, match="does not allow"):
        plan_pose_trajectory(
            config,
            "go_home",
            config.poses["home"].joints,
            expected_start_pose="unknown",
        )


def test_guarded_home_approach_accepts_latest_measured_near_home_state():
    config = load_pose_config(CONFIG)
    measured = {
        "shoulder_pan.pos": -3.648351648351648,
        "shoulder_lift.pos": -103.25274725274726,
        "elbow_flex.pos": 97.71428571428571,
        "wrist_flex.pos": 76.52747252747252,
        "wrist_roll.pos": -12.351648351648352,
        "gripper.pos": 1.8288700195950358,
    }
    plan = plan_pose_trajectory(
        config,
        "go_home",
        measured,
        expected_start_pose="home",
        state_source="measured_follower_state",
        allow_home_approach=True,
    )
    assert plan.trajectory_valid, plan.errors
    assert plan.duration_s == 1.0
    assert "home_approach" in plan.start_source


def test_home_approach_tolerance_still_rejects_distant_state():
    config = load_pose_config(CONFIG)
    measured = dict(config.poses["home"].joints)
    measured["elbow_flex.pos"] -= 20.0
    plan = plan_pose_trajectory(
        config,
        "go_home",
        measured,
        expected_start_pose="home",
        state_source="measured_follower_state",
        allow_home_approach=True,
    )
    assert not plan.trajectory_valid
    assert any("exceeding tolerance 10.000" in error for error in plan.errors)


def test_home_approach_is_for_live_go_home_only():
    config = load_pose_config(CONFIG)
    with pytest.raises(Exception, match="live measured go_home"):
        plan_pose_trajectory(
            config,
            "go_home",
            config.poses["home"].joints,
            expected_start_pose="home",
            state_source="configured_pose_simulation",
            allow_home_approach=True,
        )
