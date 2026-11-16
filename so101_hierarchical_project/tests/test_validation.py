from so101_hierarchical.models import PlanStep, TaskPlan
from so101_hierarchical.registry import build_default_registry
from so101_hierarchical.validation import PlanValidator


def _validate(step: PlanStep):
    return PlanValidator(build_default_registry()).validate(TaskPlan("test", (step,)))


def test_valid_fixed_plan():
    report = _validate(
        PlanStep(
            "one",
            "pick_place_fixed",
            {"object": "stapler_pin_box", "target": "fixed_target", "start_mode": "fixed_start"},
        )
    )
    assert report.valid


def test_rejects_arbitrary_target():
    report = _validate(
        PlanStep(
            "one",
            "pick_place_variable",
            {"object": "box", "target": "top_of_blue_block", "start_mode": "variable_start"},
        )
    )
    assert not report.valid
    assert {issue.code for issue in report.issues} == {"unsupported_target"}


def test_rejects_wrong_start_mode():
    report = _validate(
        PlanStep(
            "one",
            "pick_place_fixed",
            {"target": "fixed_target", "start_mode": "variable_start"},
        )
    )
    assert not report.valid
    assert "unsupported_start_mode" in {issue.code for issue in report.issues}


def test_rejects_untrained_object():
    report = _validate(
        PlanStep(
            "one",
            "pick_place_variable",
            {"object": "red_block", "target": "fixed_target", "start_mode": "variable_start"},
        )
    )
    assert not report.valid
    assert "unsupported_object" in {issue.code for issue in report.issues}


def test_rejects_unknown_skill():
    report = _validate(PlanStep("one", "stack_blocks", {"target": "fixed_target"}))
    assert not report.valid
    assert report.issues[0].code == "unknown_skill"


def test_accepts_registered_pose_skill_and_rejects_other_pose():
    validator = PlanValidator(build_default_registry())
    valid = validator.validate(TaskPlan("home", (PlanStep("one", "go_home", {"pose": "home"}),)))
    invalid = validator.validate(
        TaskPlan("wrong", (PlanStep("one", "go_home", {"pose": "inspection"}),))
    )
    assert valid.valid
    assert not invalid.valid
    assert invalid.issues[0].code == "unsupported_target"


def test_rejects_missing_required_camera_when_runtime_inventory_is_supplied():
    plan = TaskPlan("detect", (PlanStep("one", "detect_blocks", {"camera": "wrist"}),))
    report = PlanValidator(build_default_registry(), available_cameras=set()).validate(plan)
    assert "required_camera_unavailable" in {issue.code for issue in report.issues}


def test_logical_camera_inventory_satisfies_act_observation_keys():
    plan = TaskPlan("fixed", (PlanStep(
        "one", "pick_place_fixed",
        {"object": "stapler_pin_box", "target": "fixed_target", "start_mode": "fixed_start"},
    ),))
    report = PlanValidator(
        build_default_registry(), available_cameras={"wrist", "external"}
    ).validate(plan)
    assert report.valid


def test_hardware_mode_requires_preconditions_and_remains_locked():
    plan = TaskPlan("home", (PlanStep("one", "go_home", {"pose": "home"}),))
    report = PlanValidator(build_default_registry(), mode="hardware").validate(plan)
    assert {issue.code for issue in report.issues} >= {"precondition_unconfirmed", "hardware_execution_locked"}

