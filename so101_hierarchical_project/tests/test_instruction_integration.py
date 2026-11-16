import pytest

from so101_hierarchical.execution import DryRunExecutor
from so101_hierarchical.planner import RuleBasedPlanner
from so101_hierarchical.registry import build_default_registry
from so101_hierarchical.validation import PlanValidator


@pytest.mark.parametrize(
    ("instruction", "skills"),
    [
        ("Identify the red box.", ["go_inspection", "detect_blocks", "identify_color"]),
        ("Identify the blue box.", ["go_inspection", "detect_blocks", "identify_color"]),
        ("Identify all boxes.", ["go_inspection", "detect_blocks"]),
        ("Go to inspection position.", ["go_inspection"]),
        (
            "Pick up the original training box and place it at the fixed target.",
            ["pick_place_fixed"],
        ),
    ],
)
def test_supported_instruction_plans_validate_and_dry_run(instruction, skills):
    registry = build_default_registry()
    plan = RuleBasedPlanner(registry).plan(instruction)
    assert [step.skill for step in plan.steps] == skills
    assert PlanValidator(registry).validate(plan).valid
    assert DryRunExecutor(registry).execute(plan).status == "simulated"


def test_stacking_plan_is_semantic_but_safely_non_executable():
    registry = build_default_registry()
    plan = RuleBasedPlanner(registry).plan("Stack the red box on the blue box.")
    assert plan.steps[-1].arguments["target"] == "on_blue_box"
    report = PlanValidator(registry).validate(plan)
    assert not report.valid
    assert {issue.code for issue in report.issues} >= {"unsupported_object", "unsupported_target"}
