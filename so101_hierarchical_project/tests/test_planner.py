import json
from types import SimpleNamespace

import pytest

from so101_hierarchical.planner import OpenAICompatiblePlanner, OpenAIPlanner, RuleBasedPlanner
from so101_hierarchical.registry import build_default_registry
from so101_hierarchical.validation import PlanValidator


def test_rule_planner_selects_variable_skill():
    planner = RuleBasedPlanner(build_default_registry())
    plan = planner.plan("Pick up the box from a random position and put it in the fixed target")
    assert plan.steps[0].skill == "pick_place_variable"


def test_rule_planner_preserves_stacking_semantics_but_validator_rejects_execution():
    planner = RuleBasedPlanner(build_default_registry())
    plan = planner.plan("Stack the red block on top of the blue block")
    assert [step.skill for step in plan.steps] == [
        "go_inspection", "detect_blocks", "identify_color", "identify_color", "pick_place_variable"
    ]
    report = PlanValidator(build_default_registry()).validate(plan)
    assert not report.valid
    assert {issue.code for issue in report.issues} >= {"unsupported_object", "unsupported_target"}


def test_rule_planner_selects_guarded_home_skill():
    planner = RuleBasedPlanner(build_default_registry())
    plan = planner.plan("Go home")
    assert plan.steps[0].skill == "go_home"
    assert plan.steps[0].arguments == {"pose": "home"}


class _FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_text=json.dumps(self.payload))


def test_openai_planner_uses_schema_and_validates_output():
    responses = _FakeResponses(
        {
            "schema_version": "1.0",
            "instruction": "fixed task",
            "steps": [
                {
                    "step_id": "step_1",
                    "skill": "pick_place_fixed",
                    "arguments": {
                        "object": "stapler_pin_box",
                        "target": "fixed_target",
                        "start_mode": "fixed_start",
                    },
                    "on_failure": "abort",
                }
            ],
        }
    )
    planner = OpenAIPlanner(
        build_default_registry(), model="fake-model", client=SimpleNamespace(responses=responses)
    )
    plan = planner.plan("fixed task")
    assert plan.steps[0].skill == "pick_place_fixed"
    assert responses.kwargs["text"]["format"]["type"] == "json_schema"


def test_openai_planner_rejects_capability_violation():
    responses = _FakeResponses(
        {
            "schema_version": "1.0",
            "instruction": "new target",
            "steps": [
                {
                    "step_id": "step_1",
                    "skill": "pick_place_variable",
                    "arguments": {
                        "object": "box",
                        "target": "new_target",
                        "start_mode": "variable_start",
                    },
                    "on_failure": "abort",
                }
            ],
        }
    )
    planner = OpenAIPlanner(
        build_default_registry(), model="fake-model", client=SimpleNamespace(responses=responses)
    )
    plan = planner.plan("new target")
    report = PlanValidator(build_default_registry()).validate(plan)
    assert not report.valid
    assert "unsupported_target" in {issue.code for issue in report.issues}


def test_identification_plans_motion_detection_and_selection():
    plan = RuleBasedPlanner(build_default_registry()).plan("Identify the blue box")
    assert [step.skill for step in plan.steps] == ["go_inspection", "detect_blocks", "identify_color"]
    assert plan.steps[-1].arguments == {"color": "blue"}


def test_identify_all_plans_one_detector_call():
    plan = RuleBasedPlanner(build_default_registry()).plan("Identify all boxes")
    assert [step.skill for step in plan.steps] == ["go_inspection", "detect_blocks"]


def test_local_compatible_planner_uses_json_schema():
    message = SimpleNamespace(content=json.dumps({"schema_version": "1.0", "instruction": "home", "steps": [
        {"step_id": "step_1", "skill": "go_home", "arguments": {"pose": "home"}, "on_failure": "abort"}
    ]}), refusal=None)
    completions = _FakeResponses(None)
    completions.create = lambda **kwargs: (setattr(completions, "kwargs", kwargs) or SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    ))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    planner = OpenAICompatiblePlanner(build_default_registry(), model="local", base_url="http://local/v1", client=client)
    assert planner.plan("home").steps[0].skill == "go_home"
    assert completions.kwargs["response_format"]["type"] == "json_schema"

