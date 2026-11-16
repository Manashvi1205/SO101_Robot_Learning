import json

import pytest

from so101_hierarchical.execution import (
    DryRunExecutor, JsonlExecutionLogger, LiveExecutionDisabled, TaskManager, execute_live,
)
from so101_hierarchical.models import PlanStep, TaskPlan
from so101_hierarchical.planner import RuleBasedPlanner
from so101_hierarchical.registry import build_default_registry


def test_dry_run_is_explicitly_hardware_free(tmp_path):
    registry = build_default_registry()
    plan = RuleBasedPlanner(registry).plan(
        "Pick up the box from a variable position and place it at the fixed target"
    )
    log = tmp_path / "runs.jsonl"
    result = DryRunExecutor(registry, logger=JsonlExecutionLogger(log)).execute(plan)
    assert result.status == "simulated"
    event = result.events[-1]
    assert event.details["hardware_accessed"] is False
    assert event.details["model_loaded"] is False
    saved = json.loads(log.read_text().strip())
    assert saved["run_id"] == result.run_id


def test_live_execution_is_disabled():
    with pytest.raises(LiveExecutionDisabled):
        execute_live()


def test_task_manager_flows_observations_sequentially():
    registry = build_default_registry()
    seen = []

    def detect(arguments, observations):
        seen.append((dict(arguments), dict(observations)))
        return {"detected_colors": ["red"]}

    def select(arguments, observations):
        assert observations["one"]["detected_colors"] == ["red"]
        return {"success": True, "color": arguments["color"]}

    manager = TaskManager(registry, {"color_detector": detect, "color_selector": select})
    plan = TaskPlan("red", (
        PlanStep("one", "detect_blocks", {"camera": "wrist"}),
        PlanStep("two", "identify_color", {"color": "red"}),
    ))
    result = manager.execute(plan)
    assert result.status == "completed"
    assert seen


def test_task_manager_stops_and_requests_replanning_on_failure():
    registry = build_default_registry()

    def fail(arguments, observations):
        raise RuntimeError("no frame")

    manager = TaskManager(registry, {"color_detector": fail})
    plan = TaskPlan("detect", (PlanStep("one", "detect_blocks", {"camera": "wrist"}),))
    result = manager.execute(plan)
    assert result.status == "replanning_required"
    assert result.events[-1].details["next_action"] == "request_replanning"

