from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .models import TaskPlan
from .registry import SkillRegistry
from .validation import PlanValidator


@dataclass(frozen=True)
class ExecutionEvent:
    event: str
    status: str
    step_id: str | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionResult:
    run_id: str
    mode: str
    status: str
    plan: dict[str, Any]
    events: tuple[ExecutionEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "status": self.status,
            "plan": self.plan,
            "events": [event.to_dict() for event in self.events],
        }


class JsonlExecutionLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, result: ExecutionResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")


class DryRunExecutor:
    """Validate and simulate a plan without opening hardware or loading weights."""

    def __init__(
        self,
        registry: SkillRegistry,
        validator: PlanValidator | None = None,
        logger: JsonlExecutionLogger | None = None,
    ) -> None:
        self.registry = registry
        self.validator = validator or PlanValidator(registry)
        self.logger = logger

    def execute(self, plan: TaskPlan) -> ExecutionResult:
        self.validator.require_valid(plan)
        run_id = datetime.now(timezone.utc).strftime("dryrun-%Y%m%dT%H%M%S.%fZ")
        events: list[ExecutionEvent] = [
            ExecutionEvent("plan_validated", "ok", None, {"step_count": len(plan.steps)})
        ]
        for step in plan.steps:
            skill = self.registry.get(step.skill)
            events.append(
                ExecutionEvent(
                    "skill_simulated",
                    "simulated",
                    step.step_id,
                    {
                        "skill": skill.name,
                        "checkpoint": skill.checkpoint_path,
                        "arguments": dict(step.arguments),
                        "required_observations": [
                            {"key": camera.key, "shape_chw": list(camera.chw)} for camera in skill.cameras
                        ],
                        "action_dimension": skill.action.dimension,
                        "action_representation": skill.action.representation,
                        "hardware_accessed": False,
                        "model_loaded": False,
                        "observations": {"simulated": True},
                        "success_criteria": list(skill.success_criteria),
                        "next_action": "continue" if step != plan.steps[-1] else "complete",
                    },
                )
            )
        result = ExecutionResult(
            run_id=run_id,
            mode="dry_run",
            status="simulated",
            plan=plan.to_dict(),
            events=tuple(events),
        )
        if self.logger:
            self.logger.append(result)
        return result


SkillBackend = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


class TaskManager:
    """Sequential skill dispatcher with observation flow and failure policy.

    Backends are injected and cannot be invented by planner output. A backend
    returns a structured observation or raises; retries occur only when the
    plan explicitly says ``retry_once``.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        backends: Mapping[str, SkillBackend],
        *,
        validator: PlanValidator | None = None,
        logger: JsonlExecutionLogger | None = None,
        mode: str = "managed",
    ) -> None:
        self.registry = registry
        self.backends = dict(backends)
        self.validator = validator or PlanValidator(registry)
        self.logger = logger
        self.mode = mode

    def execute(self, plan: TaskPlan) -> ExecutionResult:
        self.validator.require_valid(plan)
        run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S.%fZ")
        observations: dict[str, Any] = {}
        events: list[ExecutionEvent] = [
            ExecutionEvent("plan_validated", "ok", None, {"step_count": len(plan.steps)})
        ]
        status = "completed"
        for step in plan.steps:
            skill = self.registry.get(step.skill)
            backend = self.backends.get(skill.execution_backend)
            if backend is None:
                events.append(ExecutionEvent("skill_failed", "failed", step.step_id, {
                    "skill": skill.name,
                    "error": f"backend_not_bound:{skill.execution_backend}",
                    "next_action": "request_replanning",
                }))
                status = "replanning_required"
                break
            attempts = 2 if step.on_failure == "retry_once" else 1
            for attempt in range(1, attempts + 1):
                events.append(ExecutionEvent("skill_started", "running", step.step_id, {
                    "skill": skill.name, "attempt": attempt, "arguments": dict(step.arguments)
                }))
                try:
                    observation = dict(backend(step.arguments, observations))
                    observations[step.step_id] = observation
                    events.append(ExecutionEvent("skill_completed", "ok", step.step_id, {
                        "skill": skill.name,
                        "attempt": attempt,
                        "observation": observation,
                        "success_criteria": list(skill.success_criteria),
                    }))
                    break
                except Exception as exc:
                    retrying = attempt < attempts
                    events.append(ExecutionEvent("skill_failed", "retrying" if retrying else "failed", step.step_id, {
                        "skill": skill.name,
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                        "next_action": "retry_once" if retrying else "request_replanning",
                    }))
                    if not retrying:
                        status = "replanning_required"
            if status != "completed":
                break
        result = ExecutionResult(run_id, self.mode, status, plan.to_dict(), tuple(events))
        if self.logger:
            self.logger.append(result)
        return result


class LiveExecutionDisabled(RuntimeError):
    pass


def execute_live(*_: Any, **__: Any) -> None:
    raise LiveExecutionDisabled(
        "Live execution is intentionally disabled. Explicit hardware approval and a supervised "
        "execution implementation are required."
    )

