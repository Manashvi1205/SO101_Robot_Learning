from __future__ import annotations

import json
import re
from typing import Any, Protocol

from .models import PlanStep, TaskPlan
from .registry import SkillRegistry
from .validation import PlanValidator


class Planner(Protocol):
    def plan(self, instruction: str) -> TaskPlan: ...


def build_system_prompt(registry: SkillRegistry) -> str:
    capabilities = json.dumps(registry.to_dict(), indent=2)
    return f"""You are a capability-aware robot task planner.
Return only a TaskPlan matching the supplied JSON schema.
Use only registered skills. ACT policies do not understand language.
Never invent target retargeting, arbitrary placement, stacking, Cartesian motion,
object classes, or recovery behavior that the registry does not support.
Never output motor angles, joint targets, or arbitrary robot commands.
For unsupported manipulation, preserve the requested semantics with the closest
registered perception and manipulation operations; local validation will mark
the unsupported object or target constraints and prevent execution. Use an empty
steps array only when no registered semantic operation applies.

Registered capabilities:
{capabilities}
"""


class RuleBasedPlanner:
    """Deterministic offline planner used by tests and hardware-free dry runs."""

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def plan(self, instruction: str) -> TaskPlan:
        lowered = instruction.lower()
        steps: list[PlanStep] = []
        if "go home" in lowered or "home pose" in lowered:
            return TaskPlan(
                instruction=instruction,
                steps=(PlanStep("step_1", "go_home", {"pose": "home"}, "abort"),),
            )
        if "inspection" in lowered:
            steps.append(PlanStep("step_1", "go_inspection", {"pose": "inspection"}, "abort"))

        colors = [color for color in ("red", "green", "blue") if re.search(rf"\b{color}\b", lowered)]
        asks_identify = any(token in lowered for token in ("identify", "detect", "find", "locate"))
        asks_all = asks_identify and any(token in lowered for token in ("all", "three", "boxes", "blocks")) and not colors
        if asks_identify:
            if not steps:
                steps.append(PlanStep("step_1", "go_inspection", {"pose": "inspection"}, "abort"))
            steps.append(
                PlanStep(f"step_{len(steps)+1}", "detect_blocks", {"camera": "wrist"}, "abort")
            )
            if not asks_all:
                for color in colors:
                    steps.append(
                        PlanStep(
                            f"step_{len(steps)+1}",
                            "identify_color",
                            {"color": color},
                            "abort",
                        )
                    )
            return TaskPlan(instruction=instruction, steps=tuple(steps))

        if steps:
            return TaskPlan(instruction=instruction, steps=tuple(steps))

        if "stack" in lowered or "on top" in lowered:
            # Keep the requested semantics visible. Validation rejects the ACT
            # step because colored-object and arbitrary-target placement were
            # never demonstrated by either policy.
            source = colors[0] if colors else "red"
            target = colors[1] if len(colors) > 1 else "blue"
            return TaskPlan(
                instruction=instruction,
                steps=(
                    PlanStep("step_1", "go_inspection", {"pose": "inspection"}),
                    PlanStep("step_2", "detect_blocks", {"camera": "wrist"}),
                    PlanStep("step_3", "identify_color", {"color": source}),
                    PlanStep("step_4", "identify_color", {"color": target}),
                    PlanStep(
                        "step_5",
                        "pick_place_variable",
                        {
                            "object": f"{source}_box",
                            "target": f"on_{target}_box",
                            "target_object": f"{target}_box",
                            "start_mode": "variable_start",
                        },
                    ),
                ),
            )
        target_is_fixed = any(token in lowered for token in ("fixed target", "fixed destination", "finish area"))
        asks_unsupported_target = any(token in lowered for token in ("new target", "unseen target", "arbitrary target"))
        if not any(token in lowered for token in ("pick", "place", "move")):
            return TaskPlan(instruction=instruction, steps=())

        variable = not any(token in lowered for token in ("original training box", "stapler", "fixed start"))
        variable = variable or any(token in lowered for token in ("variable", "random", "wherever", "different position"))
        if variable:
            skill = self.registry.get("pick_place_variable")
        else:
            skill = self.registry.get("pick_place_fixed")
        arguments = dict(skill.default_arguments)
        if asks_unsupported_target or not target_is_fixed:
            arguments["target"] = "arbitrary_target"
        arguments["start_mode"] = skill.start_mode
        return TaskPlan(
            instruction=instruction,
            steps=(PlanStep("step_1", skill.name, arguments, "abort"),),
        )


class OpenAIPlanner:
    """Structured-output LLM planner.

    A client can be injected for tests. With no client, the optional ``openai``
    package and ``OPENAI_API_KEY`` are used. This class only plans; it never
    connects to robot hardware.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        model: str,
        client: Any | None = None,
        validator: PlanValidator | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install the optional 'openai' dependency to use OpenAIPlanner") from exc
            client = OpenAI()
        self.client = client
        self.model = model
        self.registry = registry
        self.validator = validator or PlanValidator(registry)

    def plan(self, instruction: str) -> TaskPlan:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": build_system_prompt(self.registry)},
                {"role": "user", "content": instruction},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "task_plan",
                    "strict": True,
                    "schema": TaskPlan.json_schema(self.registry.names()),
                }
            },
        )
        raw = json.loads(response.output_text)
        return TaskPlan.from_dict(raw)


class OpenAICompatiblePlanner:
    """Planner for local/vLLM/Ollama-style OpenAI-compatible endpoints.

    The endpoint must expose Chat Completions and accept ``json_schema``
    response formatting. No model output is ever dispatched directly.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        model: str,
        base_url: str,
        api_key: str = "local-not-secret",
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install the optional 'openai' dependency for compatible endpoints") from exc
            client = OpenAI(base_url=base_url, api_key=api_key)
        self.client = client
        self.model = model
        self.registry = registry

    def plan(self, instruction: str) -> TaskPlan:
        schema = TaskPlan.json_schema(self.registry.names())
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": build_system_prompt(self.registry)},
                {"role": "user", "content": instruction},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "task_plan", "strict": True, "schema": schema},
            },
        )
        message = response.choices[0].message
        if getattr(message, "refusal", None):
            raise RuntimeError(f"Planner refused the request: {message.refusal}")
        if not message.content:
            raise RuntimeError("Planner returned no structured content")
        return TaskPlan.from_dict(json.loads(message.content))

