from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


class SchemaError(ValueError):
    """Raised when structured planner output does not match the plan schema."""


@dataclass(frozen=True)
class CameraRequirement:
    key: str
    channels: int
    height: int
    width: int

    @property
    def chw(self) -> tuple[int, int, int]:
        return (self.channels, self.height, self.width)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionSpec:
    names: tuple[str, ...]
    representation: str = "absolute_calibrated_joint_position_targets"
    dimension: int = 6
    chunk_size: int = 100

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["names"] = list(self.names)
        return result


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    checkpoint_path: str
    dataset_path: str
    training_steps: int
    cameras: tuple[CameraRequirement, ...]
    action: ActionSpec
    start_mode: str
    supported_objects: tuple[str, ...]
    supported_targets: tuple[str, ...]
    default_arguments: Mapping[str, str] = field(default_factory=dict)
    camera_aliases: Mapping[str, str] = field(default_factory=dict)
    language_conditioned: bool = False
    supports_target_retargeting: bool = False
    implementation_type: str = "act_policy"
    config_path: str | None = None
    required_inputs: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    execution_backend: str = ""
    availability_status: str = "available"
    success_criteria: tuple[str, ...] = ()
    error_handling: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "checkpoint_path": self.checkpoint_path,
            "dataset_path": self.dataset_path,
            "training_steps": self.training_steps,
            "cameras": [camera.to_dict() for camera in self.cameras],
            "action": self.action.to_dict(),
            "start_mode": self.start_mode,
            "supported_objects": list(self.supported_objects),
            "supported_targets": list(self.supported_targets),
            "default_arguments": dict(self.default_arguments),
            "camera_aliases": dict(self.camera_aliases),
            "language_conditioned": self.language_conditioned,
            "supports_target_retargeting": self.supports_target_retargeting,
            "implementation_type": self.implementation_type,
            "config_path": self.config_path,
            "required_inputs": list(self.required_inputs),
            "preconditions": list(self.preconditions),
            "execution_backend": self.execution_backend,
            "availability_status": self.availability_status,
            "success_criteria": list(self.success_criteria),
            "error_handling": list(self.error_handling),
        }


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    skill: str
    arguments: Mapping[str, Any]
    on_failure: str = "abort"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PlanStep":
        allowed = {"step_id", "skill", "arguments", "on_failure"}
        unknown = set(raw) - allowed
        if unknown:
            raise SchemaError(f"Unknown plan-step fields: {sorted(unknown)}")
        missing = {"step_id", "skill", "arguments"} - set(raw)
        if missing:
            raise SchemaError(f"Missing plan-step fields: {sorted(missing)}")
        if not isinstance(raw["arguments"], Mapping):
            raise SchemaError("Plan-step 'arguments' must be an object")
        on_failure = raw.get("on_failure", "abort")
        if on_failure not in {"abort", "retry_once"}:
            raise SchemaError("on_failure must be 'abort' or 'retry_once'")
        return cls(
            step_id=str(raw["step_id"]),
            skill=str(raw["skill"]),
            arguments={key: value for key, value in raw["arguments"].items() if value is not None},
            on_failure=on_failure,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "skill": self.skill,
            "arguments": dict(self.arguments),
            "on_failure": self.on_failure,
        }


@dataclass(frozen=True)
class TaskPlan:
    instruction: str
    steps: tuple[PlanStep, ...]
    schema_version: str = "1.0"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TaskPlan":
        allowed = {"schema_version", "instruction", "steps"}
        unknown = set(raw) - allowed
        if unknown:
            raise SchemaError(f"Unknown plan fields: {sorted(unknown)}")
        missing = {"instruction", "steps"} - set(raw)
        if missing:
            raise SchemaError(f"Missing plan fields: {sorted(missing)}")
        if not isinstance(raw["steps"], list):
            raise SchemaError("Plan 'steps' must be an array")
        return cls(
            instruction=str(raw["instruction"]),
            steps=tuple(PlanStep.from_dict(step) for step in raw["steps"]),
            schema_version=str(raw.get("schema_version", "1.0")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "instruction": self.instruction,
            "steps": [step.to_dict() for step in self.steps],
        }

    @staticmethod
    def json_schema(skill_names: list[str]) -> dict[str, Any]:
        # Strict Structured Outputs requires every declared object property to be
        # required. Nullable values are removed by ``PlanStep.from_dict``.
        argument_names = (
            "object",
            "target",
            "start_mode",
            "pose",
            "expected_start",
            "color",
            "camera",
            "target_object",
            "verification_mode",
        )
        nullable_string = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["schema_version", "instruction", "steps"],
            "properties": {
                "schema_version": {"type": "string", "const": "1.0"},
                "instruction": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["step_id", "skill", "arguments", "on_failure"],
                        "properties": {
                            "step_id": {"type": "string"},
                            "skill": {"type": "string", "enum": skill_names},
                            "arguments": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": list(argument_names),
                                "properties": {name: nullable_string for name in argument_names},
                            },
                            "on_failure": {
                                "type": "string",
                                "enum": ["abort", "retry_once"],
                            },
                        },
                    },
                },
            },
        }

