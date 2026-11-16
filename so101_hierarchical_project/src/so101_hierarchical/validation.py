from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import TaskPlan
from .registry import SkillRegistry


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    step_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "step_id": self.step_id}


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    issues: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


class PlanValidationError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        text = "; ".join(issue.message for issue in report.issues)
        super().__init__(text)


class PlanValidator:
    """Reject plans that exceed the explicitly registered ACT capabilities."""

    def __init__(
        self,
        registry: SkillRegistry,
        max_steps: int = 8,
        *,
        mode: str = "dry_run",
        available_cameras: set[str] | None = None,
        satisfied_preconditions: set[str] | None = None,
    ) -> None:
        self.registry = registry
        self.max_steps = max_steps
        self.mode = mode
        self.available_cameras = available_cameras
        self.satisfied_preconditions = satisfied_preconditions or set()

    def validate(self, plan: TaskPlan) -> ValidationReport:
        issues: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []
        if not plan.steps:
            issues.append(ValidationIssue("empty_plan", "A plan must contain at least one step"))
        if len(plan.steps) > self.max_steps:
            issues.append(
                ValidationIssue(
                    "too_many_steps",
                    f"Plan has {len(plan.steps)} steps; maximum is {self.max_steps}",
                )
            )

        seen_ids: set[str] = set()
        for step in plan.steps:
            if step.step_id in seen_ids:
                issues.append(
                    ValidationIssue("duplicate_step_id", f"Duplicate step id: {step.step_id}", step.step_id)
                )
            seen_ids.add(step.step_id)

            try:
                skill = self.registry.get(step.skill)
            except KeyError:
                issues.append(
                    ValidationIssue("unknown_skill", f"Unknown skill: {step.skill}", step.step_id)
                )
                continue

            if not skill.execution_backend:
                issues.append(ValidationIssue("missing_backend", f"{step.skill} has no executable backend", step.step_id))
            if skill.availability_status == "unavailable":
                issues.append(ValidationIssue("skill_unavailable", f"{step.skill} is unavailable", step.step_id))
            if self.mode == "hardware" and "hardware_locked" in skill.availability_status:
                issues.append(
                    ValidationIssue(
                        "hardware_execution_locked",
                        f"{step.skill} is not authorized for hardware execution",
                        step.step_id,
                    )
                )
            if self.available_cameras is not None:
                if skill.implementation_type in {"perception", "verification"}:
                    required_cameras = {str(step.arguments.get("camera", "wrist"))}
                else:
                    required_cameras = {camera.key for camera in skill.cameras}
                aliases = {
                    "observation.images.wrist": "wrist",
                    "observation.images.front": "external",
                    "observation.images.phone": "external",
                }
                missing_cameras = {
                    key
                    for key in required_cameras
                    if key not in self.available_cameras and aliases.get(key) not in self.available_cameras
                }
                if missing_cameras:
                    issues.append(
                        ValidationIssue(
                            "required_camera_unavailable",
                            f"{step.skill} requires unavailable cameras {sorted(missing_cameras)}",
                            step.step_id,
                        )
                    )
            for precondition in skill.preconditions:
                if precondition not in self.satisfied_preconditions:
                    item = ValidationIssue(
                        "precondition_unconfirmed",
                        f"{step.skill}: {precondition}",
                        step.step_id,
                    )
                    (issues if self.mode == "hardware" else warnings).append(item)

            if skill.implementation_type == "joint_trajectory":
                allowed_arguments = {"pose", "expected_start"}
                unknown_arguments = set(step.arguments) - allowed_arguments
                if unknown_arguments:
                    issues.append(
                        ValidationIssue(
                            "unsupported_arguments",
                            f"{step.skill} does not accept arguments {sorted(unknown_arguments)}",
                            step.step_id,
                        )
                    )
                pose = step.arguments.get("pose", skill.default_arguments.get("pose"))
                if pose not in skill.supported_targets:
                    issues.append(
                        ValidationIssue(
                            "unsupported_target",
                            f"{step.skill} cannot move to pose {pose!r}; supported poses: "
                            f"{list(skill.supported_targets)}",
                            step.step_id,
                        )
                    )
                continue

            if skill.implementation_type == "perception":
                allowed_arguments = {"camera"}
                unknown_arguments = set(step.arguments) - allowed_arguments
                if unknown_arguments:
                    issues.append(ValidationIssue("unsupported_arguments", f"{step.skill} does not accept arguments {sorted(unknown_arguments)}", step.step_id))
                if step.arguments.get("camera", "wrist") not in {"wrist", "external"}:
                    issues.append(ValidationIssue("unsupported_camera", f"Unsupported detector camera {step.arguments.get('camera')!r}", step.step_id))
                continue

            if skill.implementation_type == "perception_query":
                allowed_arguments = {"color"}
                unknown_arguments = set(step.arguments) - allowed_arguments
                if unknown_arguments:
                    issues.append(ValidationIssue("unsupported_arguments", f"{step.skill} does not accept arguments {sorted(unknown_arguments)}", step.step_id))
                if step.arguments.get("color") not in {"red", "green", "blue"}:
                    issues.append(ValidationIssue("unsupported_color", f"Unsupported color {step.arguments.get('color')!r}", step.step_id))
                continue

            if skill.implementation_type == "verification":
                allowed_arguments = {"object", "target", "color", "camera", "verification_mode"}
                unknown_arguments = set(step.arguments) - allowed_arguments
                if unknown_arguments:
                    issues.append(ValidationIssue("unsupported_arguments", f"{step.skill} does not accept arguments {sorted(unknown_arguments)}", step.step_id))
                if step.arguments.get("target") != "configured_image_region":
                    issues.append(ValidationIssue("unsupported_target", "Visual verification requires a configured image-space target region", step.step_id))
                continue

            allowed_arguments = {"object", "target", "start_mode", "target_object"}
            unknown_arguments = set(step.arguments) - allowed_arguments
            if unknown_arguments:
                issues.append(
                    ValidationIssue(
                        "unsupported_arguments",
                        f"{step.skill} does not accept arguments {sorted(unknown_arguments)}",
                        step.step_id,
                    )
                )

            target = step.arguments.get("target", skill.default_arguments.get("target"))
            if target not in skill.supported_targets:
                issues.append(
                    ValidationIssue(
                        "unsupported_target",
                        f"{step.skill} cannot place at target {target!r}; supported targets: "
                        f"{list(skill.supported_targets)}",
                        step.step_id,
                    )
                )

            object_name = step.arguments.get("object", skill.default_arguments.get("object"))
            if object_name not in skill.supported_objects:
                issues.append(
                    ValidationIssue(
                        "unsupported_object",
                        f"{step.skill} is not validated for object {object_name!r}; supported objects: "
                        f"{list(skill.supported_objects)}",
                        step.step_id,
                    )
                )

            start_mode = step.arguments.get("start_mode", skill.start_mode)
            if start_mode != skill.start_mode:
                issues.append(
                    ValidationIssue(
                        "unsupported_start_mode",
                        f"{step.skill} requires start_mode={skill.start_mode!r}, got {start_mode!r}",
                        step.step_id,
                    )
                )

            if step.arguments.get("retarget", False) or step.arguments.get("target_offset") is not None:
                issues.append(
                    ValidationIssue(
                        "retargeting_not_supported",
                        "Geometric target retargeting is not implemented or validated",
                        step.step_id,
                    )
                )

        return ValidationReport(valid=not issues, issues=tuple(issues), warnings=tuple(warnings))

    def require_valid(self, plan: TaskPlan) -> None:
        report = self.validate(plan)
        if not report.valid:
            raise PlanValidationError(report)

