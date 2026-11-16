from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .act_adapter import ACTPolicyAdapter, inspect_checkpoint
from .execution import DryRunExecutor, JsonlExecutionLogger
from .planner import OpenAICompatiblePlanner, OpenAIPlanner, RuleBasedPlanner
from .poses import JOINT_ORDER, load_pose_config, plan_pose_trajectory
from .registry import build_default_registry
from .validation import PlanValidator


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hardware-safe SO-101 hierarchical prototype")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inventory", help="Print registered skills and validate local assets")

    plan = sub.add_parser("plan", help="Generate and validate a structured plan")
    plan.add_argument("instruction")
    plan.add_argument("--planner", choices=("rule", "openai", "local"), default="rule")
    plan.add_argument("--model", default="gpt-5-mini")
    plan.add_argument("--base-url", default="http://localhost:11434/v1")

    dry = sub.add_parser("dry-run", help="Plan and simulate without loading models or hardware")
    dry.add_argument("instruction")
    dry.add_argument("--planner", choices=("rule", "openai", "local"), default="rule")
    dry.add_argument("--model", default="gpt-5-mini")
    dry.add_argument("--base-url", default="http://localhost:11434/v1")
    dry.add_argument("--log", type=Path, default=None, help="Optional JSONL execution log")

    smoke = sub.add_parser("checkpoint-smoke", help="Hardware-free checkpoint validation")
    smoke.add_argument("--skill", choices=("all", "pick_place_fixed", "pick_place_variable"), default="all")
    smoke.add_argument("--load-model", action="store_true", help="Also instantiate model weights on CPU")
    smoke.add_argument(
        "--predict-synthetic",
        action="store_true",
        help="Run one inference on zero-valued synthetic observations (implies --load-model)",
    )

    pose = sub.add_parser(
        "pose-dry-run",
        help="Plan and print a guarded joint trajectory without connecting to hardware",
    )
    pose.add_argument("skill", choices=("go_home", "go_inspection"))
    pose.add_argument("--from-pose", choices=("home", "inspection"), required=True)
    pose.add_argument(
        "--current",
        nargs=6,
        type=float,
        metavar=("PAN", "LIFT", "ELBOW", "WRIST_FLEX", "WRIST_ROLL", "GRIPPER"),
        help="Provisional measured current state in driver order; defaults to the configured --from-pose",
    )
    pose.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config" / "poses.yaml",
    )
    pose.add_argument("--summary-only", action="store_true")

    vision = sub.add_parser("vision-detect", help="Run structured color detection on an existing image")
    vision.add_argument("image", type=Path)
    vision.add_argument("--camera", choices=("wrist", "external"), required=True)
    vision.add_argument("--color", choices=("red", "green", "blue"), default=None)
    return parser


def _planner(name: str, model: str, base_url: str | None = None):
    registry = build_default_registry()
    if name == "openai":
        return OpenAIPlanner(registry, model=model), registry
    if name == "local":
        return OpenAICompatiblePlanner(
            registry,
            model=model,
            base_url=base_url or "http://localhost:11434/v1",
        ), registry
    return RuleBasedPlanner(registry), registry


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        registry = build_default_registry()
        _print_json({"skills": registry.to_dict(), "asset_problems": registry.validate_assets()})
        return 0

    if args.command == "plan":
        planner, registry = _planner(args.planner, args.model, args.base_url)
        plan = planner.plan(args.instruction)
        report = PlanValidator(registry).validate(plan)
        _print_json({"plan": plan.to_dict(), "validation": report.to_dict()})
        return 0 if report.valid else 2

    if args.command == "dry-run":
        planner, registry = _planner(args.planner, args.model, args.base_url)
        plan = planner.plan(args.instruction)
        report = PlanValidator(registry).validate(plan)
        if not report.valid:
            _print_json({"plan": plan.to_dict(), "validation": report.to_dict(), "execution": "rejected"})
            return 2
        logger = JsonlExecutionLogger(args.log) if args.log else None
        result = DryRunExecutor(registry, logger=logger).execute(plan)
        _print_json(result.to_dict())
        return 0

    if args.command == "checkpoint-smoke":
        registry = build_default_registry()
        names = (
            [skill.name for skill in registry if skill.implementation_type == "act_policy"]
            if args.skill == "all"
            else [args.skill]
        )
        results = {}
        for name in names:
            skill = registry.get(name)
            inspection = inspect_checkpoint(skill.checkpoint_path, validate_weights=True)
            result = inspection.to_dict()
            if (args.load_model or args.predict_synthetic) and inspection.complete:
                adapter = ACTPolicyAdapter(skill, device="cpu")
                adapter.load()
                result["model_loaded"] = adapter.loaded
                result["parameter_count"] = sum(p.numel() for p in adapter.policy.parameters())
                if args.predict_synthetic:
                    import numpy as np

                    observation = {"observation.state": np.zeros(6, dtype=np.float32)}
                    for camera in skill.cameras:
                        observation[camera.key] = np.zeros(
                            (camera.height, camera.width, camera.channels), dtype=np.uint8
                        )
                    action = adapter.predict_action(observation)
                    result["synthetic_action_shape"] = list(action.shape)
                    result["synthetic_action_finite"] = bool(action.isfinite().all())
            results[name] = result
        _print_json(results)
        return 0 if all(item["complete"] for item in results.values()) else 1

    if args.command == "pose-dry-run":
        config = load_pose_config(args.config)
        if args.current is None:
            current = config.poses[args.from_pose].joints
            source = "configured_pose_simulation"
        else:
            current = dict(zip(JOINT_ORDER, args.current, strict=True))
            source = "provisional_measured_state"
        trajectory = plan_pose_trajectory(
            config,
            args.skill,
            current,
            expected_start_pose=args.from_pose,
            state_source=source,
        )
        _print_json(trajectory.to_dict(include_waypoints=not args.summary_only))
        return 0 if trajectory.trajectory_valid else 2

    if args.command == "vision-detect":
        import cv2

        from .vision import ColorVisionPipeline

        image = cv2.imread(str(args.image))
        if image is None:
            raise SystemExit(f"Could not read image: {args.image}")
        pipeline = ColorVisionPipeline()
        colors = (args.color,) if args.color else None
        result = pipeline.detect_blocks(image, camera=args.camera, colors=colors)
        _print_json(result.to_dict())
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

