from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .models import SkillSpec
from .registry import SkillRegistry


class ObservationError(ValueError):
    pass


@dataclass(frozen=True)
class CheckpointInspection:
    path: str
    complete: bool
    missing_files: tuple[str, ...]
    safetensors_valid: bool | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "complete": self.complete,
            "missing_files": list(self.missing_files),
            "safetensors_valid": self.safetensors_valid,
            "error": self.error,
        }


def inspect_checkpoint(path: str | Path, validate_weights: bool = False) -> CheckpointInspection:
    checkpoint = Path(path)
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    missing = tuple(name for name in required if not (checkpoint / name).is_file())
    valid: bool | None = None
    error: str | None = None
    if validate_weights and "model.safetensors" not in missing:
        try:
            from safetensors import safe_open

            with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as handle:
                list(handle.keys())
            valid = True
        except Exception as exc:  # safetensors exposes backend-specific exceptions
            valid = False
            error = f"{type(exc).__name__}: {exc}"
    return CheckpointInspection(
        path=str(checkpoint),
        complete=not missing and valid is not False,
        missing_files=missing,
        safetensors_valid=valid,
        error=error,
    )


def validate_saved_contract(skill: SkillSpec) -> None:
    """Fail if registry observation/action metadata drifts from saved ACT config."""
    config_path = Path(skill.checkpoint_path) / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    input_features = raw.get("input_features", {})
    expected = {"observation.state": [skill.action.dimension]}
    expected.update({camera.key: list(camera.chw) for camera in skill.cameras})
    actual = {key: value.get("shape") for key, value in input_features.items()}
    if actual != expected:
        raise RuntimeError(f"Saved input contract mismatch for {skill.name}: {actual} != {expected}")
    output_shape = raw.get("output_features", {}).get("action", {}).get("shape")
    if output_shape != [skill.action.dimension]:
        raise RuntimeError(f"Saved action shape mismatch for {skill.name}: {output_shape}")
    if int(raw.get("chunk_size", -1)) != skill.action.chunk_size:
        raise RuntimeError(f"Saved chunk size mismatch for {skill.name}: {raw.get('chunk_size')}")


class ACTPolicyAdapter:
    """Callable ACT inference adapter with no robot-dispatch capability.

    ``load`` and ``predict_action`` are hardware-free. Camera and robot I/O are
    intentionally outside this class so an LLM cannot directly command motors.
    """

    def __init__(self, skill: SkillSpec, device: str = "cpu") -> None:
        self.skill = skill
        self.device = device
        self.config: Any | None = None
        self.policy: Any | None = None
        self.preprocessor: Any | None = None
        self.postprocessor: Any | None = None

    @property
    def loaded(self) -> bool:
        return self.policy is not None

    def load(self) -> None:
        inspection = inspect_checkpoint(self.skill.checkpoint_path, validate_weights=True)
        if not inspection.complete:
            raise RuntimeError(f"Checkpoint is incomplete: {inspection.to_dict()}")
        validate_saved_contract(self.skill)

        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        config = ACTConfig.from_pretrained(self.skill.checkpoint_path, device=self.device)
        policy = ACTPolicy.from_pretrained(
            self.skill.checkpoint_path,
            config=config,
            local_files_only=True,
        )
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=self.skill.checkpoint_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        self.config = config
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def required_observation_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {"observation.state": (6,)}
        shapes.update({camera.key: camera.chw for camera in self.skill.cameras})
        return shapes

    def validate_observation(self, observation: Mapping[str, Any]) -> None:
        required = self.required_observation_shapes()
        missing = set(required) - set(observation)
        if missing:
            raise ObservationError(f"Missing observation keys: {sorted(missing)}")

        for key, expected in required.items():
            value = observation[key]
            shape = tuple(getattr(value, "shape", ()))
            accepted = {expected}
            if key.startswith("observation.images"):
                accepted.add((expected[1], expected[2], expected[0]))
            if shape not in accepted:
                raise ObservationError(f"{key} has shape {shape}; expected one of {sorted(accepted)}")

    def _to_policy_input(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np
        import torch

        result: dict[str, Any] = {}
        for key in self.required_observation_shapes():
            value = observation[key]
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
            tensor = tensor.to(dtype=torch.float32)
            if key.startswith("observation.images"):
                if tensor.ndim == 3 and tensor.shape[-1] == 3:
                    tensor = tensor.permute(2, 0, 1)
                if tensor.numel() and tensor.max() > 1:
                    tensor = tensor / 255.0
            result[key] = tensor
        return result

    def predict_action(self, observation: Mapping[str, Any]) -> Any:
        if not self.loaded:
            raise RuntimeError("Call load() before predict_action()")
        self.validate_observation(observation)
        import torch

        policy_input = self.preprocessor(self._to_policy_input(observation))
        with torch.inference_mode():
            normalized_action = self.policy.select_action(policy_input)
            action = self.postprocessor(normalized_action)
        return action.squeeze(0).detach().cpu()

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()


def build_act_adapter(
    registry: SkillRegistry,
    skill_name: str,
    *,
    device: str = "cpu",
) -> ACTPolicyAdapter:
    """Construct an inference-only wrapper for one of the two registered 50k policies."""
    skill = registry.get(skill_name)
    if skill_name not in {"pick_place_fixed", "pick_place_variable"} or skill.implementation_type != "act_policy":
        raise ValueError(f"{skill_name!r} is not a registered ACT manipulation policy")
    return ACTPolicyAdapter(skill, device=device)

