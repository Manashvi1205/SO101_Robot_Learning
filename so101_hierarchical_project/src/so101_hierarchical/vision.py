from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .color_detection import DEFAULT_CONFIG, Detection, detect_colored_boxes, load_detection_config


@dataclass(frozen=True)
class DetectionResult:
    camera: str
    image_width: int
    image_height: int
    detections: tuple[Detection, ...]
    requested_colors: tuple[str, ...]

    def select(self, color: str) -> Detection | None:
        return next((item for item in self.detections if item.color == color), None)

    def to_dict(self) -> dict[str, Any]:
        found = {item.color for item in self.detections}
        return {
            "camera": self.camera,
            "coordinate_space": "image_pixels",
            "robot_frame_coordinates_available": False,
            "image_size": {"width": self.image_width, "height": self.image_height},
            "requested_colors": list(self.requested_colors),
            "detected_colors": sorted(found),
            "missing_colors": [color for color in self.requested_colors if color not in found],
            "detections": [item.to_dict() for item in self.detections],
        }


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    color: str
    target_region: tuple[int, int, int, int]
    selected_detection: dict[str, Any] | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "color": self.color,
            "coordinate_space": "image_pixels",
            "target_region_xywh": list(self.target_region),
            "selected_detection": self.selected_detection,
            "reason": self.reason,
        }


class ColorVisionPipeline:
    """Structured wrapper around the frozen camera-specific color detector."""

    COLORS = ("red", "green", "blue")

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG) -> None:
        labels, cameras = load_detection_config(config_path)
        self.labels = labels
        self.cameras = cameras

    def detect_blocks(
        self,
        image: Any,
        *,
        camera: str,
        colors: tuple[str, ...] | None = None,
    ) -> DetectionResult:
        if camera not in self.cameras:
            raise ValueError(f"Unknown detector camera {camera!r}; choose from {sorted(self.cameras)}")
        requested = colors or self.COLORS
        unknown = set(requested) - set(self.labels)
        if unknown:
            raise ValueError(f"Unsupported colors: {sorted(unknown)}")
        detections = tuple(
            item for item in detect_colored_boxes(image, self.cameras[camera]) if item.color in requested
        )
        return DetectionResult(camera, int(image.shape[1]), int(image.shape[0]), detections, requested)

    @staticmethod
    def identify_color(result: DetectionResult, color: str) -> dict[str, Any]:
        selected = result.select(color)
        return {
            "success": selected is not None,
            "color": color,
            "coordinate_space": "image_pixels",
            "robot_frame_coordinates_available": False,
            "detection": selected.to_dict() if selected else None,
            "error": None if selected else "object_not_found",
        }

    def verify_placement(
        self,
        image: Any,
        *,
        camera: str,
        color: str,
        target_region: tuple[int, int, int, int],
    ) -> VerificationResult:
        result = self.detect_blocks(image, camera=camera, colors=(color,))
        selected = result.select(color)
        if selected is None:
            return VerificationResult(False, color, target_region, None, "object_not_found")
        x, y, width, height = target_region
        cx, cy = selected.center
        inside = x <= cx <= x + width and y <= cy <= y + height
        return VerificationResult(
            inside,
            color,
            target_region,
            selected.to_dict(),
            "center_inside_target_region" if inside else "center_outside_target_region",
        )


class VisionSkillBackends:
    """Injectable non-hardware backends for an execution manager.

    Frames are supplied by the caller. This class never opens cameras.
    """

    def __init__(
        self,
        pipeline: ColorVisionPipeline,
        frames: Mapping[str, Any],
        target_regions: Mapping[str, tuple[int, int, int, int]] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.frames = frames
        self.target_regions = dict(target_regions or {})
        self.latest: DetectionResult | None = None

    def detect_blocks(self, arguments: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
        camera = str(arguments.get("camera", "wrist"))
        if camera not in self.frames:
            raise RuntimeError(f"No supplied frame for camera {camera!r}")
        self.latest = self.pipeline.detect_blocks(self.frames[camera], camera=camera)
        return self.latest.to_dict()

    def identify_color(self, arguments: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
        if self.latest is None:
            raise RuntimeError("identify_color requires a preceding detect_blocks result")
        result = self.pipeline.identify_color(self.latest, str(arguments["color"]))
        if not result["success"]:
            raise RuntimeError(str(result["error"]))
        return result

    def verify_placement(self, arguments: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
        camera = str(arguments.get("camera", "wrist"))
        target = str(arguments.get("target", "configured_image_region"))
        color = str(arguments.get("color") or str(arguments.get("object", "")).removesuffix("_box"))
        if camera not in self.frames:
            raise RuntimeError(f"No supplied frame for camera {camera!r}")
        if target not in self.target_regions:
            raise RuntimeError(f"No image-space target region configured for {target!r}")
        result = self.pipeline.verify_placement(
            self.frames[camera], camera=camera, color=color, target_region=self.target_regions[target]
        )
        if not result.success:
            raise RuntimeError(result.reason)
        return result.to_dict()
