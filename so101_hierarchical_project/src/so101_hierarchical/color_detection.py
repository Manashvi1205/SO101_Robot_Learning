from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "color_detection.yaml"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "vision"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "perception_eval_v1"
DRAW_COLORS = {"red": (40, 40, 255), "green": (40, 210, 40), "blue": (255, 100, 20)}


class DetectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class HSVRange:
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HSVRange":
        result = cls(**{field: int(value[field]) for field in cls.__dataclass_fields__})
        if not (0 <= result.h_min <= result.h_max <= 179):
            raise DetectionError(f"Invalid hue range: {result}")
        for low, high in ((result.s_min, result.s_max), (result.v_min, result.v_max)):
            if not 0 <= low <= high <= 255:
                raise DetectionError(f"Invalid HSV range: {result}")
        return result


@dataclass(frozen=True)
class LabRange:
    l_min: int
    l_max: int
    a_min: int
    a_max: int
    b_min: int
    b_max: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LabRange":
        result = cls(**{field: int(value[field]) for field in cls.__dataclass_fields__})
        for low, high in (
            (result.l_min, result.l_max),
            (result.a_min, result.a_max),
            (result.b_min, result.b_max),
        ):
            if not 0 <= low <= high <= 255:
                raise DetectionError(f"Invalid Lab range: {result}")
        return result


@dataclass(frozen=True)
class CameraDetectionConfig:
    name: str
    algorithm: str
    min_contour_area: float
    max_area_fraction: float
    min_bbox_width: int
    min_bbox_height: int
    max_bbox_width: int
    max_bbox_height: int
    morphology_kernel: int
    morphology_iterations: int
    close_kernel: int
    close_iterations: int
    review_score_threshold: float
    roi_polygon: tuple[tuple[float, float], ...]
    exclusion_polygons: tuple[tuple[tuple[float, float], ...], ...]
    colors: Mapping[str, tuple[HSVRange, ...]]
    lab_colors: Mapping[str, tuple[LabRange, ...]]
    min_rotated_rectangularity: float
    min_solidity: float


@dataclass(frozen=True)
class Detection:
    color: str
    bounding_box: tuple[int, int, int, int]
    center: tuple[float, float]
    contour_area: float
    quality_score: float
    rectangularity: float
    solidity: float
    rotated_box: tuple[tuple[float, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        x, y, width, height = self.bounding_box
        return {
            "color": self.color,
            "bounding_box": {"x": x, "y": y, "width": width, "height": height},
            "center_pixel": {"x": round(self.center[0], 2), "y": round(self.center[1], 2)},
            "contour_area": round(self.contour_area, 2),
            "detection_quality_score": round(self.quality_score, 4),
            "quality_components": {
                "rectangularity": round(self.rectangularity, 4),
                "solidity": round(self.solidity, 4),
            },
            "rotated_box": [
                {"x": round(point[0], 2), "y": round(point[1], 2)}
                for point in self.rotated_box
            ],
        }


def load_detection_config(path: str | Path) -> tuple[tuple[str, ...], dict[str, CameraDetectionConfig]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    labels = tuple(str(value) for value in raw["labels"])
    cameras: dict[str, CameraDetectionConfig] = {}
    for name, values in raw["cameras"].items():
        kernel = int(values["morphology_kernel"])
        if kernel < 1 or kernel % 2 == 0:
            raise DetectionError(f"{name}: morphology_kernel must be a positive odd integer")
        colors = {
            label: tuple(HSVRange.from_mapping(item) for item in values["colors"][label]["ranges"])
            for label in labels
        }
        lab_colors = {
            label: tuple(
                LabRange.from_mapping(item)
                for item in values.get("lab_colors", {}).get(label, {}).get("ranges", [])
            )
            for label in labels
        }
        close_kernel = int(values.get("close_kernel", kernel))
        if close_kernel < 1 or close_kernel % 2 == 0:
            raise DetectionError(f"{name}: close_kernel must be a positive odd integer")
        cameras[name] = CameraDetectionConfig(
            name=name,
            algorithm=str(values.get("algorithm", "hsv_legacy")),
            min_contour_area=float(values["min_contour_area"]),
            max_area_fraction=float(values["max_area_fraction"]),
            min_bbox_width=int(values["min_bbox_width"]),
            min_bbox_height=int(values["min_bbox_height"]),
            max_bbox_width=int(values["max_bbox_width"]),
            max_bbox_height=int(values["max_bbox_height"]),
            morphology_kernel=kernel,
            morphology_iterations=int(values["morphology_iterations"]),
            close_kernel=close_kernel,
            close_iterations=int(values.get("close_iterations", values["morphology_iterations"])),
            review_score_threshold=float(values["review_score_threshold"]),
            roi_polygon=tuple(
                (float(point[0]), float(point[1])) for point in values["roi_polygon"]
            ),
            exclusion_polygons=tuple(
                tuple((float(point[0]), float(point[1])) for point in polygon)
                for polygon in values.get("exclusion_polygons", [])
            ),
            colors=colors,
            lab_colors=lab_colors,
            min_rotated_rectangularity=float(values.get("min_rotated_rectangularity", 0.0)),
            min_solidity=float(values.get("min_solidity", 0.0)),
        )
    return labels, cameras


def _mask_for_ranges(hsv_image: Any, ranges: Sequence[HSVRange]) -> Any:
    import cv2
    import numpy as np

    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for item in ranges:
        current = cv2.inRange(
            hsv_image,
            (item.h_min, item.s_min, item.v_min),
            (item.h_max, item.s_max, item.v_max),
        )
        mask = cv2.bitwise_or(mask, current)
    return mask


def _mask_for_lab_ranges(lab_image: Any, ranges: Sequence[LabRange]) -> Any:
    import cv2
    import numpy as np

    mask = np.zeros(lab_image.shape[:2], dtype=np.uint8)
    for item in ranges:
        current = cv2.inRange(
            lab_image,
            (item.l_min, item.a_min, item.b_min),
            (item.l_max, item.a_max, item.b_max),
        )
        mask = cv2.bitwise_or(mask, current)
    return mask


def _normalized_polygon(points: Sequence[tuple[float, float]], image: Any) -> Any:
    import numpy as np

    return np.array(
        [
            [
                round(x * (image.shape[1] - 1)),
                round(y * (image.shape[0] - 1)),
            ]
            for x, y in points
        ],
        dtype=np.int32,
    )


def detect_colored_boxes(image: Any, config: CameraDetectionConfig) -> list[Detection]:
    import cv2
    import numpy as np

    if image is None or image.size == 0:
        raise DetectionError("Cannot detect objects in an empty image")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_area = float(image.shape[0] * image.shape[1])
    roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    roi_points = _normalized_polygon(config.roi_polygon, image)
    cv2.fillPoly(roi_mask, [roi_points], 255)
    for polygon in config.exclusion_polygons:
        cv2.fillPoly(roi_mask, [_normalized_polygon(polygon, image)], 0)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.morphology_kernel, config.morphology_kernel)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.close_kernel, config.close_kernel)
    )
    detections: list[Detection] = []
    for label, ranges in config.colors.items():
        if config.algorithm == "lab_color_shape_v2":
            lab_ranges = config.lab_colors.get(label, ())
            if not lab_ranges:
                raise DetectionError(f"{config.name}: no Lab ranges configured for {label}")
            mask = _mask_for_lab_ranges(lab, lab_ranges)
        else:
            mask = _mask_for_ranges(hsv, ranges)
        mask = cv2.bitwise_and(mask, roi_mask)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations=config.morphology_iterations
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, close_kernel, iterations=config.close_iterations
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < config.min_contour_area or area > image_area * config.max_area_fraction:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if not (
                config.min_bbox_width <= width <= config.max_bbox_width
                and config.min_bbox_height <= height <= config.max_bbox_height
            ):
                continue
            rect_area = float(max(1, width * height))
            hull_area = float(max(1.0, cv2.contourArea(cv2.convexHull(contour))))
            rotated_rect = cv2.minAreaRect(contour)
            rotated_width, rotated_height = rotated_rect[1]
            rotated_area = float(max(1.0, rotated_width * rotated_height))
            geometry_area = rotated_area if config.algorithm == "lab_color_shape_v2" else rect_area
            rectangularity = min(1.0, area / geometry_area)
            solidity = min(1.0, area / hull_area)
            if (
                rectangularity < config.min_rotated_rectangularity
                or solidity < config.min_solidity
            ):
                continue
            moments = cv2.moments(contour)
            if moments["m00"]:
                center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
            else:
                center = (x + width / 2.0, y + height / 2.0)
            area_score = min(1.0, math.log1p(area) / math.log1p(config.min_contour_area * 12.0))
            quality = max(
                0.0,
                min(1.0, 0.35 * area_score + 0.35 * rectangularity + 0.30 * solidity),
            )
            candidates.append(
                Detection(
                    color=label,
                    bounding_box=(x, y, width, height),
                    center=center,
                    contour_area=area,
                    quality_score=quality,
                    rectangularity=rectangularity,
                    solidity=solidity,
                    rotated_box=tuple(
                        (float(point[0]), float(point[1]))
                        for point in cv2.boxPoints(rotated_rect)
                    ),
                )
            )
        # There is one physical box of each color. Keep the strongest component
        # and expose missing labels for review instead of returning background blobs.
        if candidates:
            # Within the configured workspace ROI, the box is expected to have
            # the largest connected color support. Ranking by quality first can
            # incorrectly prefer a small, very rectangular gripper reflection.
            detections.append(max(candidates, key=lambda item: (item.contour_area, item.quality_score)))
    return sorted(detections, key=lambda item: item.color)


def annotate_image(image: Any, detections: Sequence[Detection], scene_id: str, camera: str) -> Any:
    import cv2

    annotated = image.copy()
    cv2.putText(
        annotated,
        f"{scene_id} | {camera}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for detection in detections:
        x, y, width, height = detection.bounding_box
        color = DRAW_COLORS[detection.color]
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 3)
        if detection.rotated_box:
            import numpy as np

            rotated = np.asarray(detection.rotated_box, dtype=np.int32)
            cv2.polylines(annotated, [rotated], True, color, 2, cv2.LINE_AA)
        cv2.circle(
            annotated,
            (int(round(detection.center[0])), int(round(detection.center[1]))),
            5,
            color,
            -1,
        )
        label = f"{detection.color} {detection.quality_score:.2f} A={detection.contour_area:.0f}"
        cv2.putText(
            annotated,
            label,
            (x, max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _write_contact_sheet(paths: Sequence[Path], destination: Path, columns: int = 4) -> None:
    import cv2
    import numpy as np

    tiles = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise DetectionError(f"Could not read annotated image {path}")
        image = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
        tiles.append(image)
    rows = math.ceil(len(tiles) / columns)
    blank = np.zeros_like(tiles[0])
    while len(tiles) < rows * columns:
        tiles.append(blank.copy())
    sheet = np.vstack([np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)])
    if not cv2.imwrite(str(destination), sheet):
        raise DetectionError(f"Failed to write review sheet {destination}")


def evaluate_dataset(
    input_root: str | Path,
    output_root: str | Path,
    config_path: str | Path,
    *,
    start_scene: int = 2,
    end_scene: int = 32,
) -> dict[str, Any]:
    import cv2

    input_path = Path(input_root)
    output_path = Path(output_root)
    if output_path.exists() and any(output_path.iterdir()):
        raise DetectionError(f"Refusing to overwrite non-empty derived output directory: {output_path}")
    labels, camera_configs = load_detection_config(config_path)
    output_path.mkdir(parents=True, exist_ok=True)
    annotated_root = output_path / "annotated"
    results_root = output_path / "results"
    annotated_root.mkdir()
    results_root.mkdir()

    scene_records: list[dict[str, Any]] = []
    annotated_paths: dict[str, list[Path]] = {name: [] for name in camera_configs}
    for index in range(start_scene, end_scene + 1):
        scene_id = f"scene_{index:03d}"
        scene_record: dict[str, Any] = {"scene_id": scene_id, "cameras": {}}
        for camera, camera_config in camera_configs.items():
            source = input_path / camera / f"{scene_id}.png"
            image = cv2.imread(str(source))
            if image is None:
                raise DetectionError(f"Missing or unreadable source image: {source}")
            detections = detect_colored_boxes(image, camera_config)
            found = {item.color for item in detections}
            missing = [label for label in labels if label not in found]
            low_quality = [
                item.color
                for item in detections
                if item.quality_score < camera_config.review_score_threshold
            ]
            camera_record = {
                "source_image": str(source.resolve()),
                "image_size": {"width": image.shape[1], "height": image.shape[0]},
                "detections": [item.to_dict() for item in detections],
                "detected_labels": sorted(found),
                "algorithm_missing_labels": missing,
                "low_quality_labels": low_quality,
                "requires_manual_review": bool(missing or low_quality),
            }
            scene_record["cameras"][camera] = camera_record
            camera_output = annotated_root / camera
            camera_output.mkdir(exist_ok=True)
            annotated_path = camera_output / f"{scene_id}.png"
            if not cv2.imwrite(str(annotated_path), annotate_image(image, detections, scene_id, camera)):
                raise DetectionError(f"Could not write {annotated_path}")
            annotated_paths[camera].append(annotated_path)
        result_path = results_root / f"{scene_id}.json"
        result_path.write_text(json.dumps(scene_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        scene_records.append(scene_record)

    summaries: dict[str, Any] = {}
    for camera, camera_config in camera_configs.items():
        per_label: dict[str, Any] = {}
        for label in labels:
            label_detections = [
                detection
                for scene in scene_records
                for detection in scene["cameras"][camera]["detections"]
                if detection["color"] == label
            ]
            per_label[label] = {
                "detected_scenes": len(label_detections),
                "total_scenes": len(scene_records),
                "algorithm_detection_rate": round(len(label_detections) / len(scene_records), 4),
                "mean_quality_score": (
                    round(mean(item["detection_quality_score"] for item in label_detections), 4)
                    if label_detections
                    else None
                ),
            }
        review_scenes = [
            scene["scene_id"]
            for scene in scene_records
            if scene["cameras"][camera]["requires_manual_review"]
        ]
        summaries[camera] = {
            "scene_count": len(scene_records),
            "per_label": per_label,
            "algorithm_flagged_review_scenes": review_scenes,
            "review_score_threshold": camera_config.review_score_threshold,
        }
        sheet_path = output_path / f"review_sheet_{camera}.png"
        _write_contact_sheet(annotated_paths[camera], sheet_path)

    report = {
        "dataset": {
            "input_root": str(input_path.resolve()),
            "included_scene_range": [start_scene, end_scene],
            "excluded_scenes": ["scene_001"],
            "scene_count": len(scene_records),
        },
        "method": {
            camera: camera_config.algorithm
            for camera, camera_config in camera_configs.items()
        },
        "ground_truth_status": "unavailable; metrics are algorithm coverage pending manual review",
        "camera_summaries": summaries,
        "cross_camera_comparison": {
            label: {
                camera: summaries[camera]["per_label"][label]["algorithm_detection_rate"]
                for camera in camera_configs
            }
            for label in labels
        },
    }
    (output_path / "detections.json").write_text(
        json.dumps(scene_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_path / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate colored-box HSV detection")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start-scene", type=int, default=2)
    parser.add_argument("--end-scene", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_dataset(
            args.input,
            args.output,
            args.config,
            start_scene=args.start_scene,
            end_scene=args.end_scene,
        )
    except DetectionError as exc:
        print(f"DETECTION ERROR: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
