from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from .color_detection import (
    DEFAULT_CONFIG,
    DEFAULT_INPUT,
    HSVRange,
    DetectionError,
    _mask_for_lab_ranges,
    _mask_for_ranges,
    _normalized_polygon,
    load_detection_config,
)


# Exact wrist settings used for the failed scenes 033-061 evaluation.
LEGACY_ROI = ((0.03, 0.34), (0.77, 0.34), (0.77, 1.0), (0.03, 1.0))


def _percentiles(values: Any) -> dict[str, float]:
    import numpy as np

    return {
        name: round(float(value), 2)
        for name, value in zip(("p01", "p05", "p50", "p95", "p99"), np.percentile(values, (1, 5, 50, 95, 99)))
    }


def _component_summary(mask: Any) -> dict[str, Any]:
    import cv2

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    areas = sorted((int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)), reverse=True)
    return {"component_count": len(areas), "largest_component_pixels": areas[0] if areas else 0}


def diagnose(
    input_root: Path,
    annotations_path: Path,
    config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    if output_path.exists() and any(output_path.iterdir()):
        raise DetectionError(f"Refusing to overwrite non-empty diagnostic output: {output_path}")
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    _labels, cameras = load_detection_config(config_path)
    wrist = cameras["wrist"]
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["cameras"]["wrist"]
    legacy_ranges = {
        color: tuple(HSVRange.from_mapping(item) for item in raw_config["colors"][color]["ranges"])
        for color in ("red", "green", "blue")
    }
    output_path.mkdir(parents=True, exist_ok=True)
    mask_root = output_path / "masks"
    mask_root.mkdir()

    grouped: dict[str, list[Any]] = {color: [] for color in ("red", "green", "blue")}
    background_chunks: list[Any] = []
    coverage: dict[str, dict[str, list[float]]] = {
        color: {"legacy_color": [], "legacy_color_and_roi": [], "improved_color_and_roi": []}
        for color in grouped
    }
    per_scene: dict[str, list[dict[str, Any]]] = {}
    for region in annotations["regions"]:
        per_scene.setdefault(region["scene_id"], []).append(region)

    for scene_id, regions in per_scene.items():
        image = cv2.imread(str(input_root / "wrist" / f"{scene_id}.png"))
        if image is None:
            raise DetectionError(f"Could not read {scene_id}")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        legacy_roi = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(legacy_roi, [_normalized_polygon(LEGACY_ROI, image)], 255)
        improved_roi = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(improved_roi, [_normalized_polygon(wrist.roi_polygon, image)], 255)
        for polygon in wrist.exclusion_polygons:
            cv2.fillPoly(improved_roi, [_normalized_polygon(polygon, image)], 0)

        object_union = np.zeros(image.shape[:2], dtype=np.uint8)
        for region in regions:
            cv2.fillPoly(object_union, [np.asarray(region["polygon"], dtype=np.int32)], 255)
        background = (improved_roi > 0) & (object_union == 0)
        background_chunks.append(np.column_stack((hsv[background], lab[background]))[::20])

        for region in regions:
            color = region["color"]
            polygon_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(polygon_mask, [np.asarray(region["polygon"], dtype=np.int32)], 255)
            selected = polygon_mask > 0
            grouped[color].append(np.column_stack((hsv[selected], lab[selected])))
            old = _mask_for_ranges(hsv, legacy_ranges[color])
            new = _mask_for_lab_ranges(lab, wrist.lab_colors[color])
            denominator = max(1, int(np.count_nonzero(selected)))
            coverage[color]["legacy_color"].append(float(np.count_nonzero(old[selected])) / denominator)
            coverage[color]["legacy_color_and_roi"].append(
                float(np.count_nonzero((old > 0) & (legacy_roi > 0) & selected)) / denominator
            )
            coverage[color]["improved_color_and_roi"].append(
                float(np.count_nonzero((new > 0) & (improved_roi > 0) & selected)) / denominator
            )

        for color in grouped:
            old = cv2.bitwise_and(_mask_for_ranges(hsv, legacy_ranges[color]), legacy_roi)
            new = cv2.bitwise_and(_mask_for_lab_ranges(lab, wrist.lab_colors[color]), improved_roi)
            cv2.imwrite(str(mask_root / f"{scene_id}_{color}_legacy.png"), old)
            cv2.imwrite(str(mask_root / f"{scene_id}_{color}_improved.png"), new)

    distributions: dict[str, Any] = {}
    coverage_report: dict[str, Any] = {}
    for color, chunks in grouped.items():
        pixels = np.concatenate(chunks, axis=0)
        distributions[color] = {
            channel: _percentiles(pixels[:, index])
            for index, channel in enumerate(("h", "s", "v", "l", "a", "b"))
        }
        coverage_report[color] = {
            metric: round(float(np.mean(values)), 4) for metric, values in coverage[color].items()
        }
    background_pixels = np.concatenate(background_chunks, axis=0)
    distributions["workspace_background"] = {
        channel: _percentiles(background_pixels[:, index])
        for index, channel in enumerate(("h", "s", "v", "l", "a", "b"))
    }

    representative = "scene_039"
    image = cv2.imread(str(input_root / "wrist" / f"{representative}.png"))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    legacy_blue = _mask_for_ranges(hsv, legacy_ranges["blue"])
    new_blue = _mask_for_lab_ranges(lab, wrist.lab_colors["blue"])
    report = {
        "annotations": str(annotations_path.resolve()),
        "tuning_scenes": annotations["tuning_scenes"],
        "pixel_distributions": distributions,
        "annotated_pixel_coverage": coverage_report,
        "representative_blue_mask_components_before_roi": {
            "scene_id": representative,
            "legacy": _component_summary(legacy_blue),
            "improved": _component_summary(new_blue),
        },
        "diagnosis": [
            "The legacy wrist ROI begins at normalized y=0.34 and removes upper object faces in the new inspection view.",
            "The legacy saturated-blue HSV range overlaps the blue-gray mat, merging foreground and background into oversized components.",
            "Small blue/purple highlights around the gripper survive as isolated legacy candidates.",
            "Pale red and green faces are split from their saturated side faces by single broad HSV rules and ROI clipping.",
            "Lab a separates red from green, while Lab b plus luminance separates vivid and pale blue faces from the mat and dark gripper."
        ],
    }
    (output_path / "diagnostic_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure wrist color distributions and compare masks")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = diagnose(args.input, args.annotations, args.config, args.output)
    except DetectionError as exc:
        print(f"DIAGNOSTIC ERROR: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
