"""Offline target-highlighting and appearance-augmentation prototype.

This module deliberately operates on derived images.  It does not write to a
LeRobot dataset or checkpoint.  Production video generation should consume
reviewed per-frame masks rather than the coarse prototype polygons used here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


Point = tuple[int, int]


def polygon_mask(shape: tuple[int, int] | tuple[int, int, int], points: Sequence[Point]) -> np.ndarray:
    """Return a uint8 visible-object mask from image-coordinate polygon points."""
    if len(points) < 3:
        raise ValueError("A target polygon requires at least three points")
    height, width = shape[:2]
    polygon = np.asarray(points, dtype=np.int32)
    if np.any(polygon[:, 0] < 0) or np.any(polygon[:, 0] >= width):
        raise ValueError("Target polygon x coordinate is outside the image")
    if np.any(polygon[:, 1] < 0) or np.any(polygon[:, 1] >= height):
        raise ValueError("Target polygon y coordinate is outside the image")
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255, lineType=cv2.LINE_AA)
    return mask


def highlight_target(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply the proposed camera-independent canonical target representation.

    A translucent cyan interior makes the cue survive backbone downsampling;
    nested white/black contours remain visible on both dark and light scenes.
    The same function can be applied after an online live-camera mask is found.
    """
    _validate_image_mask(image_bgr, mask)
    result = image_bgr.copy()
    selected = mask > 0
    tint = np.empty_like(result)
    tint[:] = (255, 220, 0)  # BGR cyan, fixed across cameras and appearances.
    result[selected] = np.round(0.70 * result[selected] + 0.30 * tint[selected]).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 255), 9, cv2.LINE_AA)
    cv2.drawContours(result, contours, -1, (0, 0, 0), 3, cv2.LINE_AA)
    return result


def recolor_target(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int] = (45, 70, 205),
    *,
    texture_strength: float = 0.07,
    seed: int = 0,
) -> np.ndarray:
    """Recolor only the visible object while retaining its luminance geometry.

    The original Lab lightness channel preserves shading, folds, printed-edge
    contrast, and pose.  Low-frequency multiplicative texture avoids a flat
    synthetic fill.  No pixel outside the supplied mask is changed.
    """
    _validate_image_mask(image_bgr, mask)
    if not 0 <= texture_strength <= 0.25:
        raise ValueError("texture_strength must be in [0, 0.25]")

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2LAB)[0, 0]
    recolored_lab = lab.copy()
    recolored_lab[..., 1] = target_lab[1]
    recolored_lab[..., 2] = target_lab[2]

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, mask.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=18, sigmaY=18)
    peak = float(np.max(np.abs(noise)))
    if peak > 0:
        noise /= peak
    lightness = recolored_lab[..., 0].astype(np.float32)
    lightness *= 1.0 + texture_strength * noise
    recolored_lab[..., 0] = np.clip(lightness, 0, 255).astype(np.uint8)
    recolored = cv2.cvtColor(recolored_lab, cv2.COLOR_LAB2BGR)

    result = image_bgr.copy()
    result[mask > 0] = recolored[mask > 0]
    return result


def comparison_panel(image_bgr: np.ndarray, mask: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Create labeled original / highlighted / recolored+highlighted columns."""
    variants = [
        ("ORIGINAL", image_bgr),
        ("HIGHLIGHTED", highlight_target(image_bgr, mask)),
        ("AUGMENTED + HIGHLIGHT", highlight_target(recolor_target(image_bgr, mask, seed=seed), mask)),
    ]
    panels = []
    for label, frame in variants:
        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 48), (0, 0, 0), -1)
        cv2.putText(panel, label, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    return cv2.hconcat(panels)


def _validate_image_mask(image_bgr: np.ndarray, mask: np.ndarray) -> None:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("image must be an HxWx3 uint8 BGR array")
    if mask.shape != image_bgr.shape[:2] or mask.dtype != np.uint8:
        raise ValueError("mask must be an image-sized uint8 array")
    if not np.any(mask):
        raise ValueError("target mask is empty")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.annotations.read_text(encoding="utf-8"))
    dataset_root = Path(spec["dataset_root"]).expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source_dataset": str(dataset_root.resolve()), "artifacts": []}

    for index, sample in enumerate(spec["samples"]):
        video = dataset_root / sample["video"]
        capture = cv2.VideoCapture(str(video))
        capture.set(cv2.CAP_PROP_POS_MSEC, float(sample["timestamp_s"]) * 1000.0)
        ok, frame = capture.read()
        actual_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        capture.release()
        if not ok:
            raise RuntimeError(f"Could not decode {video} at {sample['timestamp_s']} s")
        mask = polygon_mask(frame.shape, [tuple(point) for point in sample["polygon_xy"]])
        panel = comparison_panel(frame, mask, seed=index)
        output = args.output_dir / f"{sample['camera']}_comparison.png"
        if not cv2.imwrite(str(output), panel):
            raise RuntimeError(f"Could not write {output}")
        manifest["artifacts"].append(
            {
                "camera": sample["camera"],
                "source_video": str(video.resolve()),
                "requested_timestamp_s": sample["timestamp_s"],
                "decoded_frame_index": actual_frame,
                "annotation_kind": "prototype_manual_visible_polygon",
                "output": output.name,
            }
        )

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
