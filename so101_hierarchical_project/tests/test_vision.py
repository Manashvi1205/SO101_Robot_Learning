from pathlib import Path

import cv2
import numpy as np

from so101_hierarchical.vision import ColorVisionPipeline


CONFIG = Path(__file__).resolve().parents[1] / "config" / "color_detection.yaml"


def _external_scene():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (70, 285), (190, 390), (0, 0, 220), -1)
    cv2.rectangle(image, (235, 280), (365, 390), (0, 180, 0), -1)
    cv2.rectangle(image, (410, 275), (535, 390), (220, 80, 0), -1)
    return image


def test_structured_detection_reports_pixels_not_robot_coordinates():
    result = ColorVisionPipeline(CONFIG).detect_blocks(_external_scene(), camera="external")
    payload = result.to_dict()
    assert payload["detected_colors"] == ["blue", "green", "red"]
    assert payload["coordinate_space"] == "image_pixels"
    assert payload["robot_frame_coordinates_available"] is False


def test_color_selection_and_region_verification():
    pipeline = ColorVisionPipeline(CONFIG)
    image = _external_scene()
    detected = pipeline.detect_blocks(image, camera="external")
    selected = pipeline.identify_color(detected, "red")
    assert selected["success"]
    assert selected["detection"]["center_pixel"]["x"] > 70
    verification = pipeline.verify_placement(
        image, camera="external", color="red", target_region=(50, 260, 180, 140)
    )
    assert verification.success
