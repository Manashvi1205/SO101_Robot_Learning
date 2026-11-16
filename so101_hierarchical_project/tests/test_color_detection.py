from pathlib import Path

import cv2
import numpy as np
import pytest

from so101_hierarchical.color_detection import (
    DetectionError,
    detect_colored_boxes,
    load_detection_config,
)


CONFIG = Path(__file__).resolve().parents[1] / "config" / "color_detection.yaml"


def test_camera_specific_thresholds_load():
    labels, cameras = load_detection_config(CONFIG)
    assert labels == ("red", "green", "blue")
    assert set(cameras) == {"wrist", "external"}
    assert cameras["wrist"].colors["green"] != cameras["external"].colors["green"]


def test_detector_returns_box_center_area_and_quality():
    _, cameras = load_detection_config(CONFIG)
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (70, 285), (190, 390), (0, 0, 220), -1)
    cv2.rectangle(image, (235, 280), (365, 390), (0, 180, 0), -1)
    cv2.rectangle(image, (410, 275), (535, 390), (220, 80, 0), -1)
    detections = detect_colored_boxes(image, cameras["external"])
    assert {item.color for item in detections} == {"red", "green", "blue"}
    for item in detections:
        assert item.contour_area > 1_000
        assert 0.0 <= item.quality_score <= 1.0
        assert len(item.bounding_box) == 4
        assert len(item.center) == 2


def test_detector_does_not_invent_objects_on_blank_image():
    _, cameras = load_detection_config(CONFIG)
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    assert detect_colored_boxes(image, cameras["wrist"]) == []


def test_empty_image_is_rejected():
    _, cameras = load_detection_config(CONFIG)
    with pytest.raises(DetectionError):
        detect_colored_boxes(np.array([], dtype=np.uint8), cameras["wrist"])


def test_wrist_lab_detector_rejects_mat_and_finds_three_boxes():
    _, cameras = load_detection_config(CONFIG)
    image = np.full((720, 1280, 3), (95, 72, 65), dtype=np.uint8)
    cv2.rectangle(image, (250, 100), (430, 280), (60, 70, 240), -1)
    cv2.rectangle(image, (520, 100), (700, 280), (60, 210, 80), -1)
    cv2.rectangle(image, (790, 100), (970, 280), (240, 120, 20), -1)
    detections = detect_colored_boxes(image, cameras["wrist"])
    assert {item.color for item in detections} == {"red", "green", "blue"}
    assert all(item.rotated_box for item in detections)
