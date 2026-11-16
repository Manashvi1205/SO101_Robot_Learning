from pathlib import Path

import numpy as np

from so101_hierarchical.camera_capture import (
    CameraDevice,
    CapturedFrame,
    PairedCaptureStore,
    discover_camera_devices,
    load_camera_config,
)


CONFIG = Path(__file__).resolve().parents[1] / "config" / "cameras.yaml"


def test_phone_stream_label_maps_to_external_logical_camera():
    specs = load_camera_config(CONFIG)
    assert specs["external"].device_name_contains == "Iriun Webcam"
    assert "droidcam" in specs["external"].source_description


def test_discovery_ignores_metadata_node():
    specs = load_camera_config(CONFIG)
    devices = [
        CameraDevice("/dev/video0", "Iriun Webcam"),
        CameraDevice("/dev/video3", "USB2.0_CAM1: USB2.0_CAM1"),
        CameraDevice("/dev/video4", "USB2.0_CAM1: USB2.0_CAM1"),
    ]
    selected = discover_camera_devices(
        specs,
        devices=devices,
        capture_probe=lambda path: path in {"/dev/video0", "/dev/video3"},
    )
    assert selected["external"].path == "/dev/video0"
    assert selected["wrist"].path == "/dev/video3"


def _frame(device: str) -> CapturedFrame:
    return CapturedFrame(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        wall_time="2026-11-15T00:00:00+00:00",
        monotonic_time=1.0,
        device=CameraDevice(device, "test camera"),
    )


def test_paired_capture_increments_and_never_overwrites(tmp_path):
    store = PairedCaptureStore(tmp_path)
    first = store.save({"wrist": _frame("/dev/a"), "external": _frame("/dev/b")})
    second = store.save({"wrist": _frame("/dev/a"), "external": _frame("/dev/b")})
    assert first["scene_id"] == "scene_001"
    assert second["scene_id"] == "scene_002"
    assert first["complete_pair"] is True
    assert (tmp_path / "metadata" / "scene_001.json").is_file()


def test_partial_capture_is_not_labeled_complete(tmp_path):
    record = PairedCaptureStore(tmp_path).save({"wrist": _frame("/dev/a"), "external": None})
    assert record["complete_pair"] is False
    assert record["cameras"]["external"]["success"] is False


def test_capture_context_is_stored_in_metadata(tmp_path):
    store = PairedCaptureStore(tmp_path)
    record = store.save(
        {"wrist": _frame("/dev/a"), "external": _frame("/dev/b")},
        context={"robot_pose_validation": {"valid": True}},
    )
    assert record["context"]["robot_pose_validation"]["valid"] is True
    metadata = (tmp_path / "metadata" / "scene_001.json").read_text()
    assert '"robot_pose_validation"' in metadata

