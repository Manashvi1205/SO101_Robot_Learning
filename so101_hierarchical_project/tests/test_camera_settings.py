import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from so101_hierarchical.camera_capture import load_camera_config
from so101_hierarchical.camera_settings import (
    CameraSettingsError,
    apply_and_verify_camera_settings,
    find_latest_settings,
    load_saved_camera_settings,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeV4L2:
    def __init__(self):
        self.state = {
            "white_balance_automatic": 0,
            "auto_exposure": 1,
            "brightness": 0,
            "contrast": 42,
            "saturation": 64,
            "gamma": 100,
            "gain": 0,
            "sharpness": 3,
            "white_balance_temperature": 4600,
            "exposure_time_absolute": 200,
        }
        self.set_calls = []

    def __call__(self, command, **_kwargs):
        argument = command[-1]
        if argument.startswith("--set-ctrl="):
            name, value = argument.removeprefix("--set-ctrl=").split("=", 1)
            self.state[name] = int(value)
            self.set_calls.append((name, int(value)))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if argument.startswith("--get-ctrl="):
            name = argument.removeprefix("--get-ctrl=")
            return SimpleNamespace(returncode=0, stdout=f"{name}: {self.state[name]}\n", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="unsupported")


def test_saved_wrist_settings_exist_and_load_auto_modes():
    path = find_latest_settings(ROOT / "camera_tuning")
    assert path is not None
    saved = load_saved_camera_settings(path)
    assert saved.requested["white_balance_automatic"] == 1
    assert saved.requested["auto_exposure"] == 3


def test_auto_modes_are_applied_and_manual_dependents_are_not_overwritten():
    path = find_latest_settings(ROOT / "camera_tuning")
    fake = FakeV4L2()
    result = apply_and_verify_camera_settings(path, "/dev/video-test", runner=fake)
    assert result.verified
    assert fake.set_calls[:2] == [("white_balance_automatic", 1), ("auto_exposure", 3)]
    assert "white_balance_temperature" in result.skipped_auto_managed
    assert "exposure_time_absolute" in result.skipped_auto_managed
    assert not any(name == "white_balance_temperature" for name, _ in fake.set_calls)
    assert not any(name == "exposure_time_absolute" for name, _ in fake.set_calls)


def test_readback_mismatch_fails_verification():
    path = find_latest_settings(ROOT / "camera_tuning")
    fake = FakeV4L2()
    original = fake.__call__

    def mismatching_runner(command, **kwargs):
        result = original(command, **kwargs)
        if command[-1] == "--get-ctrl=brightness":
            return SimpleNamespace(returncode=0, stdout="brightness: 0\n", stderr="")
        return result

    with pytest.raises(CameraSettingsError, match="verification failed"):
        apply_and_verify_camera_settings(path, "/dev/video-test", runner=mismatching_runner)


def test_only_wrist_camera_has_saved_settings_in_dual_camera_config():
    cameras = load_camera_config(ROOT / "config" / "cameras.yaml")
    assert cameras["wrist"].v4l2_settings_file.name == "wrist_20261115_165351.json"
    assert cameras["external"].v4l2_settings_file is None


def test_missing_settings_file_is_reported(tmp_path):
    with pytest.raises(CameraSettingsError, match="does not exist"):
        load_saved_camera_settings(tmp_path / "missing.json")
