from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


AUTO_CONTROLS = ("white_balance_automatic", "auto_exposure")
AUTO_DEPENDENT_CONTROLS = {
    "white_balance_temperature": ("white_balance_automatic", lambda value: value == 1),
    "exposure_time_absolute": ("auto_exposure", lambda value: value != 1),
}


class CameraSettingsError(RuntimeError):
    pass


@dataclass(frozen=True)
class SavedCameraSettings:
    path: Path
    saved_device: str
    requested: Mapping[str, int]
    saved_readback: Mapping[str, int]
    timestamp: str


@dataclass(frozen=True)
class SettingsApplicationResult:
    settings_file: str
    device: str
    requested: Mapping[str, int]
    applied: Mapping[str, int]
    skipped_auto_managed: Mapping[str, str]
    readback: Mapping[str, int]
    verified: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings_file": self.settings_file,
            "device": self.device,
            "requested": dict(self.requested),
            "applied": dict(self.applied),
            "skipped_auto_managed": dict(self.skipped_auto_managed),
            "readback": dict(self.readback),
            "verified": self.verified,
            "errors": list(self.errors),
        }


Runner = Callable[..., Any]


def find_latest_settings(directory: str | Path, prefix: str = "wrist_") -> Path | None:
    candidates = sorted(Path(directory).glob(f"{prefix}*.json"))
    return candidates[-1].resolve() if candidates else None


def load_saved_camera_settings(path: str | Path) -> SavedCameraSettings:
    settings_path = Path(path).resolve()
    if not settings_path.is_file():
        raise CameraSettingsError(f"Camera settings JSON does not exist: {settings_path}")
    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CameraSettingsError(f"Could not parse camera settings {settings_path}: {exc}") from exc
    required = {"camera", "requested", "readback", "timestamp"}
    missing = required - set(raw)
    if missing:
        raise CameraSettingsError(f"Camera settings missing keys: {sorted(missing)}")
    requested = {str(name): int(value) for name, value in raw["requested"].items()}
    readback = {str(name): int(value) for name, value in raw["readback"].items()}
    if not requested:
        raise CameraSettingsError("Camera settings requested-control map is empty")
    return SavedCameraSettings(
        path=settings_path,
        saved_device=str(raw["camera"]),
        requested=requested,
        saved_readback=readback,
        timestamp=str(raw["timestamp"]),
    )


def _run_v4l2(device: str, argument: str, runner: Runner) -> str:
    result = runner(
        ["v4l2-ctl", "--device", device, argument],
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode != 0:
        raise CameraSettingsError(f"V4L2 command failed for {device}: {argument}: {output}")
    return output


def _read_control(device: str, name: str, runner: Runner) -> int:
    output = _run_v4l2(device, f"--get-ctrl={name}", runner)
    match = re.search(r":\s*(-?\d+)", output)
    if not match:
        raise CameraSettingsError(f"Could not parse V4L2 readback for {name!r}: {output!r}")
    return int(match.group(1))


def _is_auto_managed(name: str, requested: Mapping[str, int]) -> tuple[bool, str]:
    if name not in AUTO_DEPENDENT_CONTROLS:
        return False, ""
    auto_name, enabled = AUTO_DEPENDENT_CONTROLS[name]
    auto_value = requested.get(auto_name)
    if auto_value is not None and enabled(auto_value):
        return True, f"preserved {auto_name}={auto_value}; {name} is driver-managed/inactive"
    return False, ""


def verify_applied_settings(
    saved: SavedCameraSettings,
    device: str,
    expected_applied: Mapping[str, int],
    *,
    runner: Runner = subprocess.run,
    skipped_auto_managed: Mapping[str, str] | None = None,
) -> SettingsApplicationResult:
    readback: dict[str, int] = {}
    errors: list[str] = []
    for name, expected in expected_applied.items():
        try:
            actual = _read_control(device, name, runner)
            readback[name] = actual
            if actual != expected:
                errors.append(f"{name}: requested {expected}, read back {actual}")
        except CameraSettingsError as exc:
            errors.append(str(exc))
    # Read auto-managed values for observability without comparing them to a
    # stale manual target while the corresponding automatic mode is enabled.
    for name in (skipped_auto_managed or {}):
        try:
            readback[name] = _read_control(device, name, runner)
        except CameraSettingsError as exc:
            errors.append(str(exc))
    return SettingsApplicationResult(
        settings_file=str(saved.path),
        device=device,
        requested=saved.requested,
        applied=dict(expected_applied),
        skipped_auto_managed=dict(skipped_auto_managed or {}),
        readback=readback,
        verified=not errors,
        errors=tuple(errors),
    )


def apply_and_verify_camera_settings(
    settings_path: str | Path,
    device: str,
    *,
    runner: Runner = subprocess.run,
) -> SettingsApplicationResult:
    saved = load_saved_camera_settings(settings_path)
    requested = saved.requested
    order = [name for name in AUTO_CONTROLS if name in requested]
    order.extend(name for name in requested if name not in AUTO_CONTROLS)
    applied: dict[str, int] = {}
    skipped: dict[str, str] = {}
    for name in order:
        auto_managed, reason = _is_auto_managed(name, requested)
        if auto_managed:
            skipped[name] = reason
            continue
        value = requested[name]
        _run_v4l2(device, f"--set-ctrl={name}={value}", runner)
        applied[name] = value
    result = verify_applied_settings(
        saved,
        device,
        applied,
        runner=runner,
        skipped_auto_managed=skipped,
    )
    if not result.verified:
        raise CameraSettingsError(
            "Camera control verification failed: " + "; ".join(result.errors)
        )
    return result


def test_settings_after_reopen(
    settings_path: str | Path,
    device: str,
) -> dict[str, Any]:
    import cv2

    initial = apply_and_verify_camera_settings(settings_path, device)
    camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise CameraSettingsError(
            f"Could not reopen wrist camera {device}; close tune_wrist_camera.py, ffplay, "
            "or any other process that owns the device and retry"
        )
    try:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame_ok, _ = camera.read()
    finally:
        camera.release()
    saved = load_saved_camera_settings(settings_path)
    after_reopen = verify_applied_settings(
        saved,
        device,
        initial.applied,
        skipped_auto_managed=initial.skipped_auto_managed,
    )
    if not after_reopen.verified:
        raise CameraSettingsError(
            "Settings did not persist after reopen: " + "; ".join(after_reopen.errors)
        )
    return {
        "frame_read_after_reopen": bool(frame_ok),
        "initial_application": initial.to_dict(),
        "after_reopen": after_reopen.to_dict(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply and verify saved wrist-camera V4L2 settings")
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--device", default="/dev/video3")
    parser.add_argument("--reopen-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = (
            test_settings_after_reopen(args.settings, args.device)
            if args.reopen_test
            else apply_and_verify_camera_settings(args.settings, args.device).to_dict()
        )
    except CameraSettingsError as exc:
        print(f"CAMERA SETTINGS ERROR: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
