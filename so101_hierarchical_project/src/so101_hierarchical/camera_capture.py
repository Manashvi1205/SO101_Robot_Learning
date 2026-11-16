from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraSpec:
    logical_name: str
    device_name_contains: str
    width: int
    height: int
    fps: float
    fourcc: str
    preview_name: str
    source_description: str = ""
    v4l2_settings_file: Path | None = None


@dataclass(frozen=True)
class CameraDevice:
    path: str
    system_name: str


@dataclass(frozen=True)
class CapturedFrame:
    image: Any
    wall_time: str
    monotonic_time: float
    device: CameraDevice


def load_camera_config(path: str | Path) -> dict[str, CameraSpec]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    result = {}
    for name, value in raw["cameras"].items():
        result[name] = CameraSpec(
            logical_name=name,
            device_name_contains=str(value["device_name_contains"]),
            width=int(value["width"]),
            height=int(value["height"]),
            fps=float(value["fps"]),
            fourcc=str(value["fourcc"]),
            preview_name=str(value["preview_name"]),
            source_description=str(value.get("source_description", "")),
            v4l2_settings_file=(
                (config_path.parent / str(value["v4l2_settings_file"])).resolve()
                if value.get("v4l2_settings_file")
                else None
            ),
        )
    return result


def enumerate_video_devices(sysfs_root: str | Path = "/sys/class/video4linux") -> list[CameraDevice]:
    root = Path(sysfs_root)
    devices = []
    for node in sorted(root.glob("video*"), key=lambda item: item.name):
        name_file = node / "name"
        if name_file.is_file():
            devices.append(CameraDevice(f"/dev/{node.name}", name_file.read_text().strip()))
    return devices


def _is_capture_node(path: str, runner: Callable[..., Any] = subprocess.run) -> bool:
    result = runner(
        ["v4l2-ctl", "--device", path, "--all"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    return result.returncode == 0 and "Video Capture" in output and "Format Video Capture" in output


def discover_camera_devices(
    specs: Mapping[str, CameraSpec],
    *,
    devices: list[CameraDevice] | None = None,
    capture_probe: Callable[[str], bool] = _is_capture_node,
) -> dict[str, CameraDevice]:
    available = devices if devices is not None else enumerate_video_devices()
    selected: dict[str, CameraDevice] = {}
    used: set[str] = set()
    for logical_name, spec in specs.items():
        matches = [
            device
            for device in available
            if spec.device_name_contains.lower() in device.system_name.lower()
            and device.path not in used
            and capture_probe(device.path)
        ]
        if len(matches) != 1:
            raise CameraError(
                f"Expected exactly one capture node for {logical_name!r} matching "
                f"{spec.device_name_contains!r}; found {[item.path for item in matches]}"
            )
        selected[logical_name] = matches[0]
        used.add(matches[0].path)
    return selected


class OpenCVCameraStream:
    def __init__(self, spec: CameraSpec, device: CameraDevice) -> None:
        self.spec = spec
        self.device = device
        self._capture: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: CapturedFrame | None = None
        self._error: str | None = None
        self._settings_application: dict[str, Any] | None = None

    def start(self) -> None:
        import cv2

        if self.spec.v4l2_settings_file is not None:
            from .camera_settings import CameraSettingsError, apply_and_verify_camera_settings

            try:
                result = apply_and_verify_camera_settings(
                    self.spec.v4l2_settings_file, self.device.path
                )
            except CameraSettingsError as exc:
                raise CameraError(
                    f"Could not apply verified V4L2 settings to {self.spec.logical_name}: {exc}"
                ) from exc
            self._settings_application = result.to_dict()

        capture = cv2.VideoCapture(self.device.path, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.spec.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
        capture.set(cv2.CAP_PROP_FPS, self.spec.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"Could not open {self.device.path} for {self.spec.logical_name}")
        self._capture = capture
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            ok, image = self._capture.read()
            if not ok or image is None:
                self._error = f"Frame read failed from {self.device.path}"
                time.sleep(0.05)
                continue
            frame = CapturedFrame(
                image=image,
                wall_time=datetime.now(timezone.utc).isoformat(),
                monotonic_time=time.monotonic(),
                device=self.device,
            )
            with self._lock:
                self._latest = frame

    def latest(self, copy_image: bool = True) -> CapturedFrame | None:
        with self._lock:
            if self._latest is None:
                return None
            frame = self._latest
            image = frame.image.copy() if copy_image else frame.image
            return CapturedFrame(image, frame.wall_time, frame.monotonic_time, frame.device)

    def status(self) -> dict[str, Any]:
        frame = self.latest(copy_image=False)
        return {
            "logical_name": self.spec.logical_name,
            "device": self.device.path,
            "system_name": self.device.system_name,
            "has_frame": frame is not None,
            "last_frame_time": frame.wall_time if frame else None,
            "error": self._error,
            "v4l2_settings": self._settings_application,
        }

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._capture:
            self._capture.release()


class PairedCaptureStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.wrist_dir = self.root / "wrist"
        self.external_dir = self.root / "external"
        self.metadata_dir = self.root / "metadata"

    def next_scene_index(self) -> int:
        indices = []
        for folder in (self.wrist_dir, self.external_dir, self.metadata_dir):
            if not folder.exists():
                continue
            for path in folder.glob("scene_*.*"):
                try:
                    indices.append(int(path.stem.split("_")[1]))
                except (IndexError, ValueError):
                    continue
        return max(indices, default=0) + 1

    def save(
        self,
        frames: Mapping[str, CapturedFrame | None],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        import cv2

        index = self.next_scene_index()
        scene = f"scene_{index:03d}"
        self.wrist_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "wrist": self.wrist_dir / f"{scene}.png",
            "external": self.external_dir / f"{scene}.png",
        }
        record: dict[str, Any] = {
            "scene_id": scene,
            "capture_requested_at": datetime.now(timezone.utc).isoformat(),
            "complete_pair": False,
            "cameras": {},
        }
        if context is not None:
            record["context"] = dict(context)
        success = True
        for name in ("wrist", "external"):
            frame = frames.get(name)
            camera_record: dict[str, Any]
            if frame is None:
                success = False
                camera_record = {"success": False, "error": "no current frame", "path": None}
            elif paths[name].exists():
                success = False
                camera_record = {"success": False, "error": "refusing to overwrite", "path": str(paths[name])}
            else:
                written = bool(cv2.imwrite(str(paths[name]), frame.image))
                success = success and written
                camera_record = {
                    "success": written,
                    "error": None if written else "cv2.imwrite failed",
                    "path": str(paths[name]) if written else None,
                    "frame_wall_time": frame.wall_time,
                    "frame_monotonic_time": frame.monotonic_time,
                    "device_path": frame.device.path,
                    "device_name": frame.device.system_name,
                }
            record["cameras"][name] = camera_record
        record["complete_pair"] = success
        metadata_path = self.metadata_dir / f"{scene}.json"
        metadata_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        record["metadata_path"] = str(metadata_path)
        return record


class TerminalCommandReader:
    def __init__(self) -> None:
        self.commands: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        while True:
            try:
                command = input("command> ").strip().lower()
            except EOFError:
                command = "quit"
            self.commands.put(command)
            if command == "quit":
                return

