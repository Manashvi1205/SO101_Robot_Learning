from __future__ import annotations

import argparse
import json
import select
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from .camera_capture import (
    CameraError,
    OpenCVCameraStream,
    PairedCaptureStore,
    TerminalCommandReader,
    discover_camera_devices,
    load_camera_config,
)
from .poses import JOINT_ORDER, load_pose_config, plan_pose_trajectory
from .robot_motion import (
    RobotSafetyError,
    connect_follower,
    discover_serial_ports,
    execute_validated_trajectory,
    plan_from_measured_state,
    require_motion_approval,
    validate_measured_named_pose,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POSES = PROJECT_ROOT / "config" / "poses.yaml"
DEFAULT_CAMERAS = PROJECT_ROOT / "config" / "cameras.yaml"
DEFAULT_DATA = PROJECT_ROOT / "data" / "vision"
CALIBRATION_DIR = Path("/home/manashvi/.cache/huggingface/lerobot/calibration/robots/so_follower")
PROVISIONAL_MOVE_PHRASE = "MOVE TO PROVISIONAL INSPECTION"
VIA_HOME_MOVE_PHRASE = "MOVE VIA HOME TO PROVISIONAL INSPECTION"


class EmergencyStopMonitor:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self._finish = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        print("During motion type STOP and press Enter, or press Ctrl+C, to stop issuing commands.")
        print("Software stop does not instantly remove motor torque; cut robot power if motion is unsafe.")
        self._thread.start()

    def _loop(self) -> None:
        while not self._finish.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if readable and sys.stdin.readline().strip() == "STOP":
                self.stop_event.set()
                return

    def close(self) -> None:
        self._finish.set()
        self._thread.join(timeout=1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supervised SO-101 positioning and paired image collection")
    parser.add_argument("--hardware", action="store_true", help="Allow supervised follower motion")
    parser.add_argument("--camera-only", action="store_true", help="Skip all robot connection and motion")
    parser.add_argument("--robot-port", help="Explicit follower port; never inferred when two arms are present")
    parser.add_argument(
        "--accept-provisional-inspection",
        action="store_true",
        help=(
            "Allow the screenshot-transcribed inspection target only after measured-start "
            "validation and an additional exact confirmation phrase"
        ),
    )
    parser.add_argument(
        "--move-via-home",
        action="store_true",
        help=(
            "From a live state within the guarded HOME approach envelope, move to HOME, "
            "recheck the normal HOME tolerance, then move to INSPECTION"
        ),
    )
    parser.add_argument("--expected-start", choices=("home", "inspection"), default="home")
    parser.add_argument("--poses", type=Path, default=DEFAULT_POSES)
    parser.add_argument("--cameras", type=Path, default=DEFAULT_CAMERAS)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--with-cameras", action="store_true", help="Open cameras after a dry run")
    parser.add_argument("--camera-smoke", action="store_true", help="Capture one pair and exit")
    parser.add_argument(
        "--allow-unvalidated-capture",
        action="store_true",
        help="Diagnostic only: permit saving without a connected follower pose check",
    )
    parser.add_argument("--no-preview", action="store_true")
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _wait_for_frames(streams: dict[str, OpenCVCameraStream], timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(stream.latest(copy_image=False) is not None for stream in streams.values()):
            return
        time.sleep(0.05)
    raise CameraError(f"Timed out waiting for camera frames: {[s.status() for s in streams.values()]}")


def opencv_preview_available(cv2_module: Any) -> bool:
    """Return whether this OpenCV build advertises a desktop GUI backend."""
    try:
        build_info = cv2_module.getBuildInformation()
    except Exception:
        return False
    for line in build_info.splitlines():
        if line.strip().startswith("GUI:"):
            backend = line.split(":", 1)[1].strip().upper()
            return backend not in {"", "NONE", "NO"}
    return False


def _render_previews(
    cv2_module: Any,
    specs: dict[str, Any],
    streams: dict[str, OpenCVCameraStream],
) -> tuple[bool, bool]:
    """Render one preview cycle; return (still_available, quit_requested)."""
    try:
        for name, stream in streams.items():
            frame = stream.latest(copy_image=False)
            if frame is not None:
                cv2_module.imshow(specs[name].preview_name, frame.image)
        quit_requested = cv2_module.waitKey(1) & 0xFF == ord("q")
        return True, quit_requested
    except Exception as exc:
        print(
            "WARNING: OpenCV preview is unavailable; continuing with terminal-only "
            f"capture. Use 'capture' to save frames. Details: {exc}",
            file=sys.stderr,
        )
        return False, False


def _close_preview_windows(cv2_module: Any) -> None:
    try:
        cv2_module.destroyAllWindows()
    except Exception:
        # Headless OpenCV builds expose this function but raise when it is called.
        pass


def run_camera_session(
    config_path: Path,
    output: Path,
    *,
    preview: bool,
    smoke: bool,
    robot_status: dict[str, Any] | None = None,
    robot: Any | None = None,
    pose_config: Any | None = None,
    allow_unvalidated_capture: bool = False,
) -> int:
    import cv2

    specs = load_camera_config(config_path)
    devices = discover_camera_devices(specs)
    streams = {name: OpenCVCameraStream(specs[name], devices[name]) for name in ("wrist", "external")}
    store = PairedCaptureStore(output)
    preview_active = preview and opencv_preview_available(cv2)
    if preview and not preview_active:
        print(
            "WARNING: This OpenCV build has no desktop GUI backend; continuing in "
            "terminal-only mode. Type 'capture' to save paired images.",
            file=sys.stderr,
        )
    try:
        for stream in streams.values():
            stream.start()
        _wait_for_frames(streams)
        print("Camera mapping:")
        _print({name: stream.status() for name, stream in streams.items()})
        def capture_pair() -> dict[str, Any]:
            context: dict[str, Any]
            if robot is not None:
                if pose_config is None:
                    raise RobotSafetyError("Pose configuration is required for validated capture")
                validation = validate_measured_named_pose(robot, pose_config, "inspection")
                context = {"robot_pose_validation": validation.to_dict()}
                if not validation.valid:
                    return {
                        "capture_refused": True,
                        "reason": "robot is outside inspection-pose tolerance",
                        "context": context,
                    }
            else:
                context = {
                    "robot_pose_validation": {
                        "valid": None,
                        "available": False,
                        "reason": "camera-only session has no robot connection",
                    }
                }
                if not allow_unvalidated_capture:
                    return {
                        "capture_refused": True,
                        "reason": (
                            "no connected follower is available for inspection-pose validation; "
                            "use the connected hardware workflow"
                        ),
                        "context": context,
                    }
            return store.save(
                {name: stream.latest() for name, stream in streams.items()},
                context=context,
            )

        if smoke:
            record = capture_pair()
            _print(record)
            return 0 if record.get("complete_pair") else 2

        reader = TerminalCommandReader()
        reader.start()
        print("Commands: capture | status | help | quit")
        while True:
            if preview_active:
                preview_active, quit_requested = _render_previews(cv2, specs, streams)
                if quit_requested:
                    break
            try:
                command = reader.commands.get(timeout=0.02)
            except Exception:
                continue
            if command == "capture":
                _print(capture_pair())
            elif command == "status":
                _print(
                    {
                        "robot": robot_status or {"connected": False},
                        "cameras": {name: stream.status() for name, stream in streams.items()},
                        "next_scene": store.next_scene_index(),
                    }
                )
            elif command == "help":
                print("capture — save paired frames; status — show state; quit — exit; help — commands")
            elif command == "quit":
                break
            else:
                print(f"Unknown command: {command!r}. Type help.")
        return 0
    finally:
        for stream in streams.values():
            stream.stop()
        if preview:
            _close_preview_windows(cv2)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pose_config = load_pose_config(args.poses)
    print(f"Discovered serial ports: {discover_serial_ports()}")
    if args.camera_only:
        return run_camera_session(
            args.cameras,
            args.output,
            preview=not args.no_preview,
            smoke=args.camera_smoke,
            allow_unvalidated_capture=args.allow_unvalidated_capture,
        )

    robot = None
    robot_status: dict[str, Any] = {"connected": False}
    home_approach_plan = None
    try:
        if args.hardware:
            if not args.robot_port:
                raise RobotSafetyError("--hardware requires an explicit --robot-port for the follower")
            robot = connect_follower(args.robot_port, CALIBRATION_DIR)
            if args.move_via_home:
                if args.expected_start != "home":
                    raise RobotSafetyError("--move-via-home requires --expected-start home")
                measured, home_approach_plan = plan_from_measured_state(
                    robot,
                    pose_config,
                    "go_home",
                    "home",
                    allow_home_approach=True,
                )
                inspection_preview = plan_pose_trajectory(
                    pose_config,
                    "go_inspection",
                    pose_config.poses["home"].joints,
                    expected_start_pose="home",
                    state_source="configured_pose_simulation",
                )
                plan = home_approach_plan
                _print(
                    {
                        "staged_motion": "measured_state -> home -> inspection",
                        "home_approach_plan": home_approach_plan.to_dict(include_waypoints=False),
                        "inspection_preview": inspection_preview.to_dict(include_waypoints=False),
                    }
                )
            else:
                measured, plan = plan_from_measured_state(
                    robot, pose_config, "go_inspection", args.expected_start
                )
            robot_status = {"connected": True, "port": args.robot_port, "joint_state": measured}
        else:
            start = pose_config.poses[args.expected_start].joints
            plan = plan_pose_trajectory(
                pose_config,
                "go_inspection",
                start,
                expected_start_pose=args.expected_start,
                state_source="configured_pose_simulation",
            )
        if home_approach_plan is None:
            _print(plan.to_dict(include_waypoints=False))

        if args.hardware:
            if not plan.trajectory_valid:
                raise RobotSafetyError(f"Refusing invalid trajectory: {list(plan.errors)}")
            provisional_move = not plan.target_verified and args.accept_provisional_inspection
            if not plan.execution_authorized and not provisional_move:
                raise RobotSafetyError(
                    "Refusing movement: inspection pose is provisional. After reviewing the measured "
                    "state and trajectory, add --accept-provisional-inspection to use the guarded "
                    "provisional-target workflow."
                )
            if args.move_via_home:
                required_phrase = VIA_HOME_MOVE_PHRASE
                print(
                    "The arm will first move to HOME, verify HOME within the normal tolerance, "
                    "then move to INSPECTION. Ensure both swept paths are clear."
                )
            else:
                required_phrase = PROVISIONAL_MOVE_PHRASE if provisional_move else "MOVE"
            confirmation = input(
                f"Type {required_phrase} to execute the validated inspection trajectory: "
            )
            require_motion_approval(confirmation, required_phrase)
            monitor = EmergencyStopMonitor()
            monitor.start()
            try:
                result = execute_validated_trajectory(
                    robot,
                    plan,
                    pose_config,
                    monitor.stop_event,
                    allow_provisional_target=provisional_move,
                )
                if args.move_via_home and result.completed:
                    _print(
                        {
                            "stage": "home_approach",
                            "motion_completed": result.completed,
                            "reason": result.reason,
                            "final_state": dict(result.final_state),
                            "final_error": dict(result.final_error),
                        }
                    )
                    time.sleep(0.5)
                    _, inspection_plan = plan_from_measured_state(
                        robot, pose_config, "go_inspection", "home"
                    )
                    _print(
                        {
                            "stage": "inspection_after_home_recheck",
                            "plan": inspection_plan.to_dict(include_waypoints=False),
                        }
                    )
                    if not inspection_plan.trajectory_valid:
                        raise RobotSafetyError(
                            "HOME recheck failed; refusing INSPECTION motion: "
                            f"{list(inspection_plan.errors)}"
                        )
                    result = execute_validated_trajectory(
                        robot,
                        inspection_plan,
                        pose_config,
                        monitor.stop_event,
                        allow_provisional_target=provisional_move,
                    )
            except KeyboardInterrupt:
                monitor.stop_event.set()
                raise RobotSafetyError("Ctrl+C received; stopped issuing commands")
            finally:
                monitor.close()
            _print(
                {
                    "motion_completed": result.completed,
                    "reason": result.reason,
                    "final_state": dict(result.final_state),
                    "final_error": dict(result.final_error),
                }
            )
            if not result.completed:
                return 2
            robot_status.update(
                {
                    "held_pose": "inspection",
                    "provisional_target_accepted": provisional_move,
                    "final_joint_state": dict(result.final_state),
                    "final_error": dict(result.final_error),
                }
            )

        if args.with_cameras or args.hardware:
            return run_camera_session(
                args.cameras,
                args.output,
                preview=not args.no_preview,
                smoke=args.camera_smoke,
                robot_status=robot_status,
                robot=robot,
                pose_config=pose_config,
            )
        return 0 if plan.trajectory_valid else 2
    except (RobotSafetyError, CameraError) as exc:
        print(f"SAFETY REFUSAL: {exc}", file=sys.stderr)
        return 2
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())

