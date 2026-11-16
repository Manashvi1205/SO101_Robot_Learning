# SO-101 Hierarchical Manipulation Prototype

This project provides a hardware-safe language-to-skill pipeline around two
verified LeRobot 0.6.1 ACT checkpoints. Robot access is never enabled by
default.

## Implemented

- Explicit registry for named motion, perception, ACT, and verification skills.
- Structured `TaskPlan` schema and capability-aware plan validation.
- Deterministic offline planner for reproducible dry runs.
- OpenAI Responses structured-output planner and a local OpenAI-compatible
  endpoint backend (for example vLLM or Ollama when configured compatibly).
- Hardware-free ACT checkpoint inspection and callable inference adapter.
- Sequential task manager with observation flow, stop/replan failure handling,
  dry-run simulation, and optional JSONL execution logging.
- Live execution hard-disabled pending explicit approval and Day 2 supervision.
- Provisional named `home` and `inspection` pose configuration.
- Calibration-derived joint-limit checks and cubic smoothstep interpolation.
- Per-joint velocity validation and known-start-pose guards.
- Paired wrist/external-camera capture with previews, timestamps, metadata, and
  non-overwriting scene indices.
- Explicitly gated follower connection and monitored inspection-pose trajectory.
- Structured color detections and selections with explicit image-pixel coordinates.
- Unit and integration tests for the required language instructions and safety boundaries.

The ACT policies are not language-conditioned. Language selects a registered
skill; it does not change the learned object, start distribution, or target.
Neither skill supports unseen-target placement or stacking.

The registered skills are `go_home`, `go_inspection`, `detect_blocks`,
`identify_color`, `pick_place_fixed`, `pick_place_variable`, and
`verify_placement`. Each registry entry declares required inputs, preconditions,
its bound backend, availability, success criteria, and failure behavior.

The pipeline boundary is:

```text
instruction -> structured semantic plan -> registry validation -> task manager
            -> bound skill backend -> observations/visual check -> continue or replan
```

Planner output is data only. It cannot contain motor angles, create backends, or
bypass physical-object, target, camera, or precondition checks.

## Environment

Use the existing LeRobot environment:

```bash
conda activate lerobot
cd /home/manashvi/.cache/huggingface/lerobot/so101_hierarchical_project
python -m pip install -e .
```

The editable install changes only the Conda environment's package metadata. To
run without installing, set `PYTHONPATH=src` as shown below.

## Hardware-free commands

```bash
cd /home/manashvi/.cache/huggingface/lerobot/so101_hierarchical_project

PYTHONPATH=src python -m so101_hierarchical.cli inventory

PYTHONPATH=src python -m so101_hierarchical.cli plan \
  "Identify the blue box."

PYTHONPATH=src python -m so101_hierarchical.cli dry-run \
  "Pick up the box from a variable position and place it at the fixed target" \
  --log logs/dry_runs.jsonl

PYTHONPATH=src python -m so101_hierarchical.cli checkpoint-smoke --predict-synthetic

PYTHONPATH=src python -m so101_hierarchical.cli pose-dry-run \
  go_home --from-pose home

PYTHONPATH=src python -m so101_hierarchical.cli pose-dry-run \
  go_inspection --from-pose home --summary-only

PYTHONPATH=src python -m so101_hierarchical.collect_images \
  --expected-start home

PYTHONPATH=src python -m so101_hierarchical.collect_images \
  --camera-only --camera-smoke --no-preview

PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Use the hosted API planner (credentials are read by the OpenAI client):

```bash
PYTHONPATH=src python -m so101_hierarchical.cli plan \
  --planner openai --model <structured-output-capable-model> \
  "Go to inspection position and identify the blue box."
```

Use a local OpenAI-compatible server:

```bash
PYTHONPATH=src python -m so101_hierarchical.cli plan \
  --planner local --base-url http://localhost:11434/v1 --model <local-model> \
  "Identify all three colored boxes."
```

Run the frozen detector on an existing frame without opening a camera:

```bash
PYTHONPATH=src python -m so101_hierarchical.cli vision-detect \
  data/vision/wrist/scene_061.png --camera wrist --color blue
```

`Stack the red box on the blue box` intentionally produces a semantic detection,
selection, and placement proposal, then fails validation because the ACT policy
does not support colored boxes, arbitrary targets, or stacking.

`verify_placement` is available only when a caller supplies a calibrated
image-space target region and a colored object. It is not automatically appended
to the two ACT plans: their physical training boxes are not established colored
detector classes, and no fixed-target image ROI has yet been calibrated. This
prevents a visual heuristic from being reported as manipulation success.

## Paired image collection

The logical camera mapping is stored in `config/cameras.yaml`:

- `wrist`: capture-capable `USB2.0_CAM1` node (currently `/dev/video3`).
- `external`: the phone/DroidCam stream exposed by the V4L2 loopback device
  named `Iriun Webcam` (currently `/dev/video0`).

The wrist entry references the saved tuning file
`camera_tuning/wrist_20261115_165351.json`. Before opening the wrist stream, the
pipeline applies its V4L2 controls and verifies them by readback. Saved automatic
white balance (`white_balance_automatic=1`) and automatic exposure
(`auto_exposure=3`) are applied first; their inactive manual temperature and
exposure-time controls are observed but not overwritten. The external camera has
no settings-file entry and its controls are not changed.

After closing the interactive tuner or any other process using `/dev/video3`,
test control persistence across a camera open/read/close cycle with:

```bash
PYTHONPATH=src python -m so101_hierarchical.camera_settings \
  --settings camera_tuning/wrist_20261115_165351.json \
  --device /dev/video3 \
  --reopen-test
```

The command applies and verifies controls, opens and reads one wrist frame,
closes the device, then verifies the controls again without reapplying them. A
missing JSON or readback mismatch is a hard error; no fallback values are
invented.

Device paths are discovered from their kernel names each run; the metadata-only
siblings such as `/dev/video4` are rejected. Start a live camera-only session,
which never constructs or connects a robot, with:

```bash
PYTHONPATH=src python -m so101_hierarchical.collect_images --camera-only
```

The terminal commands are `capture`, `status`, `help`, and `quit`. Camera-only
preview and status remain available, but saving is refused by default because no
follower pose can be measured. `--allow-unvalidated-capture` is an explicit
diagnostic override and records that validation was unavailable. A validated capture
creates a matched scene in `data/vision/wrist`, `data/vision/external`, and
`data/vision/metadata`. Existing files are never overwritten. If OpenCV was
installed without a desktop GUI backend, the utility automatically continues in
terminal-only mode; `--no-preview` selects that mode explicitly.

The follower is currently identified as `/dev/ttyACM1`. Because Linux device
numbers can change after reconnecting USB devices, the port remains an explicit
argument. To read the measured state, validate that it is near HOME, move to the
provisional INSPECTION target, hold there, and then enter the camera command loop:

```bash
PYTHONPATH=src python -m so101_hierarchical.collect_images \
  --hardware --robot-port /dev/ttyACM1 --expected-start home \
  --accept-provisional-inspection
```

Hardware mode first prints the measured-state trajectory and does not move if
the arm is not within the configured HOME tolerance. Because the inspection
target is still provisional, it requires the exact terminal confirmation
`MOVE TO PROVISIONAL INSPECTION`. After a successful move the follower remains
connected at INSPECTION while the terminal accepts `capture`; `quit` disconnects
without automatically returning home. `STOP` or Ctrl+C aborts further waypoint
commands. It never retries automatically.

If the arm is close to HOME but just outside the normal 5-degree start guard,
use the staged workflow:

```bash
PYTHONPATH=src python -m so101_hierarchical.collect_images \
  --hardware --robot-port /dev/ttyACM1 --expected-start home \
  --accept-provisional-inspection --move-via-home
```

This permits only a live measured state within 10 degrees/units of HOME. It
first executes `go_home`, then reads the follower again and reapplies the normal
5-degree/unit HOME guard before planning `go_inspection`. It requires the exact
phrase `MOVE VIA HOME TO PROVISIONAL INSPECTION`. A state outside the narrow
HOME-approach envelope remains rejected.

To use an actual LLM planner, install the optional OpenAI dependency. Hosted
requests use `--planner openai`; local compatible endpoints use `--planner local`
with `--base-url`. In both cases, returned JSON is parsed into `TaskPlan` and
validated locally before execution.

## Safety boundary

The hierarchical ACT executor remains dry-run only: `execute_live()` always
raises `LiveExecutionDisabled`. Checkpoint inference returns calibrated
six-dimensional absolute joint-position targets but does not send them anywhere.
The separate image-collection utility contains a narrowly scoped, supervised
named-pose motion path; it cannot dispatch ACT outputs.

The five arm joint positions are in the driver's calibrated degree
representation. `gripper.pos` is different: the SO follower driver represents it
in its calibrated `0–100` range. Both named poses remain provisional. The updated
targets are inside the current calibration-derived limits, but physical execution
is refused until the target is verified from measured follower state. An explicit
follower port is also mandatory because `/dev/ttyACM0` and `/dev/ttyACM1` expose
the same USB model and cannot be distinguished safely from names alone.

For the current ACM1 follower, the project deliberately uses robot ID `None` and
the existing `calibration/robots/so_follower/None.json`. This matches the recent
teleoperation and dataset-recording commands, which omitted `--robot.id`. The
older `so101_follower.json` belongs to a different command/calibration identity
and must not be mixed with the screenshot-transcribed poses.

## Offline colored-box evaluation

Camera-specific HSV thresholds and workspace masks are stored in
`config/color_detection.yaml`. Evaluate only the accepted scenes 002–032 into a
new derived-output directory with:

```bash
PYTHONPATH=src python -m so101_hierarchical.color_detection \
  --input data/vision \
  --output outputs/perception_eval_v5_reproduced \
  --config config/color_detection.yaml \
  --start-scene 2 --end-scene 32
```

The evaluator writes per-scene JSON, aggregate JSON, annotated images, and one
review sheet per camera. It refuses to overwrite a non-empty output directory.
The checked baseline is under `outputs/perception_eval_v5`; its manual review is
explicitly provisional because no pixel or box ground truth was collected.

In a connected capture session, every new `capture` command now reuses the
existing follower connection to read measured joints and compare them with
INSPECTION. Out-of-tolerance captures are refused without consuming a scene
number. Successful new metadata records measured positions, per-joint errors,
tolerances, and units. Camera-only sessions record that pose validation was
unavailable; they do not open a second robot connection.

