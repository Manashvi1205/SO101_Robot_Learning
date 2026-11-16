# Implementation and Experiment Log

This is the canonical cross-chat memory for the repository. The root
`AGENTS.md` requires future agents to read it at the start of a task and update
it before finishing substantive work. Entries preserve completed work,
decisions, verification, failures, safety impact, blockers, and next steps; they
are not a verbatim transcript of conversations.

## 2026-11-15 — Day 1

Implemented:

- Registered the fixed-to-fixed 50k ACT checkpoint.
- Registered the variable-start-to-fixed-target 50k ACT checkpoint.
- Excluded the corrupted 100k checkpoint.
- Preserved fixed-policy `wrist` and `front` observation requirements.
- Preserved variable-policy `wrist` and `phone` observation requirements.
- Added structured planning, validation, ACT inference adapter, dry-run execution,
  JSONL logging, CLI commands, and tests.

Safety status:

- No datasets, checkpoints, calibration files, or LeRobot source files modified.
- No camera opened.
- No serial device opened.
- No motor command sent.
- Live execution remains disabled in code.

Verification results:

- Unit tests: 17 passed.
- Python bytecode compilation: passed.
- Registry asset validation: both 50k checkpoints complete; no missing files.
- Safetensors structural validation: passed for both checkpoints.
- CPU model loading: passed for both checkpoints, 51,597,190 parameters each.
- Full saved preprocessor -> ACT -> saved postprocessor synthetic inference:
  - Fixed policy returned a finite `(6,)` action in approximately 1.8 seconds.
  - Variable policy returned a finite `(6,)` action in approximately 2.8 seconds.
- Variable-policy dry run: passed and wrote `logs/dry_runs.jsonl`.
- Unsupported stacking request: rejected as an empty, invalid plan.

Open items / blockers:

- No OpenAI API request was made; it requires the optional `openai` dependency,
  credentials, and a chosen model. The structured-output path is covered with an
  injected fake client in unit tests.
- Physical success of the variable policy has not been established from a retained
  local rollout.
- Live execution, perception, visual verification, and recovery remain intentionally
  disabled until Day 2 and explicit hardware approval.

## 2026-11-15 — Provisional named poses

Implemented:

- Saved screenshot-transcribed `home` and `inspection` states in
  `config/poses.yaml`, marked provisional and unverified.
- Confirmed the driver uses calibrated degrees for the five arm joints and a
  calibrated `0–100` representation for `gripper.pos`.
- Registered `go_home` and `go_inspection` as joint-trajectory skills, not ACT
  policies.
- Added cubic smoothstep interpolation at 30 Hz with configurable per-joint peak
  velocity limits.
- Derived calibrated joint limits read-only from `so101_follower.json`.
- Added known-start-pose proximity guards, waypoint limit checks, and finite-step
  velocity validation.
- Added `pose-dry-run`; it never imports or constructs robot hardware.

Results:

- Full test suite: 27 passed.
- Stationary home dry run: trajectory valid, 31 waypoints, execution unauthorized
  because the pose is provisional.
- The original home-to-inspection dry run correctly rejected the original
  `wrist_flex.pos=100.97°` transcription because it exceeded the
  calibration-derived maximum `100.7473°`.
- An arbitrary all-zero current state presented as `home` was correctly rejected
  by the start-proximity guard.
- No calibration, checkpoint, or dataset file was modified; no hardware device was
  opened.

### Updated pose transcription

The user supplied replacement HOME and INSPECTION values. The pose configuration
and exact-value tests were updated. Both replacement targets are within the
calibration-derived limits. They remain provisional and physical execution remains
disabled until confirmed from measured follower state.

Validation after replacement:

- Full suite: 27 tests passed.
- HOME → INSPECTION: valid 9.1667-second dry-run trajectory, 276 waypoints
  at 30 Hz, execution unauthorized.
- INSPECTION → HOME: valid 9.1667-second dry-run trajectory, 276 waypoints
  at 30 Hz, execution unauthorized.
- No hardware was accessed.

## 2026-11-15 — Paired perception image collection

Implemented:

- Added logical camera configuration for the wrist camera and external phone
  stream. The device named `Iriun Webcam` is documented as the DroidCam phone
  stream, per the user's clarification.
- Added read-only V4L2 discovery that selects capture-capable nodes and rejects
  metadata-only sibling nodes.
- Added threaded OpenCV streams, live previews, camera status, and terminal
  `capture`, `status`, `help`, and `quit` commands.
- Added paired, non-overwriting PNG capture and JSON metadata with individual
  frame timestamps, device identifiers, and completeness state.
- Added a single collection entry point with hardware-free dry run,
  camera-only, camera-smoke, and explicitly gated supervised hardware modes.
- Added explicit follower-port validation, measured-state trajectory planning,
  exact `MOVE` confirmation, waypoint limit and tracking-error checks, `STOP`
  handling, no automatic retries, and final-position error reporting.

Observed device mapping:

- External phone/DroidCam stream: `/dev/video0`, kernel name `Iriun Webcam`,
  capture capable at 1280×720.
- Wrist camera: `/dev/video3`, kernel name `USB2.0_CAM1`, capture capable at
  1280×720.
- `/dev/video4` is a metadata-only sibling and is not selected.
- `/dev/ttyACM0` and `/dev/ttyACM1` are both present and expose the same USB
  model; the follower cannot be identified safely from those names alone.

Verification results:

- Full unit suite: 35 passed.
- Python bytecode compilation: passed.
- Inspection trajectory dry run from configured HOME: valid, 276 waypoints over
  9.1667 seconds; execution correctly unauthorized while the target is
  provisional.
- Camera-only smoke test: opened `/dev/video0` and `/dev/video3`, acquired both
  frames, and wrote a complete paired scene to a temporary directory.
- Visual review confirmed the wrist view and the external laboratory/arm view
  are assigned to the intended logical cameras.
- Hardware-mode invocation without an explicit port was refused before any
  connection attempt.

Safety status / blockers:

- No robot was connected and no motor command was sent.
- No checkpoint, dataset, or calibration file was modified.
- Physical movement remains blocked because both named poses are marked
  unverified and because the follower serial port has not been authoritatively
  identified.
- Software `STOP` prevents subsequent waypoint commands but is not an electrical
  emergency stop; the operator must retain access to robot power.

### Headless OpenCV correction

The user's first interactive camera-only run confirmed that both streams were
healthy, then exposed a headless OpenCV build where `imshow` and
`destroyAllWindows` raise at runtime. Preview capability is now detected before
the loop, runtime GUI failures fall back to terminal-only capture, and window
cleanup is best-effort. Camera acquisition and paired saving remain available;
`--no-preview` can also request this mode explicitly.

Post-correction validation:

- Full unit suite: 37 passed.
- Python bytecode compilation: passed.
- Real PTY terminal session automatically selected terminal-only mode.
- The `capture` command wrote a complete wrist/external `scene_001` pair and
  metadata; the `quit` command shut down both streams cleanly with exit code 0.
- No robot connection or motor command occurred.

## 2026-11-15 — Guarded inspection-hold workflow

The user identified `/dev/ttyACM1` as the follower and requested the sequence:
validate HOME, move to INSPECTION, hold position, then wait for `capture`.

Implemented a deliberately explicit provisional-target path. It requires all of:

- `--hardware`, explicit `/dev/ttyACM1`, and `--expected-start home`;
- a live measured follower state within every configured HOME tolerance;
- a trajectory with no joint-limit or velocity errors;
- `--accept-provisional-inspection`;
- the exact interactive phrase `MOVE TO PROVISIONAL INSPECTION`.

After successful motion, the robot remains connected while the paired-camera
command loop runs. `quit` disconnects in place and does not command HOME. Default
behavior still rejects provisional targets, and simulated starts cannot use the
provisional-target authorization. Implementation and testing did not connect to
or move the physical arm.

Validation: 40 unit tests passed and Python bytecode compilation passed. The
motion-path tests used a fake follower, including successful provisional motion,
rejection from a simulated start, and exact-phrase enforcement.

### Calibration-identity correction

The first real measured-state check on ACM1 safely refused movement, but reported
values inconsistent with the recent teleoperation display. Investigation found
that the connector forced ID `so101_follower`, selecting the older
`so101_follower.json`, while recent ACM1 teleoperation and recording commands
omit `--robot.id` and select the newer `None.json` calibration. The connector and
pose-limit source now use the latter identity consistently. Neither calibration
file was modified. Measured plans are also now correctly labeled
`hardware_accessed: true`.

Post-correction validation: 41 unit tests passed, compilation passed, and the
HOME-to-INSPECTION hardware-free plan remains valid under the `None.json`
calibration-derived ranges (276 waypoints over 9.1667 seconds). No physical
command was sent during this correction.

### Staged HOME-to-INSPECTION workflow

The corrected live state was close to HOME but had an elbow difference of
7.784 degrees, outside the normal 5-degree guard. Added `--move-via-home` with a
narrow 10-degree/unit measured HOME-approach envelope. The staged workflow:

1. validates and executes measured state to HOME;
2. rereads the follower and reapplies the normal 5-degree/unit guard;
3. validates and executes HOME to provisional INSPECTION;
4. holds INSPECTION while waiting for camera commands.

It retains waypoint limits, velocity limits, tracking checks, `STOP`, no retry,
and explicit provisional-target acceptance. It requires the exact phrase
`MOVE VIA HOME TO PROVISIONAL INSPECTION`; arbitrary states remain rejected.
Implementation tests use a fake follower only.

## 2026-11-15 — Initial colored-box perception baseline

Implemented configurable per-camera HSV segmentation for red, green, and blue
boxes, including workspace ROIs, multiple ranges per color, morphology, contour
geometry filters, center/bounding-box/area output, and a geometry-based quality
score. Added non-overwriting offline evaluation, annotated images, per-scene and
aggregate JSON, and camera contact sheets.

Evaluated exactly scenes 002–032; scene 001 remains present and excluded. The
external camera is the reliable initial view: provisional manual review found
correct red and green localization in all 31 scenes and correct blue localization
in all 29 scenes where blue was visibly exposed. Blue is not exposed in external
scenes 015 and 027, and the detector appropriately returns no blue detection.

The wrist view is weaker because pale box tops, the blue-gray mat, and gripper
occlusion violate simple HSV assumptions. The final manual-review details and
failed-scene lists are stored in
`outputs/perception_eval_v5/manual_review.json`. Earlier v1–v4 output directories
are intentionally preserved as tuning history.

Added capture-pose validation without a second serial connection. Connected
captures reread follower joints immediately before saving, enforce the saved
INSPECTION tolerance, and store measured positions/errors/tolerances/units in new
scene metadata. Invalid captures are refused without moving the robot or writing
an image. Camera-only saving is also refused by default because pose cannot be
validated; its explicit diagnostic override records that validation was
unavailable. Existing scenes were not modified.

## 2026-11-15 — Persistent wrist-camera tuning

Located `camera_tuning/wrist_20261115_165351.json` and integrated it only with
the logical wrist stream. The reusable loader applies automatic white balance
and exposure first, preserves their enabled modes by skipping inactive manual
WB-temperature/exposure-time writes, applies the other saved controls, and
requires exact V4L2 readback. Stream status exposes the application report.

The external camera has no settings reference and does not enter the V4L2
application path. Missing files and readback mismatches fail explicitly. Unit
tests cover loading, auto-mode preservation, mismatch failure, and wrist-only
dual-camera configuration.

Real `/dev/video3` application/readback passed for all applied controls. The
full reopen test was blocked because the user's `tune_wrist_camera.py` process
(PID 51334) still owned `/dev/video3`; it was not interrupted. The documented
reopen command can be run after quitting the tuner. No robot, external-camera
control, dataset, checkpoint, or calibration was changed.

## 2026-11-15 — Updated inspection pose for the next photo batch

Replaced only the provisional INSPECTION target with the user-provided calibrated
values `[-4.66, -41.41, 5.89, 100.18, -7.43, 2.25]` in driver joint order.
HOME, calibration files, camera settings, existing images, datasets, and ACT
checkpoints were not changed. The same dual-camera collection command now targets
this pose, automatically applies the saved wrist V4L2 settings, leaves the
external camera controls unchanged, validates INSPECTION before each capture,
and continues scene numbering without overwriting prior images.

## 2026-11-15 — Colored-box evaluation for scenes 033–061

Processed exactly 29 paired scenes, 033 through 061, offline into
`outputs/perception_eval_scenes_033_061`. Existing images and previous derived
outputs were not overwritten; scene 062 was outside the requested range and was
not processed.

The raw HSV coverage is strong for the external camera (red 29/29, green 28/29,
blue 29/29). Provisional contact-sheet review confirms all external detections
and identifies the partially visible green box in scene 042 as the only miss.

The wrist camera's raw label coverage is not semantic accuracy. Provisional
review finds red correctly localized in 14/29 scenes and green in 18/29. All 26
wrist-blue detections fall on the gripper or nearby background, while the actual
blue box is missed in every scene. These reviewed results are recorded separately
in `manual_review.json`; no HSV thresholds were retuned during this evaluation.

## 2026-11-15 — Wrist detector v2 diagnosis and repair

Diagnosed the failed wrist masks using polygon annotations from tuning scenes
033, 039, 042, and 045. The old ROI discarded most upper-workspace pixels (only
7.4% of annotated blue pixels survived both color and ROI filtering), while the
broad blue HSV mask merged with the mat into a 426,060-pixel component. Small
gripper highlights survived and were incorrectly selected.

Implemented a Lab color-and-shape wrist pipeline with annotated-distribution
bands, bright/pale face modes, the corrected workspace ROI, a fixed gripper
exclusion, rotated rectangles, area/size constraints, rotated rectangularity,
and solidity. The external HSV path is unchanged. Added a headless browser-based
polygon annotation tool and a diagnostic tool that writes measured HSV/Lab
percentiles and old/new masks.

Scenes 046-061 were excluded from pixel calibration. Provisional post-freeze
visual review finds 16/16 correct wrist localizations per color, zero false
positives, and zero misses. Across 033-061 the corresponding result is 29/29 per
color. Dense held-out polygons were not created after evaluation, so these are
visual physical-object localization counts rather than IoU metrics. All 58 unit
tests pass. No robot, camera control, calibration, source image, ACT checkpoint,
or dataset was modified.

## 2026-11-15 — Automatic cross-chat project memory

Added a root-level `AGENTS.md` covering the entire workspace. It establishes
this file as the single canonical project record and instructs future agents to
read it at the beginning of every task and update it after substantive work
without waiting for a user reminder.

The maintenance rules explicitly require completed work, decisions, exact
verification results, failed attempts and causes, unresolved issues, hardware
and safety impact, and useful next steps. Corrections must preserve history and
secrets must never be recorded. The existing chronological history was retained
unchanged; no source code, dataset, checkpoint, calibration, generated output,
camera, or robot was modified or accessed for this documentation change.

## 2026-11-15 — Modular language-to-skill pipeline

Implemented the requested next-stage hierarchical pipeline in the existing
`so101_hierarchical_project` rather than creating a new project.

Completed:

- Expanded the registry to the canonical skills `go_home`, `go_inspection`,
  `detect_blocks`, `identify_color`, `pick_place_fixed`,
  `pick_place_variable`, and `verify_placement`. Every entry now declares its
  inputs, preconditions, backend, availability, success criteria, and error
  handling. The two ACT skills remain restricted to their trained physical
  objects, start distributions, fixed target, cameras, and workspace.
- Added deterministic, OpenAI Responses structured-output, and local
  OpenAI-compatible planners. Planner JSON is semantic data only; the schema
  has no motor-angle or arbitrary-command fields. Unsupported stacking retains
  a meaningful inspection/detection/selection/placement proposal, then fails
  local validation for the untrained colored object and `on_blue_box` target.
- Extended validation for skill/backend existence, static asset contracts,
  object/target/start constraints, logical or saved observation camera keys,
  availability, and runtime preconditions. Dry runs surface unconfirmed
  preconditions as warnings; hardware mode treats them as errors and all motion
  and ACT entries remain hardware locked.
- Added a backend-injected sequential task manager with structured observation
  flow, optional single retry, JSONL logging, stop-on-failure behavior, and
  `replanning_required` results. The existing dry-run executor remains
  hardware-free.
- Added `vision.py`, which wraps the frozen color detector for all-color
  detection, color selection, and conditional image-region placement checks.
  Results explicitly state `coordinate_space=image_pixels` and that robot-frame
  coordinates are unavailable. Verification is not automatically claimed for
  the ACT training boxes because they are not established colored-detector
  classes and no fixed-target image ROI is calibrated.
- Kept both 50k ACT wrappers and added saved-config contract checks. Model load
  still uses the checkpoint's saved preprocessor and postprocessor. The
  variable policy retains the exact `observation.images.phone` key. No 100k
  checkpoint is referenced.
- Updated the CLI and README with inventory, hosted/local planner, dry-run,
  checkpoint-smoke, and offline `vision-detect` commands. Added instruction,
  vision, validation, execution-manager, local-planner, and ACT-contract tests.

Verification:

- Full suite: 75 tests passed with
  `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ... python -m pytest -q`.
  Initial pytest startup without plugin autoload disabled failed because a ROS
  `launch_testing` plugin imports the absent `lark` package; project tests were
  unaffected and passed with third-party plugin autoload disabled.
- Python bytecode compilation passed for `src` and `tests`.
- Registry inventory reported all seven skills and no asset problems.
- Both 50k checkpoints passed safetensors inspection, saved input/action/chunk
  contract validation, CPU model loading (51,597,190 parameters each), and full
  saved-preprocessor -> ACT -> saved-postprocessor synthetic inference. Each
  returned a finite `(6,)` action.
- Required instruction tests passed for red identification, blue
  identification, all-box identification, inspection motion planning, and the
  original-training-box fixed-target plan. Stacking was rejected with exact
  `unsupported_object` and `unsupported_target` issues.
- Offline structured detection on wrist `scene_061.png` returned red, green,
  and blue with no missing colors. This reused an existing image; it did not
  open a camera.

Safety and remaining work:

- No camera device, serial device, robot connection, or motor command was used.
  No API request was made; hosted and local LLM paths were tested with injected
  clients. No datasets, calibration files, source images, or checkpoints were
  modified.
- Live hierarchical dispatch remains disabled. Supervised hardware execution
  still requires explicit user authorization, positive follower-port identity,
  physically verified named poses, live camera/precondition checks, a guarded
  ACT action-dispatch loop with rate/limit/tracking/emergency-stop controls, and
  supervised rollouts of each policy with its exact trained object/workspace.
- Automatic visual success for the ACT tasks additionally requires a detector
  for the actual training boxes and a calibrated fixed-target image region (or
  a validated camera-to-workspace mapping). Arbitrary-target placement and
  stacking require new demonstrations/policies and validation; pixel centers
  must not be treated as robot coordinates.

## 2026-11-15 — ACT target highlighting and appearance-augmentation feasibility

Audited the live `so101_variable_box_40episodes` dataset and variable-box ACT
checkpoint before designing the no-new-demonstration experiment. The source
dataset has 40 episodes, 25,062 finite state/action rows, contiguous episode and
frame indices, exact 30 Hz timestamps, and 25,062 H.264 frames in each of the
`wrist` and `phone` streams. LeRobot 0.6.1 decoded samples at dataset boundaries
with the saved 3x720x1280 camera contract. Dataset action/state means and standard
deviations agree with `meta/stats.json` to within 2.19e-4. Commands best align
with measured states four frames later (about 133 ms), so all experimental arms
must preserve the existing temporal labels.

The 50k `pretrained_model` has all inference/warm-start files, valid safetensors,
51,597,190 parameters, and SHA-256
`f7df348455196b93f175ad7611945a9f5369c92fc9957c359f7a5bad55b14103` for the
model weights. Saved-preprocessor ACT inference on real dataset frame 0 returned
a finite six-element action on CPU. The parent 050000 checkpoint has no optimizer,
scheduler, RNG, or step state, so future work must use `--policy.path` warm-start,
not optimizer resume.

Implemented a derived-image-only prototype in `target_augmentation.py` with a
fixed cyan interior cue plus white/black contour and Lab-based object recoloring
that retains luminance geometry and changes no pixel outside the visible mask.
Added two coarse, explicitly non-training polygons for episode 15 and generated
phone/wrist original-highlighted-augmented comparison panels plus a manifest and
audit under `outputs/target_augmentation_prototype`. Added
`TARGET_GENERALIZATION_EXPERIMENT.md` covering reviewed video-mask propagation,
occlusion and label risks, temporally coherent two-view augmentation, immutable
derived LeRobot datasets, warm-start/preprocessor checks, and evaluation.

The controlled design retains the untouched A0 baseline and adds a matched-step
original-data A1 optimization control, B highlighting, and C highlighting plus
appearance augmentation; augmentation-only is recommended as the missing 2x2
cell. Evaluation is separated into original-carton retention, zero-shot unseen
objects (geometry-matched appearance shift separate from geometry/dynamics
shift), and a later fixed-budget additional-real-demonstration learning curve.
Oracle/reviewed-mask policy trials must be reported separately from online-mask
end-to-end trials so annotation does not get mistaken for generalization.

Verification: the three new augmentation unit tests pass; the full suite passes
78 tests; Python bytecode compilation passes. No training or full derived dataset
was started. Neither source dataset nor checkpoint was modified. No camera or
serial device was opened, no robot was connected, and no motor command was sent.

Next step: annotate and review a small multi-episode pilot spanning pre-grasp,
occlusion, transport, and placement; build short derived video clips; then run
only the one-batch/one-step save-reload compatibility smoke test before choosing
an expensive training budget.

## 2026-11-15 — Three-episode highlighted ACT training pilot

Selected source episodes 3, 20, and 30 after reviewing all 40 start frames; they
represent right/far, center/near, and left/near carton starts. Added a compact
manual-polygon workflow covering approach, grasp, occlusion, transport,
placement, and release in both cameras. Episode 3 wrist has one extra entry
anchor because the carton is initially off-screen and later becomes visible.

Implemented bidirectional Lucas–Kanade/RANSAC affine mask propagation with
feature consistency, area, temporal-distance, low-confidence, discontinuity,
partial-visibility, and full-occlusion flags. Manual anchors reset tracks. Every
propagated mask remains `review_status: pending`; confidence never automatically
accepts a label. Review artifacts include canonical JSON, keyframe overlays, a
self-contained HTML correction guide, six stream contact sheets, the full
per-frame propagation report, and a source/encoded comparison sheet.

The final report contains 37 manually reviewed camera keyframes and 3,645
pending propagated camera-frames. It flags 444 low-confidence frames, one
discontinuity, and 212 empty masks. Phone masks are stronger. Wrist masks remain
coarse around close gripper occlusions and require correction before full
training. A review caught two issues during implementation: one-directional
phone drift and backward wrist propagation onto background. Propagation was
changed to use the nearer of two anchors, pre-entry wrist masks were made empty,
and an episode-3 wrist visibility anchor was added. Superseded derived/smoke
trees were moved to the desktop trash and are recoverable there.

Created the immutable-source derived dataset
`so101_variable_box_3ep_highlight_pilot`: three episodes, lengths 659/588/594,
1,841 frames in each 1280x720 30 fps AV1 camera stream. For source episodes
3/20/30, actions, states, timestamps, and frame indices are exactly equal after
concatenation. Derived episode indices are contiguous 0/1/2; provenance maps
them back to 3/20/30. LeRobot 0.6.1 decoded first, boundary, and last frames.

Ran exactly one CPU training step by warm-starting the original 50k checkpoint
with `--policy.path`, batch size 1, and no image transforms. Forward, backward,
and AdamW update completed with loss 0.390, L1 0.389, KL 0.000, and finite
gradient norm 39.779. Step `000001` saved model/processors plus optimizer, RNG,
and step state. Reloaded saved model (51,597,190 parameters) and saved processors
returned a finite `(1, 6)` action. Original model SHA-256 remains
`f7df348455196b93f175ad7611945a9f5369c92fc9957c359f7a5bad55b14103`; smoke
model SHA-256 is
`963179b7aa26f964c006bb297c45d564e8a29538517b57bce24a4bb4ace803eb`.

LeRobot warm-start kept ImageNet visual normalization exactly but recomputed
state/action mean/std from the three-episode subset (maximum differences from
the full checkpoint stats were about 6.15 mean and 5.04 std). This is acceptable
for compatibility only. A future 40-episode visual derivative has unchanged
state/action arrays and will reproduce the original full-dataset statistics.

Verification: 81 project tests passed and bytecode compilation passed. No full
training began. The original 40-episode dataset and 50k checkpoint were not
modified. No camera or serial device was opened, no robot was connected, and no
motor command was sent. Exact commands and limitations are in
`CARTON_HIGHLIGHT_PILOT.md`; machine-readable results are in
`outputs/carton_pilot/pilot_audit.json`.

Next step: correct all pending/flagged wrist intervals, then scale annotation
and derivation to all 40 episodes. Re-run only a one-step A1/B/C compatibility
check with matched preprocessing before choosing any full training budget.

## 2026-11-16 — Corrected three-episode A/B datasets prepared

Continued only the existing source episodes 3, 20, and 30; none of the other 37
episodes were processed. Added seven review anchors: terminal phone/wrist
anchors for all three episodes and episode-20 wrist frame 120. The first encoded
preview revealed that the three terminal wrist polygons were too broad, so they
were manually tightened to the visible carton and propagation/B encoding were
rerun. The superseded B build was moved to the desktop trash and is recoverable.

The corrected propagation report has 44 manual anchors and 3,638 propagated
camera-frames that remain explicitly pending review. Low-confidence flags fell
from 444 to 210, the prior discontinuity was eliminated, and 212 empty wrist
masks remain for pre-entry/off-screen or explicit occlusion intervals. Exact
remaining frame spans are in `outputs/carton_pilot/MASK_REVIEW_REPORT.md`; the
six camera review sheets, HTML guide, and `ab_dataset_preview.jpg` cover both
cameras. No propagated mask was automatically accepted.

Created two matched derived datasets:
`so101_variable_box_3ep_original_control` (A, unchanged imagery after video
re-encoding) and `so101_variable_box_3ep_highlight_corrected` (B). Each has
three episodes, lengths 659/588/594, 1,841 rows, and two AV1 yuv420p 1280x720
30 fps streams with 1,841 frames. For both A and B, actions, states, timestamps,
and local frame indices are exactly equal to the selected source rows; first,
episode-boundary, and final samples decode through LeRobot.

To preserve the original ACT normalization for the matched comparison, each
active `meta/stats.json` is a byte-identical copy of the original 40-episode
stats (SHA-256 `d5d3760857db874e55ad5e4973fa8de592601ef5aeb81713659a990129420a4b`).
The computed subset stats remain as `stats.empirical_3episode.json`, and
`normalization_provenance.json` documents the choice. A training-path processor
check, including LeRobot's in-memory ImageNet camera override, produced finite
expected shapes and matched the saved 50k checkpoint normalizer exactly for A
and B.

Extended `target_annotation_pilot.py` minimally with explicit original/highlight
build modes and an A/B preview command. Updated `CARTON_HIGHLIGHT_PILOT.md` with
the workflow and matched future commands. Machine-readable results are in
`outputs/carton_pilot/ab_dataset_audit.json`. Verification: 81 tests passed,
bytecode compilation passed, and the source dataset/checkpoint hashes remain
`79dc2c84...` (dataset info), `d5d37608...` (dataset stats), and
`f7df3484...` (model). No training was started in this task. No camera, robot,
or serial hardware was accessed and no motor command was sent.

Next step: manually review/correct the remaining 210 low-confidence frames,
especially wrist occlusion/late-release intervals, or explicitly accept that
annotation noise as a pilot limitation before running the matched A/B commands.

## 2026-11-16 — Nine new multibox demonstrations audited and A/B review datasets built

Audited exactly the nine new local roots `arr1/arr2/arr3 × red/blue/green` under
`local/`; no older dataset episodes were processed. Every root has one episode,
finite six-dimensional action/state arrays, the correct single-color task text,
contiguous local frame/index arrays, exact 30 Hz timestamps, and matching wrist
and phone AV1 streams at 1280×720/30 fps. Episode lengths are
541/518/496/513/518/524/519/627/507 (4,763 rows total). Software decoding
counted exactly one video frame per row in both cameras. The full audit is
`outputs/multibox_9ep/source_audit.json`.

Created matched derived roots `MULTIBOX_ORIGINAL` and `MULTIBOX_HIGHLIGHT`, each
with 9 episodes, 4,763 rows, and 4,763 frames in each camera stream. A uses
unchanged source imagery followed by AV1 re-encoding. B uses the prior pilot's
30% cyan interior plus white/black contour convention and changes camera pixels
only. Source actions, states, timestamps, local frame indices, and episode
boundaries are exactly equal in both derivatives. `derived_audit.json` records
the equality and video checks. Both active `meta/stats.json` files are
byte-identical copies of the original 40-episode statistics (SHA-256
`d5d3760857db874e55ad5e4973fa8de592601ef5aeb81713659a990129420a4b`); empirical
9-episode statistics are retained beside them. This preserves the established
50k warm-start normalization where compatible.

Generated 18 dense source-mask review sheets, nine encoded A/B episode sheets,
and `all_9_A_vs_B_montage.jpg`. Mask generation used the frozen detector on
7,528 of 9,526 camera-frames, a task-color fallback on 1,553, and an empty mask
on 445. Review found confirmed wrist failures: most importantly arr1-red
approximately frames 0–168 and arr1-blue spans 73–146, 166–184, 210–237, and
460–517 contain non-target, partial, or missing cues. Several close placed-target
terminal spans also need correction. Exact flagged ranges and review guidance
are in `outputs/multibox_9ep/mask_generation_report.json` and
`MASK_REVIEW_REPORT.md`. A sampled Lab-fallback diagnostic improved pale-box
localization but retained false positives and was not used to silently replace
the encoded B data.

Status/decision: `MULTIBOX_ORIGINAL` is structurally ready; the current
`MULTIBOX_HIGHLIGHT` is a review candidate and must not be trained until the
reported wrist masks are manually corrected and B is rebuilt. No training was
started. Project tests pass (81 tests), and bytecode compilation of the new
multibox tool passed. No camera or serial device was opened, no robot was
connected, and no motor command was sent. The nine source datasets, original
40-episode dataset, 50k checkpoint, and calibrations were not modified.

Next step: add reviewed wrist anchors at phase/visibility transitions, propagate
between them, review fallback/empty boundaries, and rebuild only B; then rerun
the exact equality/decode audit before considering a matched training run.

## 2026-11-16 — Multibox wrist masks conservatively corrected and B regenerated

Continued from the nine-episode A/B audit without changing phone-only behavior:
the target cue remains available in both cameras whenever the selected object is
clearly visible. Repair scope was restricted to wrist intervals named by the
prior review plus two problems found by a seeded sanity sample. Accepted phone
masks were preserved byte-for-byte; the canonical phone-video SHA-256 before and
after correction is
`3840c939aa7ab83f9654b15d7ccb54aee52b0e5de4e8e3ec600436591720dff7`.

Applied the clarified conservative rule: highlight clearly visible approach
targets, but emit an explicit intentional-empty mask for close, clipped,
partially visible, grasped, gripper-occluded, or off-screen frames. Corrected
203 clearly visible wrist frames: arr1-red 40–168 and arr1-blue 73–146. They use
8 in-range manually reviewed anchors and 195 bidirectionally propagated masks.
Marked 642 wrist frames intentional empty. Exact ranges and reasons are in
`outputs/multibox_9ep/wrist_corrections.json` and the updated
`MASK_REVIEW_REPORT.md`.

A seeded one-frame-per-episode sanity review (seed `20261116`) found a false cue
in arr2-blue frame 444, where the blue target was absent, and an unsuitable
partial cue in the extremely close arr3-blue frame 335. Immediate-neighborhood
review expanded these to intentional-empty intervals arr2-blue 425–456 and
arr3-blue 300–456. It also explicitly labeled already-unhighlighted close frames
arr3-red 264 and arr3-green 300. The same sample was rerun after rebuilding and
showed no remaining wrong-object cue. No remaining frame in the repaired scope
requires user intervention; accepted masks outside that scope were deliberately
not exhaustively rereviewed.

Regenerated only `MULTIBOX_HIGHLIGHT` and promoted the corrected build after
verification. The prior B remains recoverably at
`MULTIBOX_HIGHLIGHT_pre_wrist_correction_20261116`. The superseded first
temporary correction build was moved to the desktop trash and is recoverable.
Updated artifacts include `mask_generation_report.json`, nine encoded previews,
`all_9_A_vs_B_montage.jpg`, `random_sanity_sample.corrected_v2.jpg`, and
`derived_audit.json`.

Final verification: canonical B has 9 episodes, 4,763 rows, and exactly 4,763
AV1 1280×720/30 fps frames per camera. Actions, states, timestamps, local frame
indices, and episode boundaries are exactly source-equal. Active stats remain a
byte-identical copy of the original 40-episode normalizer (SHA-256
`d5d3760857db874e55ad5e4973fa8de592601ef5aeb81713659a990129420a4b`). The
original 50k model hash remains
`f7df348455196b93f175ad7611945a9f5369c92fc9957c359f7a5bad55b14103`.
Python compilation passed and the project suite passed 81 tests.

No training was started. No source dataset, 40-episode dataset, checkpoint, or
calibration was modified. No camera or serial device was opened, no robot was
connected, and no motor command was sent.

## 2026-11-16 — GitHub source-code publication handoff

Prepared a user-run Git workflow for publishing the important hierarchical
project source into `Manashvi1205/SO101_Robot_Learning` without having the agent
commit or push. Verified read-only that the remote repository exists, its
default branch is `main`, and its existing commit author is `Manashvi Garg
<170458704+Manashvi1205@users.noreply.github.com>`. The local LeRobot cache root
and `so101_hierarchical_project` are not currently Git repositories.

The publication scope is source, tests, configuration, annotations, the wrist
camera settings JSON, documentation, and this project log. The proposed
project-local `.gitignore` excludes raw/derived data, generated outputs, logs,
checkpoints, camera screenshots, caches, environments, and secrets. The local
project currently contains about 115 MB under `data/` and 1.4 GB under
`outputs/`, including model/optimizer files over GitHub's normal file-size
limit, so those artifacts must not be added by the source-code commit. No remote
write, commit, push, robot access, camera access, or motor command occurred.

### Publication command recovery

The user cloned the remote into `/home/manashvi/SO101_Robot_Learning` and staged
the intended source-only project tree. A visually mangled terminal paste did not
damage the result: all intended source, test, config, annotation, documentation,
camera-settings JSON, and tuning-script files are present, and the project-local
`.gitignore` correctly excludes generated/heavy content. No new file over 50 MB
was staged. The initial test command used the base Python 3.14 environment, which
does not have pytest, so it stopped with `No module named pytest`; this was an
environment-selection issue, not a test failure. Re-running from the existing
`lerobot` Conda environment passed all 81 tests in 1.66 seconds. Git's blank-line
messages were whitespace warnings only, and the user had entered the diff pager
before commit/push commands could run. At this point no commit or remote push had
occurred. No robot, serial device, camera, or motor command was accessed.

