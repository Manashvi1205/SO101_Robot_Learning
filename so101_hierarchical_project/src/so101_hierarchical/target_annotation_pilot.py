"""Three-episode carton-mask pilot and highlighted LeRobot dataset builder.

The propagation confidence is a review aid, never an assertion of correctness.
Manual keyframes are authoritative only for the annotated frame; propagated
frames are always emitted with ``review_status=pending``.
"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .target_augmentation import highlight_target, polygon_mask


CAMERAS = ("phone", "wrist")
DEFAULT_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != "1.0":
        raise ValueError("Unsupported pilot annotation schema")
    if sorted(spec["selected_source_episodes"]) != sorted(int(k) for k in spec["episodes"]):
        raise ValueError("selected_source_episodes and episode records disagree")
    return spec


def episode_metadata(dataset_root: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    return pq.read_table(path).to_pylist()


def decode_episode_frames(
    dataset_root: Path, episode: dict[str, Any], camera: str
) -> tuple[list[np.ndarray], str]:
    file_index = episode[f"videos/observation.images.{camera}/file_index"]
    video = dataset_root / f"videos/observation.images.{camera}/chunk-000/file-{file_index:03d}.mp4"
    start_s = float(episode[f"videos/observation.images.{camera}/from_timestamp"])
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    frames: list[np.ndarray] = []
    for frame_index in range(int(episode["length"])):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Decode failed: {video}, episode frame {frame_index}")
        frames.append(frame)
    capture.release()
    return frames, str(video.resolve())


def anchor_mask(anchor: dict[str, Any], shape: tuple[int, int, int] | np.ndarray) -> np.ndarray:
    image_shape = shape.shape if isinstance(shape, np.ndarray) else shape
    if anchor["visibility"] == "fully_occluded":
        return np.zeros(image_shape[:2], dtype=np.uint8)
    polygons = anchor.get("polygons_xy", [])
    if not polygons:
        raise ValueError(f"Visible anchor {anchor['phase']} has no polygon")
    result = np.zeros(image_shape[:2], dtype=np.uint8)
    for points in polygons:
        result = cv2.bitwise_or(result, polygon_mask(image_shape, [tuple(p) for p in points]))
    for points in anchor.get("exclude_polygons_xy", []):
        result[polygon_mask(image_shape, [tuple(p) for p in points]) > 0] = 0
    return result


def _propagate_one(
    previous_gray: np.ndarray, current_gray: np.ndarray, previous_mask: np.ndarray
) -> tuple[np.ndarray, float, list[str]]:
    if not np.any(previous_mask):
        return previous_mask.copy(), 0.0, ["occluded_or_missing_anchor"]

    eroded = cv2.erode(previous_mask, np.ones((3, 3), np.uint8))
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=160,
        qualityLevel=0.01,
        minDistance=4,
        mask=eroded,
        blockSize=5,
    )
    flags: list[str] = []
    if points is None or len(points) < 6:
        return previous_mask.copy(), 0.05, ["insufficient_object_features", "discontinuity"]

    nxt, status_f, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, points, None, winSize=(31, 31), maxLevel=3
    )
    back, status_b, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, nxt, None, winSize=(31, 31), maxLevel=3
    )
    if nxt is None or back is None:
        return previous_mask.copy(), 0.0, ["optical_flow_failure", "discontinuity"]
    forward = points.reshape(-1, 2)
    moved = nxt.reshape(-1, 2)
    returned = back.reshape(-1, 2)
    fb = np.linalg.norm(forward - returned, axis=1)
    good = (status_f.reshape(-1) > 0) & (status_b.reshape(-1) > 0) & (fb < 2.5)
    if int(good.sum()) < 5:
        return previous_mask.copy(), 0.05, ["insufficient_consistent_flow", "discontinuity"]

    matrix, inliers = cv2.estimateAffinePartial2D(
        forward[good], moved[good], method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if matrix is None:
        return previous_mask.copy(), 0.05, ["affine_fit_failure", "discontinuity"]
    height, width = previous_mask.shape
    propagated = cv2.warpAffine(
        previous_mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0
    )
    old_area = max(1, int(np.count_nonzero(previous_mask)))
    new_area = int(np.count_nonzero(propagated))
    ratio = new_area / old_area
    if ratio < 0.72 or ratio > 1.38:
        flags.extend(["area_jump", "discontinuity"])
    median_fb = float(np.median(fb[good]))
    inlier_fraction = float(np.mean(inliers)) if inliers is not None else 0.0
    feature_score = min(1.0, int(good.sum()) / 30.0)
    area_score = math.exp(-abs(math.log(max(ratio, 1e-6))))
    confidence = feature_score * area_score * inlier_fraction * math.exp(-median_fb / 2.5)
    if confidence < 0.55:
        flags.append("low_confidence")
    return propagated, float(np.clip(confidence, 0.0, 1.0)), sorted(set(flags))


def _mask_polygon(mask: np.ndarray, full_width: int, full_height: int) -> list[list[int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(1.0, 0.008 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    sx, sy = full_width / mask.shape[1], full_height / mask.shape[0]
    return [[int(round(x * sx)), int(round(y * sy))] for x, y in approx]


def propagate_camera(
    frames: list[np.ndarray], anchors: list[dict[str, Any]], *, half_size: tuple[int, int] = (640, 360)
) -> list[dict[str, Any]]:
    height, width = frames[0].shape[:2]
    grays = [cv2.cvtColor(cv2.resize(frame, half_size), cv2.COLOR_BGR2GRAY) for frame in frames]
    anchor_by_frame = {int(anchor["frame_index"]): anchor for anchor in anchors}
    if sorted(anchor_by_frame) != [int(a["frame_index"]) for a in anchors]:
        raise ValueError("Anchors must be unique and sorted")

    anchor_indices = sorted(anchor_by_frame)
    anchor_masks = {
        index: cv2.resize(anchor_mask(anchor_by_frame[index], frames[index]), half_size, interpolation=cv2.INTER_NEAREST)
        for index in anchor_indices
    }
    candidates: list[tuple[np.ndarray, float, list[str]] | None] = [None] * len(frames)

    # Bidirectional propagation bounds drift by resetting from the nearer manual
    # anchor.  Confidence still cannot establish semantic correctness.
    for left, right in zip(anchor_indices[:-1], anchor_indices[1:]):
        forward: dict[int, tuple[np.ndarray, float, list[str]]] = {}
        mask = anchor_masks[left]
        for index in range(left + 1, right):
            mask, confidence, flags = _propagate_one(grays[index - 1], grays[index], mask)
            confidence *= math.exp(-(index - left) / 180.0)
            if confidence < 0.55:
                flags = sorted(set(flags + ["low_confidence"]))
            forward[index] = (mask, confidence, flags)
        backward: dict[int, tuple[np.ndarray, float, list[str]]] = {}
        mask = anchor_masks[right]
        for index in range(right - 1, left, -1):
            mask, confidence, flags = _propagate_one(grays[index + 1], grays[index], mask)
            confidence *= math.exp(-(right - index) / 180.0)
            if confidence < 0.55:
                flags = sorted(set(flags + ["low_confidence"]))
            backward[index] = (mask, confidence, flags)
        for index in range(left + 1, right):
            candidates[index] = forward[index] if index - left <= right - index else backward[index]

    # Continue forward after the release anchor; the carton is normally static.
    last = anchor_indices[-1]
    mask = anchor_masks[last]
    for index in range(last + 1, len(frames)):
        mask, confidence, flags = _propagate_one(grays[index - 1], grays[index], mask)
        confidence *= math.exp(-(index - last) / 180.0)
        if confidence < 0.55:
            flags = sorted(set(flags + ["low_confidence"]))
        candidates[index] = (mask, confidence, flags)

    records: list[dict[str, Any]] = []
    for frame_index in range(len(frames)):
        anchor = anchor_by_frame.get(frame_index)
        if anchor is not None:
            mask = anchor_masks[frame_index]
            confidence = 1.0 if np.any(mask) else 0.0
            flags = ["manual_keyframe"]
            if anchor["visibility"] != "visible":
                flags.append(anchor["visibility"])
            status, source_phase = "manual_reviewed", anchor["phase"]
        elif candidates[frame_index] is None:
            mask = np.zeros((half_size[1], half_size[0]), dtype=np.uint8)
            confidence, flags, status, source_phase = 0.0, ["no_anchor_track"], "pending", None
        else:
            mask, confidence, flags = candidates[frame_index]
            status, source_phase = "pending", None
        records.append(
            {
                "frame_index": frame_index,
                "polygon_xy": _mask_polygon(mask, width, height),
                "confidence": round(confidence, 6),
                "flags": flags,
                "review_status": status,
                "manual_phase": source_phase,
            }
        )
    return records


def _draw_record(frame: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    result = frame.copy()
    polygon = record["polygon_xy"]
    color = (0, 255, 0) if record["review_status"] == "manual_reviewed" else (0, 190, 255)
    if "low_confidence" in record["flags"] or "discontinuity" in record["flags"]:
        color = (0, 0, 255)
    if polygon:
        cv2.polylines(result, [np.asarray(polygon, np.int32)], True, color, 5, cv2.LINE_AA)
    label = f"f{record['frame_index']:03d} c={record['confidence']:.2f} {','.join(record['flags'][:2])}"
    cv2.rectangle(result, (0, 0), (result.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(result, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return result


def make_review_sheet(frames: list[np.ndarray], records: list[dict[str, Any]], output: Path) -> None:
    chosen = sorted(
        set(range(0, len(frames), 30))
        | {r["frame_index"] for r in records if "manual_keyframe" in r["flags"]}
        | {r["frame_index"] for r in records if "discontinuity" in r["flags"]}
    )
    # Bound the review sheet while retaining every manual/discontinuity frame.
    if len(chosen) > 48:
        mandatory = {r["frame_index"] for r in records if "manual_keyframe" in r["flags"] or "discontinuity" in r["flags"]}
        sampled = chosen[:: max(1, len(chosen) // 48)]
        chosen = sorted(mandatory | set(sampled))[:64]
    thumbs = [cv2.resize(_draw_record(frames[i], records[i]), (384, 216)) for i in chosen]
    blank = np.zeros_like(thumbs[0])
    while len(thumbs) % 4:
        thumbs.append(blank)
    rows = [cv2.hconcat(thumbs[i : i + 4]) for i in range(0, len(thumbs), 4)]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.vconcat(rows))


def propagate_all(spec_path: Path, output_dir: Path) -> Path:
    spec = load_spec(spec_path)
    root = Path(spec["dataset_root"])
    episodes = episode_metadata(root)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "source_dataset": str(root.resolve()),
        "selected_source_episodes": spec["selected_source_episodes"],
        "policy": "propagated masks are pending review regardless of confidence",
        "tracks": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_episode in spec["selected_source_episodes"]:
        report["tracks"][str(source_episode)] = {}
        for camera in CAMERAS:
            frames, video = decode_episode_frames(root, episodes[source_episode], camera)
            anchors = spec["episodes"][str(source_episode)][camera]
            records = propagate_camera(frames, anchors)
            counts = {
                "frames": len(records),
                "manual": sum(r["review_status"] == "manual_reviewed" for r in records),
                "pending": sum(r["review_status"] == "pending" for r in records),
                "low_confidence": sum("low_confidence" in r["flags"] for r in records),
                "discontinuity": sum("discontinuity" in r["flags"] for r in records),
                "empty_mask": sum(not r["polygon_xy"] for r in records),
            }
            report["tracks"][str(source_episode)][camera] = {
                "source_video": video,
                "counts": counts,
                "frames": records,
            }
            make_review_sheet(frames, records, output_dir / f"episode_{source_episode:03d}_{camera}_review.jpg")
    path = output_dir / "propagation_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def extract_keyframes(spec_path: Path, output_dir: Path) -> None:
    spec = load_spec(spec_path)
    root = Path(spec["dataset_root"])
    episodes = episode_metadata(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_episode in spec["selected_source_episodes"]:
        for camera in CAMERAS:
            frames, _ = decode_episode_frames(root, episodes[source_episode], camera)
            for anchor in spec["episodes"][str(source_episode)][camera]:
                frame = frames[int(anchor["frame_index"])]
                preview = _draw_record(
                    frame,
                    {
                        "frame_index": anchor["frame_index"],
                        "polygon_xy": anchor.get("polygons_xy", [[]])[0] if anchor.get("polygons_xy") else [],
                        "confidence": 1.0,
                        "flags": ["manual_keyframe", anchor["visibility"]],
                        "review_status": "manual_reviewed",
                    },
                )
                cv2.imwrite(
                    str(output_dir / f"e{source_episode:03d}_{camera}_{anchor['frame_index']:04d}_{anchor['phase']}.png"),
                    preview,
                )


def _record_mask(record: dict[str, Any], shape: tuple[int, int, int]) -> np.ndarray:
    polygon = record["polygon_xy"]
    if not polygon:
        return np.zeros(shape[:2], np.uint8)
    return polygon_mask(shape, [tuple(p) for p in polygon])


def build_derived_dataset(
    spec_path: Path,
    propagation_path: Path | None,
    output_root: Path,
    *,
    mode: str = "highlight",
    repo_id: str | None = None,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if output_root.exists():
        raise RuntimeError(f"Refusing to overwrite derived dataset: {output_root}")
    if mode not in {"original", "highlight"}:
        raise ValueError(f"Unsupported derived dataset mode: {mode}")
    if mode == "highlight" and propagation_path is None:
        raise ValueError("Highlighted mode requires a propagation report")
    spec = load_spec(spec_path)
    propagation = (
        json.loads(propagation_path.read_text(encoding="utf-8")) if propagation_path is not None else None
    )
    source_root = Path(spec["dataset_root"])
    source = LeRobotDataset(spec["source_repo_id"], root=source_root, video_backend="pyav")
    info = json.loads((source_root / "meta/info.json").read_text(encoding="utf-8"))
    features = {k: deepcopy(v) for k, v in info["features"].items() if k not in DEFAULT_FEATURES}
    derived = LeRobotDataset.create(
        repo_id=repo_id or spec["derived_repo_id"],
        root=output_root,
        fps=info["fps"],
        features=features,
        robot_type=info["robot_type"],
        use_videos=True,
        video_backend="pyav",
        image_writer_threads=4,
    )
    episodes = episode_metadata(source_root)
    mapping = []
    try:
        for derived_episode, source_episode in enumerate(spec["selected_source_episodes"]):
            episode = episodes[source_episode]
            tracks = propagation["tracks"][str(source_episode)] if propagation is not None else None
            start = int(episode["dataset_from_index"])
            for local_frame in range(int(episode["length"])):
                item = source[start + local_frame]
                frame: dict[str, Any] = {
                    "action": item["action"].numpy().copy(),
                    "observation.state": item["observation.state"].numpy().copy(),
                    "task": item["task"],
                }
                for camera in CAMERAS:
                    key = f"observation.images.{camera}"
                    rgb = np.transpose(item[key].numpy(), (1, 2, 0))
                    bgr = cv2.cvtColor(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    if mode == "highlight":
                        record = tracks[camera]["frames"][local_frame]
                        mask = _record_mask(record, bgr.shape)
                        bgr = highlight_target(bgr, mask) if np.any(mask) else bgr
                    frame[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                derived.add_frame(frame)
            derived.save_episode(parallel_encoding=True)
            mapping.append(
                {
                    "derived_episode_index": derived_episode,
                    "source_episode_index": source_episode,
                    "source_dataset_from_index": episode["dataset_from_index"],
                    "source_dataset_to_index": episode["dataset_to_index"],
                    "length": episode["length"],
                }
            )
    finally:
        derived.finalize()
    (output_root / "pilot_provenance.json").write_text(
        json.dumps(
            {
                "source_dataset": str(source_root.resolve()),
                "source_repo_id": spec["source_repo_id"],
                "derived_repo_id": repo_id or spec["derived_repo_id"],
                "visual_mode": mode,
                "annotation_spec": str(spec_path.resolve()),
                "propagation_report": str(propagation_path.resolve()) if propagation_path is not None else None,
                "episode_mapping": mapping,
                "invariants": [
                    "action values unchanged",
                    "observation.state values unchanged",
                    "timestamp=source frame_index/30 unchanged within each episode",
                    "frame_index and episode length unchanged",
                    "derived episode indices are contiguous; source indices are in episode_mapping",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_annotation_html(spec_path: Path, keyframe_dir: Path, output: Path) -> None:
    """Write a compact correction guide with embedded keyframe previews.

    The JSON remains the canonical annotation.  The page provides exact file and
    phase context and documents the correction loop without requiring a GUI
    dependency in the training environment.
    """
    spec = load_spec(spec_path)
    cards = []
    for source_episode in spec["selected_source_episodes"]:
        for camera in CAMERAS:
            for anchor in spec["episodes"][str(source_episode)][camera]:
                name = f"e{source_episode:03d}_{camera}_{anchor['frame_index']:04d}_{anchor['phase']}.png"
                image_path = keyframe_dir / name
                encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
                cards.append(
                    f"<article><h3>E{source_episode:03d} {camera} f{anchor['frame_index']} "
                    f"{anchor['phase']} ({anchor['visibility']})</h3>"
                    f"<img src='data:image/png;base64,{encoded}'><p>Edit <code>{spec_path.name}</code>: "
                    f"episodes.{source_episode}.{camera}, phase={anchor['phase']}. "
                    "Polygons include only visible carton pixels; use exclude polygons for gripper/occluders.</p></article>"
                )
    html = """<!doctype html><meta charset='utf-8'><title>Carton pilot annotation review</title>
<style>body{font:15px sans-serif;background:#181818;color:#eee;margin:20px}article{margin:18px 0;padding:12px;background:#252525}img{max-width:900px;width:100%;height:auto}code{color:#8ee}</style>
<h1>Carton keyframe correction workflow</h1><p>Green contour = manually reviewed keyframe. Correct the canonical JSON, rerun extraction, then propagation. Empty contours are intentional only when visibility is fully_occluded.</p>""" + "\n".join(cards)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def build_ab_preview(spec_path: Path, original_root: Path, highlighted_root: Path, output: Path) -> None:
    """Create a compact encoded-video comparison for both pilot cameras."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    spec = load_spec(spec_path)
    original = LeRobotDataset(original_root.name, root=original_root, video_backend="pyav")
    highlighted = LeRobotDataset(highlighted_root.name, root=highlighted_root, video_backend="pyav")
    episode_starts = [0]
    for episode_index in range(len(original.meta.episodes) - 1):
        episode = original.meta.episodes[episode_index]
        episode_starts.append(episode_starts[-1] + int(episode["length"]))

    phases = ("approach", "grasp", "transport", "placement", "release")
    rows: list[np.ndarray] = []
    for derived_episode, source_episode in enumerate(spec["selected_source_episodes"]):
        phone_anchors = spec["episodes"][str(source_episode)]["phone"]
        for phase in phases:
            anchor = next(item for item in phone_anchors if item["phase"] == phase)
            index = episode_starts[derived_episode] + int(anchor["frame_index"])
            cells = []
            for camera in CAMERAS:
                key = f"observation.images.{camera}"
                for label, dataset in (("A original", original), ("B highlighted", highlighted)):
                    rgb = np.transpose(dataset[index][key].numpy(), (1, 2, 0))
                    bgr = cv2.cvtColor(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cell = cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA)
                    cv2.putText(
                        cell,
                        f"E{source_episode} f{anchor['frame_index']} {phase} | {camera} | {label}",
                        (7, 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cells.append(cell)
            rows.append(np.hstack(cells))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.vstack(rows)):
        raise RuntimeError(f"Failed to write preview: {output}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract-keyframes")
    extract.add_argument("--spec", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--html", type=Path)
    prop = sub.add_parser("propagate")
    prop.add_argument("--spec", type=Path, required=True)
    prop.add_argument("--output-dir", type=Path, required=True)
    build = sub.add_parser("build-dataset")
    build.add_argument("--spec", type=Path, required=True)
    build.add_argument("--propagation", type=Path)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--mode", choices=("original", "highlight"), default="highlight")
    build.add_argument("--repo-id")
    preview = sub.add_parser("preview-ab")
    preview.add_argument("--spec", type=Path, required=True)
    preview.add_argument("--original-root", type=Path, required=True)
    preview.add_argument("--highlighted-root", type=Path, required=True)
    preview.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "extract-keyframes":
        extract_keyframes(args.spec, args.output_dir)
        if args.html:
            build_annotation_html(args.spec, args.output_dir, args.html)
    elif args.command == "propagate":
        print(propagate_all(args.spec, args.output_dir))
    elif args.command == "build-dataset":
        build_derived_dataset(
            args.spec,
            args.propagation,
            args.output_root,
            mode=args.mode,
            repo_id=args.repo_id,
        )
    else:
        build_ab_preview(args.spec, args.original_root, args.highlighted_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
