"""Audit and build the nine-episode matched multibox A/B datasets.

Source roots are immutable.  The highlighted arm uses the frozen colored-box
detector when it returns the task-selected color and a constrained color-mask
fallback otherwise.  Fallbacks are reported for review; they are not silently
treated as manually verified annotations.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterator, Sequence

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

from .color_detection import (
    _mask_for_ranges,
    _normalized_polygon,
    detect_colored_boxes,
    load_detection_config,
)
from .target_augmentation import highlight_target


ROOT_PATTERN = re.compile(r"so101_multibox_(arr[123])_(red|blue|green)_[0-9_]+$")
CAMERAS = ("phone", "wrist")
EXPECTED = [(f"arr{arr}", color) for arr in range(1, 4) for color in ("red", "blue", "green")]
NON_MODEL_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_sources(local_root: Path) -> list[dict[str, Any]]:
    found: dict[tuple[str, str], Path] = {}
    for path in local_root.iterdir():
        match = ROOT_PATTERN.fullmatch(path.name)
        if match:
            key = (match.group(1), match.group(2))
            if key in found:
                raise RuntimeError(f"Multiple source roots for {key}: {found[key]} and {path}")
            found[key] = path
    missing = [key for key in EXPECTED if key not in found]
    extra = sorted(set(found) - set(EXPECTED))
    if missing or extra:
        raise RuntimeError(f"Unexpected multibox source set; missing={missing}, extra={extra}")
    return [{"arrangement": a, "color": c, "root": found[(a, c)]} for a, c in EXPECTED]


def _video_probe(path: Path) -> dict[str, Any]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        count = sum(1 for _ in container.decode(video=0))
        return {
            "path": str(path.resolve()),
            "codec": stream.codec_context.name,
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "fps": float(stream.average_rate),
            "decoded_frames": count,
        }


def audit_sources(local_root: Path) -> dict[str, Any]:
    records = []
    all_valid = True
    for source in discover_sources(local_root):
        root, color = source["root"], source["color"]
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        data_path = root / "data/chunk-000/file-000.parquet"
        episode_path = root / "meta/episodes/chunk-000/file-000.parquet"
        data = pq.read_table(data_path)
        episodes = pq.read_table(episode_path).to_pylist()
        tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
        rows = data.num_rows
        action = np.asarray(data["action"].to_pylist(), dtype=np.float32)
        state = np.asarray(data["observation.state"].to_pylist(), dtype=np.float32)
        timestamps = np.asarray(data["timestamp"].to_pylist(), dtype=np.float64)
        frames = np.asarray(data["frame_index"].to_pylist(), dtype=np.int64)
        indices = np.asarray(data["index"].to_pylist(), dtype=np.int64)
        episode_indices = np.asarray(data["episode_index"].to_pylist(), dtype=np.int64)
        task_indices = np.asarray(data["task_index"].to_pylist(), dtype=np.int64)
        task = tasks[0]["task"] if len(tasks) == 1 else ""
        videos = {
            camera: _video_probe(
                root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
            )
            for camera in CAMERAS
        }
        checks = {
            "one_episode": len(episodes) == 1 and info["total_episodes"] == 1,
            "row_episode_length_match": len(episodes) == 1
            and rows == int(episodes[0]["length"])
            and rows == int(info["total_frames"]),
            "both_videos_exist": all(Path(v["path"]).is_file() for v in videos.values()),
            "video_counts_match_rows": all(v["decoded_frames"] == rows for v in videos.values()),
            "video_contract": all(
                v["width"] == 1280 and v["height"] == 720 and abs(v["fps"] - 30.0) < 1e-6
                for v in videos.values()
            ),
            "finite_action": bool(np.isfinite(action).all()),
            "finite_state": bool(np.isfinite(state).all()),
            "task_color_correct": color in task.lower()
            and not any(other in task.lower() for other in {"red", "blue", "green"} - {color}),
            "frame_index_contiguous": bool(np.array_equal(frames, np.arange(rows))),
            "index_contiguous": bool(np.array_equal(indices, np.arange(rows))),
            "single_episode_index": bool(np.all(episode_indices == 0)),
            "single_task_index": bool(np.all(task_indices == 0)),
            "timestamp_30hz": bool(np.allclose(timestamps, np.arange(rows) / 30.0, atol=1e-5)),
        }
        valid = all(checks.values())
        all_valid &= valid
        records.append(
            {
                **{k: str(v.resolve()) if k == "root" else v for k, v in source.items()},
                "rows": rows,
                "task": task,
                "checks": checks,
                "videos": videos,
                "valid": valid,
            }
        )
    return {"schema_version": "1.0", "all_valid": all_valid, "datasets": records}


def _decode(path: Path) -> Iterator[np.ndarray]:
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="bgr24")


def _fallback_mask(image: np.ndarray, color: str, config: Any) -> tuple[np.ndarray, float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = _mask_for_ranges(hsv, config.colors[color])
    roi = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(roi, [_normalized_polygon(config.roi_polygon, image)], 255)
    for polygon in config.exclusion_polygons:
        cv2.fillPoly(roi, [_normalized_polygon(polygon, image)], 0)
    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros(mask.shape, np.uint8), 0.0
    candidates = []
    image_area = image.shape[0] * image.shape[1]
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < 180 or area > image_area * 0.12 or width < 8 or height < 8:
            continue
        # Very broad, thin regions are usually mat/background leakage.
        if width > 500 or height > 500:
            continue
        candidates.append((int(area), label))
    if not candidates:
        return np.zeros(mask.shape, np.uint8), 0.0
    area, label = max(candidates)
    selected = np.where(labels == label, 255, 0).astype(np.uint8)
    return selected, float(area)


def target_mask(image: np.ndarray, camera: str, color: str, configs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    config = configs["external" if camera == "phone" else "wrist"]
    detections = [item for item in detect_colored_boxes(image, config) if item.color == color]
    if detections:
        detection = max(detections, key=lambda item: item.quality_score)
        polygon = np.round(np.asarray(detection.rotated_box)).astype(np.int32)
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask, [polygon], 255, lineType=cv2.LINE_AA)
        return mask, {
            "method": "frozen_detector_rotated_box",
            "quality": round(float(detection.quality_score), 6),
            "mask_pixels": int(np.count_nonzero(mask)),
            "flags": [],
        }
    mask, area = _fallback_mask(image, color, config)
    if np.any(mask):
        return mask, {
            "method": "task_color_fallback",
            "quality": None,
            "mask_pixels": int(np.count_nonzero(mask)),
            "flags": ["manual_confirmation_required"],
            "fallback_component_pixels": int(area),
        }
    return mask, {
        "method": "empty",
        "quality": None,
        "mask_pixels": 0,
        "flags": ["visibility_or_mask_manual_confirmation_required"],
    }


def _ranges(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    result, start, previous = [], values[0], values[0]
    for value in values[1:]:
        if value != previous + 1:
            result.append([start, previous])
            start = value
        previous = value
    result.append([start, previous])
    return result


def _review_sheet(samples: list[np.ndarray], output: Path) -> None:
    blank = np.zeros_like(samples[0])
    while len(samples) % 4:
        samples.append(blank)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [np.hstack(samples[i : i + 4]) for i in range(0, len(samples), 4)]
    if not cv2.imwrite(str(output), np.vstack(rows)):
        raise RuntimeError(f"Failed to write {output}")


def build_pair(
    local_root: Path,
    original_root: Path,
    highlight_root: Path,
    output_dir: Path,
    normalization_stats: Path,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if original_root.exists() or highlight_root.exists():
        raise RuntimeError("Refusing to overwrite an existing derived dataset root")
    audit = audit_sources(local_root)
    if not audit["all_valid"]:
        raise RuntimeError("Source audit failed; refusing to build derived datasets")
    sources = discover_sources(local_root)
    first_info = json.loads((sources[0]["root"] / "meta/info.json").read_text(encoding="utf-8"))
    features = {k: deepcopy(v) for k, v in first_info["features"].items() if k not in NON_MODEL_FEATURES}
    original = LeRobotDataset.create(
        repo_id=original_root.name,
        root=original_root,
        fps=30,
        features=features,
        robot_type=first_info["robot_type"],
        use_videos=True,
        video_backend="pyav",
        image_writer_threads=4,
    )
    highlighted = LeRobotDataset.create(
        repo_id=highlight_root.name,
        root=highlight_root,
        fps=30,
        features=features,
        robot_type=first_info["robot_type"],
        use_videos=True,
        video_backend="pyav",
        image_writer_threads=4,
    )
    _, configs = load_detection_config(Path(__file__).resolve().parents[2] / "config/color_detection.yaml")
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema_version": "1.0", "episodes": []}
    provenance = []
    try:
        for derived_episode, source in enumerate(sources):
            root, color = source["root"], source["color"]
            data = pq.read_table(root / "data/chunk-000/file-000.parquet")
            task = pq.read_table(root / "meta/tasks.parquet").to_pylist()[0]["task"]
            streams = {
                camera: _decode(root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4")
                for camera in CAMERAS
            }
            camera_records: dict[str, list[dict[str, Any]]] = {camera: [] for camera in CAMERAS}
            review_samples: dict[str, list[np.ndarray]] = {camera: [] for camera in CAMERAS}
            length = data.num_rows
            sample_indices = set(np.linspace(0, length - 1, 36, dtype=int).tolist())
            for frame_index in range(length):
                images = {camera: next(streams[camera]) for camera in CAMERAS}
                masks: dict[str, np.ndarray] = {}
                for camera in CAMERAS:
                    mask, record = target_mask(images[camera], camera, color, configs)
                    record["frame_index"] = frame_index
                    camera_records[camera].append(record)
                    masks[camera] = mask
                    if frame_index in sample_indices:
                        shown = highlight_target(images[camera], mask) if np.any(mask) else images[camera].copy()
                        shown = cv2.resize(shown, (384, 216), interpolation=cv2.INTER_AREA)
                        cv2.rectangle(shown, (0, 0), (384, 24), (0, 0, 0), -1)
                        cv2.putText(
                            shown,
                            f"{source['arrangement']}_{color} {camera} f{frame_index} {record['method']}",
                            (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (255, 255, 255), 1, cv2.LINE_AA,
                        )
                        review_samples[camera].append(shown)
                row = data.slice(frame_index, 1).to_pylist()[0]
                base = {
                    "action": np.asarray(row["action"], dtype=np.float32),
                    "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                    "task": task,
                }
                original_frame, highlighted_frame = dict(base), dict(base)
                for camera in CAMERAS:
                    key = f"observation.images.{camera}"
                    original_frame[key] = cv2.cvtColor(images[camera], cv2.COLOR_BGR2RGB)
                    changed = highlight_target(images[camera], masks[camera]) if np.any(masks[camera]) else images[camera]
                    highlighted_frame[key] = cv2.cvtColor(changed, cv2.COLOR_BGR2RGB)
                original.add_frame(original_frame)
                highlighted.add_frame(highlighted_frame)
            original.save_episode(parallel_encoding=True)
            highlighted.save_episode(parallel_encoding=True)
            episode_report = {
                "derived_episode_index": derived_episode,
                "arrangement": source["arrangement"],
                "target_color": color,
                "source_root": str(root.resolve()),
                "length": length,
                "cameras": {},
            }
            for camera in CAMERAS:
                records = camera_records[camera]
                fallback = [r["frame_index"] for r in records if r["method"] == "task_color_fallback"]
                empty = [r["frame_index"] for r in records if r["method"] == "empty"]
                episode_report["cameras"][camera] = {
                    "detector_frames": sum(r["method"] == "frozen_detector_rotated_box" for r in records),
                    "fallback_frames": len(fallback),
                    "fallback_ranges": _ranges(fallback),
                    "empty_frames": len(empty),
                    "empty_ranges": _ranges(empty),
                }
                _review_sheet(
                    review_samples[camera],
                    output_dir / f"{source['arrangement']}_{color}_{camera}_mask_review.jpg",
                )
            report["episodes"].append(episode_report)
            provenance.append(
                {
                    "derived_episode_index": derived_episode,
                    "arrangement": source["arrangement"],
                    "target_color": color,
                    "source_root": str(root.resolve()),
                    "source_episode_index": 0,
                    "length": length,
                    "task": task,
                }
            )
    finally:
        original.finalize()
        highlighted.finalize()

    for derived_root in (original_root, highlight_root):
        empirical = derived_root / "meta/stats.empirical_9episode.json"
        (derived_root / "meta/stats.json").replace(empirical)
        shutil.copy2(normalization_stats, derived_root / "meta/stats.json")
        (derived_root / "multibox_provenance.json").write_text(
            json.dumps(
                {
                    "visual_mode": "original" if derived_root == original_root else "highlight",
                    "episode_mapping": provenance,
                    "normalization": {
                        "active_stats_source": str(normalization_stats.resolve()),
                        "active_stats_sha256": sha256(normalization_stats),
                        "empirical_stats": "meta/stats.empirical_9episode.json",
                    },
                    "invariants": [
                        "source datasets are read-only",
                        "action and observation.state values are unchanged",
                        "timestamps are regenerated from the unchanged local frame index at 30 Hz",
                        "episode lengths and boundaries are unchanged",
                        "highlight mode changes camera pixels only",
                    ],
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    report_path = output_dir / "mask_generation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def make_previews(original_root: Path, highlight_root: Path, output_dir: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    original = LeRobotDataset(original_root.name, root=original_root, video_backend="pyav")
    highlighted = LeRobotDataset(highlight_root.name, root=highlight_root, video_backend="pyav")
    start = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for episode_index, episode in enumerate(original.meta.episodes):
        length = int(episode["length"])
        provenance = json.loads((original_root / "multibox_provenance.json").read_text(encoding="utf-8"))
        mapping = provenance["episode_mapping"][episode_index]
        rows = []
        for local in np.linspace(0, length - 1, 6, dtype=int):
            cells = []
            for camera in CAMERAS:
                key = f"observation.images.{camera}"
                for label, dataset in (("A ORIGINAL", original), ("B HIGHLIGHT", highlighted)):
                    rgb = np.transpose(dataset[start + int(local)][key].numpy(), (1, 2, 0))
                    bgr = cv2.cvtColor(np.clip(rgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cell = cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(cell, (0, 0), (320, 24), (0, 0, 0), -1)
                    cv2.putText(cell, f"{camera} {label} f{local}", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1, cv2.LINE_AA)
                    cells.append(cell)
            rows.append(np.hstack(cells))
        name = f"episode_{episode_index:02d}_{mapping['arrangement']}_{mapping['target_color']}_A_vs_B.jpg"
        if not cv2.imwrite(str(output_dir / name), np.vstack(rows)):
            raise RuntimeError(f"Failed to write preview {name}")
        start += length


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--local-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    build = sub.add_parser("build-pair")
    build.add_argument("--local-root", type=Path, required=True)
    build.add_argument("--original-root", type=Path, required=True)
    build.add_argument("--highlight-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--normalization-stats", type=Path, required=True)
    previews = sub.add_parser("previews")
    previews.add_argument("--original-root", type=Path, required=True)
    previews.add_argument("--highlight-root", type=Path, required=True)
    previews.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = audit_sources(args.local_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(args.output)
        return 0 if result["all_valid"] else 1
    if args.command == "build-pair":
        build_pair(args.local_root, args.original_root, args.highlight_root, args.output_dir, args.normalization_stats)
    else:
        make_previews(args.original_root, args.highlight_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
