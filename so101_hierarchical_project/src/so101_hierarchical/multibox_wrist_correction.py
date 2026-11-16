"""Conservative wrist-only corrections for the nine-episode multibox set.

Only intervals confirmed in the review report are overridden. Clearly visible
approach frames use sparse manual anchors plus optical flow. Close, grasped,
clipped, or occluded frames are explicitly intentional-empty.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

from .color_detection import load_detection_config
from .multibox_dataset import (
    CAMERAS,
    NON_MODEL_FEATURES,
    _decode,
    _ranges,
    discover_sources,
    sha256,
    target_mask,
)
from .target_annotation_pilot import propagate_camera
from .target_augmentation import highlight_target, polygon_mask


# Full-resolution manually reviewed visible-object polygons. The final anchor in
# each track lies just outside the replacement interval and bounds propagation.
VISIBLE_TRACKS: dict[tuple[str, str], dict[str, Any]] = {
    ("arr1", "red"): {
        "replace": [40, 168],
        "anchors": [
            (40, [(438, 0), (695, 0), (675, 151), (476, 118)]),
            (45, [(512, 0), (709, 0), (691, 131), (663, 286), (486, 261), (474, 157)]),
            (80, [(633, 385), (722, 405), (704, 527), (606, 504)]),
            (100, [(529, 282), (643, 282), (648, 402), (633, 414), (528, 401)]),
            (120, [(519, 285), (633, 283), (643, 403), (629, 416), (550, 416), (519, 401)]),
            (140, [(510, 288), (625, 286), (638, 405), (623, 417), (542, 417), (511, 402)]),
            (169, [(603, 288), (807, 327), (774, 518), (579, 485)]),
        ],
    },
    ("arr1", "blue"): {
        "replace": [73, 146],
        "anchors": [
            (72, [(849, 338), (951, 397), (905, 487), (842, 506), (765, 444), (794, 387)]),
            (100, [(864, 312), (969, 378), (912, 477), (850, 488), (766, 423), (805, 369)]),
            (125, [(721, 260), (918, 295), (872, 478), (685, 441)]),
            (147, [(525, 256), (770, 263), (758, 507), (518, 495)]),
        ],
    },
}


INTENTIONAL_EMPTY: dict[tuple[str, str], list[tuple[int, int, str]]] = {
    ("arr1", "red"): [(0, 39, "off_screen_or_only_marginally_clipped_during_entry")],
    ("arr1", "blue"): [
        (166, 184, "near_grasp_close_or_partial"),
        (210, 237, "grasp_close_or_occluded"),
        (460, 517, "post_grasp_close_clipped_or_partial"),
    ],
    ("arr1", "green"): [(470, 495, "post_grasp_close_clipped_or_partial")],
    ("arr2", "red"): [(473, 512, "post_grasp_close_clipped_or_partial")],
    ("arr2", "blue"): [
        (425, 456, "selected_blue_target_not_visible_during_retraction"),
        (500, 517, "post_grasp_close_clipped_or_partial"),
    ],
    ("arr2", "green"): [(475, 523, "post_grasp_close_clipped_or_partial")],
    ("arr3", "red"): [
        (244, 246, "near_grasp_close_or_partial"),
        (252, 254, "near_grasp_close_or_partial"),
        (264, 265, "near_grasp_close_or_partial"),
        (271, 271, "near_grasp_close_or_partial"),
        (331, 331, "near_grasp_close_or_partial"),
        (466, 506, "post_grasp_close_clipped_or_partial"),
    ],
    ("arr3", "blue"): [
        (300, 456, "near_grasp_close_or_partial"),
        (577, 626, "post_grasp_close_clipped_or_partial"),
    ],
    ("arr3", "green"): [
        (289, 323, "near_grasp_close_or_partial"),
        (468, 506, "post_grasp_close_clipped_or_partial"),
    ],
}


def decode_frames(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="bgr24") for frame in container.decode(video=0)]


def generate_corrections(local_root: Path, output: Path, review_dir: Path) -> dict[str, Any]:
    overrides: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    sources = discover_sources(local_root)
    review_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        key = (source["arrangement"], source["color"])
        if key not in VISIBLE_TRACKS and key not in INTENTIONAL_EMPTY:
            continue
        frames = decode_frames(
            source["root"] / "videos/observation.images.wrist/chunk-000/file-000.mp4"
        )
        records: dict[int, dict[str, Any]] = {}
        for start, end, reason in INTENTIONAL_EMPTY.get(key, []):
            for frame_index in range(start, end + 1):
                records[frame_index] = {
                    "frame_index": frame_index,
                    "polygon_xy": [],
                    "method": "intentional_empty",
                    "review_status": "reviewed_intentional_empty",
                    "reason": reason,
                    "flags": ["intentional_empty", "occluded_offscreen_close_or_partial"],
                }
        track = VISIBLE_TRACKS.get(key)
        if track:
            anchors = track["anchors"]
            slice_start, slice_end = anchors[0][0], anchors[-1][0]
            local_anchors = [
                {
                    "frame_index": index - slice_start,
                    "phase": f"correction_anchor_{index}",
                    "visibility": "visible",
                    "polygons_xy": [[list(point) for point in polygon]],
                }
                for index, polygon in anchors
            ]
            propagated = propagate_camera(frames[slice_start : slice_end + 1], local_anchors)
            replace_start, replace_end = track["replace"]
            for local_record in propagated:
                frame_index = slice_start + local_record["frame_index"]
                if replace_start <= frame_index <= replace_end:
                    records[frame_index] = {
                        "frame_index": frame_index,
                        "polygon_xy": local_record["polygon_xy"],
                        "method": "manual_anchor" if local_record["review_status"] == "manual_reviewed" else "optical_flow_between_reviewed_anchors",
                        "review_status": local_record["review_status"],
                        "confidence": local_record["confidence"],
                        "flags": local_record["flags"],
                    }
        overrides[key] = records

        # Review only overrides plus a small boundary sample.
        chosen = sorted(
            set(index for index in records if index % 10 == 0)
            | {min(records), max(records)}
            | {a for a, _ in VISIBLE_TRACKS.get(key, {}).get("anchors", [])}
            | {min(len(frames) - 1, b + 1) for _, b, _ in INTENTIONAL_EMPTY.get(key, [])}
        )
        cells = []
        for frame_index in chosen:
            record = records.get(frame_index)
            original = frames[frame_index]
            if record and record["polygon_xy"]:
                mask = polygon_mask(original.shape, [tuple(point) for point in record["polygon_xy"]])
                corrected = highlight_target(original, mask)
                method = record["method"]
            elif record:
                corrected = original.copy()
                method = "intentional_empty"
            else:
                corrected = original.copy()
                method = "unchanged_boundary_sample"
            left = cv2.resize(original, (320, 180), interpolation=cv2.INTER_AREA)
            right = cv2.resize(corrected, (320, 180), interpolation=cv2.INTER_AREA)
            cv2.rectangle(left, (0, 0), (320, 23), (0, 0, 0), -1)
            cv2.rectangle(right, (0, 0), (320, 23), (0, 0, 0), -1)
            cv2.putText(left, f"SOURCE f{frame_index}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(right, f"CORRECTED {method}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .37, (255,255,255), 1, cv2.LINE_AA)
            cells.append(np.hstack([left, right]))
        if cells:
            review_path = review_dir / f"{key[0]}_{key[1]}_wrist_corrections.jpg"
            cv2.imwrite(str(review_path), np.vstack(cells))

    payload = {
        "schema_version": "1.0",
        "policy": "highlight only clearly visible approach targets; close, grasped, partial, clipped, occluded, and off-screen frames are intentional empty",
        "episodes": {},
    }
    source_index = {(s["arrangement"], s["color"]): i for i, s in enumerate(sources)}
    for key, records in overrides.items():
        corrected = sorted(i for i, r in records.items() if r["polygon_xy"])
        empty = sorted(i for i, r in records.items() if not r["polygon_xy"])
        payload["episodes"][str(source_index[key])] = {
            "arrangement": key[0],
            "target_color": key[1],
            "wrist_overrides": [records[index] for index in sorted(records)],
            "corrected_visible_ranges": _ranges(corrected),
            "intentional_empty_ranges": _ranges(empty),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_corrected_highlight(
    local_root: Path,
    old_highlight_root: Path,
    output_root: Path,
    corrections_path: Path,
    normalization_stats: Path,
    report_output: Path,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if output_root.exists():
        raise RuntimeError(f"Refusing to overwrite {output_root}")
    corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
    sources = discover_sources(local_root)
    first_info = json.loads((sources[0]["root"] / "meta/info.json").read_text(encoding="utf-8"))
    features = {k: deepcopy(v) for k, v in first_info["features"].items() if k not in NON_MODEL_FEATURES}
    derived = LeRobotDataset.create(
        repo_id="MULTIBOX_HIGHLIGHT",
        root=output_root,
        fps=30,
        features=features,
        robot_type=first_info["robot_type"],
        use_videos=True,
        video_backend="pyav",
        image_writer_threads=4,
    )
    _, configs = load_detection_config(Path(__file__).resolve().parents[2] / "config/color_detection.yaml")
    episode_reports = []
    try:
        for episode_index, source in enumerate(sources):
            root, color = source["root"], source["color"]
            data = pq.read_table(root / "data/chunk-000/file-000.parquet")
            task = pq.read_table(root / "meta/tasks.parquet").to_pylist()[0]["task"]
            streams = {
                camera: _decode(root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4")
                for camera in CAMERAS
            }
            override_list = corrections["episodes"].get(str(episode_index), {}).get("wrist_overrides", [])
            overrides = {int(item["frame_index"]): item for item in override_list}
            counts = {"phone": {}, "wrist": {}}
            for frame_index in range(data.num_rows):
                images = {camera: next(streams[camera]) for camera in CAMERAS}
                row = data.slice(frame_index, 1).to_pylist()[0]
                frame = {
                    "action": np.asarray(row["action"], dtype=np.float32),
                    "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                    "task": task,
                }
                for camera in CAMERAS:
                    image = images[camera]
                    override = overrides.get(frame_index) if camera == "wrist" else None
                    if override is not None:
                        method = override["method"]
                        if override["polygon_xy"]:
                            mask = polygon_mask(image.shape, [tuple(p) for p in override["polygon_xy"]])
                            changed = highlight_target(image, mask)
                        else:
                            changed = image
                    else:
                        mask, record = target_mask(image, camera, color, configs)
                        method = record["method"]
                        changed = highlight_target(image, mask) if np.any(mask) else image
                    counts[camera][method] = counts[camera].get(method, 0) + 1
                    frame[f"observation.images.{camera}"] = cv2.cvtColor(changed, cv2.COLOR_BGR2RGB)
                derived.add_frame(frame)
            derived.save_episode(parallel_encoding=True)
            corrected_ranges = corrections["episodes"].get(str(episode_index), {})
            episode_reports.append(
                {
                    "derived_episode_index": episode_index,
                    "arrangement": source["arrangement"],
                    "target_color": color,
                    "source_root": str(root.resolve()),
                    "length": data.num_rows,
                    "camera_method_counts": counts,
                    "wrist_corrected_visible_ranges": corrected_ranges.get("corrected_visible_ranges", []),
                    "wrist_intentional_empty_ranges": corrected_ranges.get("intentional_empty_ranges", []),
                }
            )
    finally:
        derived.finalize()

    empirical = output_root / "meta/stats.empirical_9episode.json"
    (output_root / "meta/stats.json").replace(empirical)
    shutil.copy2(normalization_stats, output_root / "meta/stats.json")

    # Phone masks were accepted. Preserve the already encoded phone stream byte
    # for byte instead of introducing a second visual generation.
    old_phone = old_highlight_root / "videos/observation.images.phone"
    new_phone = output_root / "videos/observation.images.phone"
    shutil.rmtree(new_phone)
    shutil.copytree(old_phone, new_phone, copy_function=shutil.copy2)

    provenance = json.loads((old_highlight_root / "multibox_provenance.json").read_text(encoding="utf-8"))
    provenance["wrist_corrections"] = {
        "corrections": str(corrections_path.resolve()),
        "policy": corrections["policy"],
        "phone_stream": "byte-for-byte copied from the accepted prior MULTIBOX_HIGHLIGHT",
    }
    provenance["normalization"]["active_stats_sha256"] = sha256(normalization_stats)
    (output_root / "multibox_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": "2.0",
        "policy": corrections["policy"],
        "phone_masks": "unchanged; encoded phone stream copied byte-for-byte from prior highlighted dataset",
        "wrist_scope": "only intervals listed in the prior MASK_REVIEW_REPORT were overridden",
        "episodes": episode_reports,
    }
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--local-root", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--review-dir", type=Path, required=True)
    build = sub.add_parser("build")
    build.add_argument("--local-root", type=Path, required=True)
    build.add_argument("--old-highlight-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--corrections", type=Path, required=True)
    build.add_argument("--normalization-stats", type=Path, required=True)
    build.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        generate_corrections(args.local_root, args.output, args.review_dir)
    else:
        build_corrected_highlight(
            args.local_root,
            args.old_highlight_root,
            args.output_root,
            args.corrections,
            args.normalization_stats,
            args.report_output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
