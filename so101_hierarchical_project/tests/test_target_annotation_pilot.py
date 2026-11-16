import numpy as np

from so101_hierarchical.target_annotation_pilot import _propagate_one, anchor_mask, propagate_camera


def test_fully_occluded_anchor_is_empty():
    anchor = {"phase": "occlusion", "visibility": "fully_occluded", "polygons_xy": []}
    assert not anchor_mask(anchor, (40, 50, 3)).any()


def test_static_texture_mask_propagates_with_review_confidence():
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 255, (120, 160), dtype=np.uint8)
    mask = np.zeros_like(frame)
    mask[25:95, 40:130] = 255
    propagated, confidence, flags = _propagate_one(frame, frame.copy(), mask)
    assert np.mean(propagated == mask) > 0.99
    assert confidence > 0.5
    assert "discontinuity" not in flags


def test_propagated_frames_are_never_marked_reviewed():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, (90, 120, 3), dtype=np.uint8)
    anchors = [
        {"frame_index": 0, "phase": "approach", "visibility": "visible", "polygons_xy": [[[20, 20], [90, 20], [90, 70], [20, 70]]]},
        {"frame_index": 2, "phase": "grasp", "visibility": "visible", "polygons_xy": [[[20, 20], [90, 20], [90, 70], [20, 70]]]},
    ]
    records = propagate_camera([image, image.copy(), image.copy()], anchors, half_size=(60, 45))
    assert records[0]["review_status"] == "manual_reviewed"
    assert records[1]["review_status"] == "pending"
    assert records[2]["review_status"] == "manual_reviewed"
