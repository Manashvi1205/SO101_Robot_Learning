import cv2
import numpy as np
import pytest

from so101_hierarchical.target_augmentation import highlight_target, polygon_mask, recolor_target


def test_polygon_mask_rejects_out_of_bounds_points():
    with pytest.raises(ValueError, match="outside"):
        polygon_mask((20, 30, 3), [(1, 1), (31, 1), (1, 10)])


def test_recolor_preserves_pixels_outside_mask_and_changes_inside():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    for row in range(80):
        image[row, :] = (row + 40, row + 50, row + 60)
    mask = polygon_mask(image.shape, [(20, 15), (80, 15), (80, 65), (20, 65)])
    result = recolor_target(image, mask, seed=7)
    assert np.array_equal(result[mask == 0], image[mask == 0])
    assert np.mean(np.abs(result[mask > 0].astype(float) - image[mask > 0])) > 5


def test_recolor_retains_lightness_structure_and_highlight_is_visible():
    gradient = np.tile(np.arange(256, dtype=np.uint8), (100, 1))
    image = cv2.merge([gradient, gradient, gradient])
    mask = polygon_mask(image.shape, [(30, 10), (220, 10), (220, 90), (30, 90)])
    recolored = recolor_target(image, mask, texture_strength=0, seed=1)
    original_l = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0][mask > 0].astype(float)
    recolored_l = cv2.cvtColor(recolored, cv2.COLOR_BGR2LAB)[..., 0][mask > 0].astype(float)
    assert np.corrcoef(original_l, recolored_l)[0, 1] > 0.99
    highlighted = highlight_target(image, mask)
    assert not np.array_equal(highlighted, image)
