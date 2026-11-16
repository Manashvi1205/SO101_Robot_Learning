# Wrist colored-box detector v2

## Failure diagnosis

The previous wrist detector failed mainly because its normalized ROI started at
`y=0.34`. With the new inspection view this retained only 7.4% of annotated blue
pixels, 25.0% of red pixels, and 40.5% of green pixels. The HSV ranges themselves
covered most box pixels, but the saturated-blue range also covered the blue-gray
mat. In representative scene 039, the largest raw legacy blue component contained
426,060 pixels, so it was rejected by the object-size filter. A small isolated
blue/purple gripper region then became the winning candidate.

Four tuning images (033, 039, 042, and 045) were polygon-annotated. Their measured
box-face distributions are in `outputs/color_diagnostics_lab_v2_final/diagnostic_report.json`.
The most useful separation is in OpenCV Lab space:

- blue `b`: p01=62, median=68, p99=81;
- green `a`: p01=70, median=74, p99=121;
- red `a`: p01=139, median=196, p99=201.

The v2 rules therefore combine a chromatic mode with a bright/pale-face mode.
They are not arbitrary HSV guesses: the configured bands are based on these
annotated distributions. The improved blue mask's largest scene-039 component is
34,665 pixels and corresponds to the physical blue box.

An empty-workspace subtraction model was not adopted. No empty frame exists in
this batch, and a median assembled from object-containing frames also contains
lighting, shadow, and displaced-object artifacts. A real empty-workspace image
could still be useful later, but it is not required for the current result.

## Detector

The external camera retains the successful HSV detector. The wrist pipeline now
uses:

1. measured two-mode Lab color masks;
2. a workspace ROI appropriate to the current fixed inspection view;
3. a fixed exclusion polygon for the gripper silhouette;
4. opening and closing to combine visible box faces;
5. contour area and axis-aligned size constraints;
6. rotated minimum-area rectangles, rotated rectangularity, and solidity;
7. one geometrically strongest candidate per known color.

The returned record includes both an axis-aligned bounding box and the four
points of the rotated rectangle. Partial color-face visibility is supported, but
severe full occlusion remains outside this classical detector's capability.

## Evaluation

Pixel calibration used only scenes 033, 039, 042, and 045. Scenes 046-061 were
excluded from the annotation-based color calibration and evaluated after the
configuration was fixed.

| Wrist result | Old red | v2 red | Old green | v2 green | Old blue | v2 blue |
|---|---:|---:|---:|---:|---:|---:|
| Held-out correct / 16 | 8 | 16 | 8 | 16 | 0 | 16 |
| Held-out false positives | 0 | 0 | 0 | 0 | 16 | 0 |
| Held-out misses | 8 | 0 | 8 | 0 | 16 | 0 |

Across all scenes 033-061, visual review finds 29/29 correct localizations for
each wrist color and no false positives. The old result was red 14/29, green
18/29, and blue 0/29, with 26 false blue detections.

These are visually verified physical-object localization counts, not dense-mask
IoU measurements. Dense held-out polygons were intentionally not invented after
the fact. The JSON manual reviews make that limitation explicit.

## Reproduce

```bash
PYTHONPATH=src python -m so101_hierarchical.color_diagnostics \
  --input data/vision \
  --annotations annotations/wrist_color_tuning.json \
  --config config/color_detection.yaml \
  --output outputs/color_diagnostics_reproduced

PYTHONPATH=src python -m so101_hierarchical.color_detection \
  --input data/vision \
  --output outputs/perception_eval_lab_v2_reproduced \
  --config config/color_detection.yaml \
  --start-scene 46 --end-scene 61
```

To annotate another representative frame without OpenCV GUI support:

```bash
PYTHONPATH=src python -m so101_hierarchical.annotate_color_regions \
  --image data/vision/wrist/scene_033.png \
  --output outputs/scene_033_annotation_tool.html
```

Open the generated HTML file, click a polygon for red, green, and blue, and use
the Download JSON button.

## Remaining limitations

- The camera, lighting, mat, boxes, and inspection pose are fixed in this study.
- The gripper exclusion polygon depends on the current camera mounting and pose.
- A box almost entirely outside the ROI or fully hidden by the gripper can fail.
- New colors or materially different lighting require fresh representative
  annotations and re-estimation, not ad hoc threshold edits.
