# Three-episode carton-highlighting pilot

Date: 2026-11-16

## Result and scope

This is a working data/training compatibility pilot, not a reviewed training
set for a full experiment. Source episodes 3, 20, and 30 were selected after a
40-episode start-frame contact-sheet review:

- episode 3: carton starts right/far;
- episode 20: carton starts center/near;
- episode 30: carton starts left/near.

The matched derived datasets are
`/home/manashvi/.cache/huggingface/lerobot/so101_variable_box_3ep_original_control`
(A) and
`/home/manashvi/.cache/huggingface/lerobot/so101_variable_box_3ep_highlight_corrected`
(B). Each contains 3 episodes and 1,841 frames per camera. Source episode
indices are mapped to contiguous derived indices in `pilot_provenance.json`.

## Exact annotation workflow

1. Extract the reviewed keyframes and rebuild the self-contained review page:

   ```bash
   cd /home/manashvi/.cache/huggingface/lerobot
   PYTHONPATH=so101_hierarchical_project/src \
     /home/manashvi/miniforge3/envs/lerobot/bin/python \
     -m so101_hierarchical.target_annotation_pilot extract-keyframes \
     --spec so101_hierarchical_project/annotations/carton_pilot_3episodes.json \
     --output-dir so101_hierarchical_project/outputs/carton_pilot/keyframes \
     --html so101_hierarchical_project/outputs/carton_pilot/annotation_review.html
   ```

2. Inspect the HTML/PNG overlays. Edit only the canonical annotation JSON.
   Each polygon must cover visible carton pixels. Mark full invisibility with
   `visibility: fully_occluded` and an empty polygon; mark gripper overlap and
   clipping as `partial`. Add `exclude_polygons_xy` when an occluder must be
   removed. The current pilot uses 6 manual phases per camera—approach, grasp,
   occlusion, transport, placement, release—plus one extra wrist entry anchor
   for episode 3 when the carton becomes visible.

3. Propagate from both adjacent manual anchors and regenerate review sheets:

   ```bash
   PYTHONPATH=so101_hierarchical_project/src \
     /home/manashvi/miniforge3/envs/lerobot/bin/python \
     -m so101_hierarchical.target_annotation_pilot propagate \
     --spec so101_hierarchical_project/annotations/carton_pilot_3episodes.json \
     --output-dir so101_hierarchical_project/outputs/carton_pilot
   ```

   Propagation uses pyramidal Lucas–Kanade flow, forward/backward consistency,
   and a RANSAC partial-affine fit at half resolution. Manual anchors reset the
   track. Confidence combines consistent feature count, forward/backward error,
   affine inlier fraction, area change, and distance from the anchor.

4. Review `episode_*_{phone,wrist}_review.jpg`. Green is a manual anchor,
   yellow is pending propagation, and red is low-confidence/discontinuous.
   Confidence is only a review priority. Every propagated record remains
   `review_status: pending`, including high-confidence records. Correct bad
   intervals by adding/replacing a manual anchor and repeat steps 1–4.

5. Build new dataset roots only after the intended report is frozen:

   ```bash
   PYTHONPATH=so101_hierarchical_project/src \
     /home/manashvi/miniforge3/envs/lerobot/bin/python \
     -m so101_hierarchical.target_annotation_pilot build-dataset \
     --spec so101_hierarchical_project/annotations/carton_pilot_3episodes.json \
     --mode original \
     --repo-id so101_variable_box_3ep_original_control \
     --output-root so101_variable_box_3ep_original_control

   PYTHONPATH=so101_hierarchical_project/src \
     /home/manashvi/miniforge3/envs/lerobot/bin/python \
     -m so101_hierarchical.target_annotation_pilot build-dataset \
     --spec so101_hierarchical_project/annotations/carton_pilot_3episodes.json \
     --propagation so101_hierarchical_project/outputs/carton_pilot/propagation_report.json \
     --mode highlight \
     --repo-id so101_variable_box_3ep_highlight_corrected \
     --output-root so101_variable_box_3ep_highlight_corrected
   ```

   The builder refuses to overwrite an existing derived root. It applies the
   same cyan/white/black renderer to each available camera mask. Empty masks
   remain unchanged, which is correct for off-screen/full-occlusion frames.

## Review status

The corrected report contains 3,682 camera-frames: 44 manually reviewed anchors
and 3,638 pending propagated masks. It flags 210 low-confidence frames, zero
discontinuities, and 212 empty masks. The empty wrist masks cover
pre-entry/off-screen intervals and the episode-3 full-occlusion interval. Exact
remaining review intervals are in `outputs/carton_pilot/MASK_REVIEW_REPORT.md`;
every polygon and flag is in `propagation_report.json`.

Phone propagation is visually stronger. Wrist annotations remain coarse during
close-up gripper occlusion and sometimes include surrounding pixels. The pilot
is suitable for a limited A/B run only if the remaining annotation uncertainty
is accepted and reported. The encoded two-camera comparison sheet is
`outputs/carton_pilot/ab_dataset_preview.jpg`.

## Preserved data contract

Each derived dataset has exactly 1,841 frames in each 1280x720, 30 fps AV1 video.
For selected source episodes, `action`, `observation.state`, `timestamp`, and
`frame_index` arrays are byte-for-value equal after concatenation. Episode
lengths remain 659, 588, and 594. Derived episode indices are necessarily 0–2;
the source mapping 3, 20, 30 is preserved in `pilot_provenance.json`.

The source dataset and original checkpoint were read only. The visual codec is
AV1 rather than the source H.264 because that is LeRobot 0.6.1's local encoder
default; camera keys, shape, pixel format, fps, and decoding contract are
unchanged.

Both active `meta/stats.json` files are byte-identical copies of the original
40-episode statistics. Their SHA-256 is
`d5d3760857db874e55ad5e4973fa8de592601ef5aeb81713659a990129420a4b`.
The naturally computed three-episode statistics remain beside them as
`stats.empirical_3episode.json` for audit. During training LeRobot replaces the
camera entries with ImageNet values in memory; state/action means and standard
deviations then match the original checkpoint normalizer exactly at float32.

## One-step compatibility command

This exact command was run on CPU; it is a warm start, not resume:

```bash
/home/manashvi/miniforge3/envs/lerobot/bin/lerobot-train \
  --dataset.repo_id=so101_variable_box_3ep_highlight_pilot \
  --dataset.root=/home/manashvi/.cache/huggingface/lerobot/so101_variable_box_3ep_highlight_pilot \
  --dataset.video_backend=pyav \
  --policy.path=/home/manashvi/so101_outputs/variable_box_40episodes_act/checkpoints/050000/pretrained_model \
  --policy.device=cpu \
  --policy.push_to_hub=false \
  --output_dir=/home/manashvi/.cache/huggingface/lerobot/so101_hierarchical_project/outputs/act_highlight_smoke \
  --steps=1 \
  --batch_size=1 \
  --num_workers=0 \
  --log_freq=1 \
  --save_freq=1 \
  --wandb.enable=false
```

Result: forward/backward and AdamW update completed in 13.16 seconds with loss
0.390, L1 loss 0.389, KL loss 0.000, and finite gradient norm 39.779. LeRobot
saved model, processor, optimizer, RNG, and training-step state at checkpoint
`000001`. Reload with the saved processor returned a finite `(1, 6)` action.

LeRobot warm-start preprocessing intentionally recomputes state/action mean and
standard deviation from this three-episode pilot. Image normalization remains
the checkpoint's ImageNet mean/std exactly. This subset-stat shift is acceptable
for the compatibility smoke test but is not the intended final comparison. A
40-episode highlighted derivative retains identical state/action arrays and
therefore reproduces the original full-dataset state/action statistics.

## Matched A/B training commands—not yet run

Use the same seed, batch size, step budget, and all other optimizer settings for
both commands. The following recommended pilot budget is 2,500 steps (about
10.9 passes at batch size 8); adjust it only before running, and adjust both
arms together.

```bash
/home/manashvi/miniforge3/envs/lerobot/bin/lerobot-train \
  --dataset.repo_id=so101_variable_box_3ep_original_control \
  --dataset.root=/home/manashvi/.cache/huggingface/lerobot/so101_variable_box_3ep_original_control \
  --dataset.video_backend=pyav \
  --policy.path=/home/manashvi/so101_outputs/variable_box_40episodes_act/checkpoints/050000/pretrained_model \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=/home/manashvi/so101_outputs/variable_box_3ep_act_A_original \
  --steps=2500 --batch_size=8 --num_workers=4 --save_freq=500 \
  --wandb.enable=false

/home/manashvi/miniforge3/envs/lerobot/bin/lerobot-train \
  --dataset.repo_id=so101_variable_box_3ep_highlight_corrected \
  --dataset.root=/home/manashvi/.cache/huggingface/lerobot/so101_variable_box_3ep_highlight_corrected \
  --dataset.video_backend=pyav \
  --policy.path=/home/manashvi/so101_outputs/variable_box_40episodes_act/checkpoints/050000/pretrained_model \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=/home/manashvi/so101_outputs/variable_box_3ep_act_B_highlighted \
  --steps=2500 --batch_size=8 --num_workers=4 --save_freq=500 \
  --wandb.enable=false
```

These are warm starts, not resumes. Do not add `--resume=true`. Neither command
has been run.
