# OpenADAS

The primary dataset behind the object-detection pipeline. Collected as ROS
bag recordings and split into per-scene folders; CVAT-annotated; converted
to YOLO format for training.

## What's in this folder

- `sample_frame.jpg` + `sample_frame_yolo_labels.txt` — one real annotated
  frame/label pair, in the exact format the training pipeline consumes.
- `sample_cvat_annotations.xml` — a sanitized 3-frame excerpt of a real
  CVAT export (task/owner metadata replaced with placeholders), showing the
  full 17-label raw taxonomy before reduction to 12 classes.
- `camera_info.json` — camera intrinsics/distortion for the recording rig
  (1920×1080, plumb_bob distortion model).
- `dataset_summary.json` — scene-collection metadata (58 scenes, 8,565
  total frames, 4,283 annotated at 3 fps).

The full dataset (images, complete annotation set) is **not** included —
see the root [`datasets/README.md`](../README.md).

## Expected directory layout

Each scene follows the same structure; this is what the scripts in
`../../object_detection/scripts/` expect:

```text
open_adas/
├── scene_0000/
│   ├── images/                    # frame_<n>_<timestamp>.jpg
│   ├── cvat for images 1.1/       # raw CVAT XML export, if using CVAT
│   ├── annotations/               # raw JSON export, if using JSON instead
│   └── YOLO/                      # generated YOLO .txt labels (output of convert_annotations.py)
├── scene_0001/
│   └── ...
└── ...
```

After conversion, `train.py` expects a flat `train/images`, `train/labels`,
`val/images`, `val/labels` split — produced by
`object_detection/scripts/merge_datasets.py` if you use the merge path, or
by any equivalent split script if you're using OpenADAS alone.

## Class taxonomy: 17 raw labels → 12 training classes

CVAT annotation used 17 labels. Five were dropped before training —
`lane markings` and `road boundaries` are segmentation concepts (out of
scope for a box detector; handled by the lane model instead), `animal` and
`Emergency vehicle` were too sparse to train reliably, and `Image` was a
stray CVAT bookkeeping label, not an object class.

| # | Final class | | # | Final class |
|---|---|---|---|---|
| 0 | person | | 6 | bus |
| 1 | rider | | 7 | truck |
| 2 | bicycle | | 8 | traffic_light |
| 3 | motorcycle | | 9 | other |
| 4 | auto_rickshaw | | 10 | traffic_sign |
| 5 | car | | 11 | number_plate |

This is the exact taxonomy baked into the shipped checkpoints — see
`../../object_detection/configs/data.yaml`.

## Processing pipeline

1. `scan_dataset.py` — inventory scenes, image counts, annotation coverage.
2. `convert_annotations.py` — CVAT XML / JSON → YOLO `.txt` labels, applying
   the 17→12 class reduction above.
3. `train.py` — fine-tune YOLO26 against the resulting dataset.

## Dataset policy

Do not commit raw OpenADAS images, full annotation exports, or any
Gahan AI-owned processed copies to this repository.
