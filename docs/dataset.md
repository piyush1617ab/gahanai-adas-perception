# Dataset

Full details live in [`../datasets/README.md`](../datasets/README.md) and
its sub-READMEs — this page is a short pointer plus the two facts that
matter most when reading the rest of the docs.

## Object detection: OpenADAS, not BDD100K

The shipped YOLO26 checkpoints were trained on a custom dataset derived
from **OpenADAS** (58 scenes, 8,565 camera frames, 4,283 annotated at
3 fps). Annotation went through CVAT with an initial 17-label taxonomy,
reduced to a final **12 classes** for training — see
[`../datasets/open_adas/README.md`](../datasets/open_adas/README.md) for
the full class list and the reasoning behind each dropped label.

A BDD100K-merge path exists (`object_detection/scripts/merge_datasets.py`)
and works, but it's an optional augmentation extension, not what produced
the published checkpoints. See
[`../datasets/bdd100k/README.md`](../datasets/bdd100k/README.md).

## Lane segmentation: a custom paired-mask dataset

`lane_segmentation/models/twinlitenetplus_large.pth` was fine-tuned on
~994 images with paired drivable-area / lane-line segmentation masks
(795 / 99 / 100 train/val/test). See
[`../datasets/lane_detection/README.md`](../datasets/lane_detection/README.md).

## What's actually in this repo

Only small, sanitized samples — one annotated frame, a trimmed CVAT
export, camera calibration, and scene-collection metadata for object
detection. No raw dataset images or full annotation sets are redistributed
here (dataset size, licensing, and Gahan AI data-ownership all apply).
