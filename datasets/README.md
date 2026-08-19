# Datasets

This directory documents the datasets used across both pipelines and ships
small, non-sensitive **samples** so the annotation format and directory
layout are self-explanatory. The full datasets are not included — see
"Data availability" below.

## Contents

| Directory | Used by | What's actually here |
|---|---|---|
| [`open_adas/`](open_adas/README.md) | Object detection | Real sample: one annotated frame + label, a sanitized 3-frame CVAT export, camera calibration, and scene-collection metadata |
| [`bdd100k/`](bdd100k/README.md) | Object detection (optional) | Documentation only — see note below |
| [`lane_detection/`](lane_detection/README.md) | Lane segmentation | Documentation of the custom lane dataset used to fine-tune TwinLiteNetPlus |

## Object detection dataset, in one paragraph

The shipped object-detection checkpoints (`../object_detection/models/`)
were trained on a custom dataset derived from **OpenADAS** — 58 recorded
scenes (8,565 camera frames total, 4,283 annotated at 3 fps), CVAT-annotated
by hand with a 17-label taxonomy, then reduced to the final **12-class**
set used for training after dropping five labels that didn't belong in a
box detector (`lane markings`, `road boundaries` — segmentation concepts;
`animal`, `Emergency vehicle` — too sparse; `Image` — a stray CVAT
bookkeeping label). See [`open_adas/README.md`](open_adas/README.md) for
the exact class list and directory layout.

**BDD100K** is not the source of the shipped models. `object_detection/scripts/merge_datasets.py`
implements a working, documented path for augmenting OpenADAS with a
BDD100K subset (remapped onto the same 12 classes) — it's included because
it's real, tested code, not because the published checkpoints depend on it.

## Lane segmentation dataset, in one paragraph

`../lane_segmentation/models/twinlitenetplus_large.pth` was fine-tuned on a
custom dataset of ~994 images (795 train / 99 val / 100 test) with paired
drivable-area and lane-line segmentation masks, starting from the upstream
TwinLiteNet training pipeline. See
[`lane_detection/README.md`](lane_detection/README.md).

## Data availability

Raw dataset images, full annotation exports, and any Gahan AI-owned data
are **not** included here — only small samples needed to understand the
format, plus dataset-processing scripts that work against your own local
copy. Before downloading OpenADAS, BDD100K, or any third-party dataset,
check its license and redistribution terms yourself.

## Reproducing training

1. Obtain the dataset from its authorized source and lay it out per the
   relevant sub-README.
2. Run `object_detection/scripts/scan_dataset.py` to inventory it.
3. Run `object_detection/scripts/convert_annotations.py` to get YOLO-format
   labels if you're starting from CVAT XML.
4. Point `object_detection/configs/data.yaml` at your local dataset root
   and train with `object_detection/scripts/train.py`.
