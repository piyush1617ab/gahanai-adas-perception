# Lane / Drivable-Area Dataset

The dataset used to fine-tune `../../lane_segmentation/models/twinlitenetplus_large.pth`.

## Overview

- ~994 images with paired binary segmentation masks: drivable area and
  lane lines (the two output heads of TwinLiteNetPlus).
- Split roughly 795 / 99 / 100 (train / val / test).
- CVAT-annotated, following the same general pre-annotation workflow as
  the object-detection dataset.

The dataset itself is not included in this repository — see the root
[`datasets/README.md`](../README.md) for why, and get in touch for access
terms if you need it for reproduction.

## Expected directory layout

TwinLiteNet's own data loader (`DataSet.py` in the upstream repo — see
[`../../lane_segmentation/README.md`](../../lane_segmentation/README.md))
expects paired image/mask folders:

```text
lane_detection/
├── train/
│   ├── images/
│   ├── segments/          # drivable-area binary masks
│   └── lane/              # lane-line binary masks
├── val/
│   └── ...
└── test/
    └── ...
```

## Reproducing training

Clone the upstream TwinLiteNet repo, point its `DataSet.py` /
`train.py` at your local copy of this dataset laid out as above, and train
starting from the "Large" configuration. Export the resulting checkpoint
with [`../../lane_segmentation/scripts/export_onnx.py`](../../lane_segmentation/scripts/export_onnx.py).
