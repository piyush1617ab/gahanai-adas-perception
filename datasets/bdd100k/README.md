# BDD100K (optional augmentation source)

BDD100K is **not** the dataset behind the shipped object-detection
checkpoints — those were trained on OpenADAS alone (see
[`../open_adas/README.md`](../open_adas/README.md)). This directory
documents an optional, working augmentation path that was built and tested
alongside the main pipeline.

## Why it's here

`object_detection/scripts/merge_datasets.py` can pull a fraction of
BDD100K's `train`/`val` split, remap its categories onto the same 12-class
taxonomy used for OpenADAS, and merge the two into one training set. It's
included because it's real, working code — useful if you want to extend
training with more volume/diversity — not because it describes how the
published models were produced.

## Category mapping

BDD100K has no equivalent for `auto_rickshaw`, `other`, `traffic_sign`
(as a distinct box class), or `number_plate`, so those stay OpenADAS-only.
Categories with no reasonable match (`train`, `other vehicle`, `other
person`, `trailer`) are skipped rather than force-mapped:

| BDD100K category | Mapped to |
|---|---|
| pedestrian | person |
| rider | rider |
| bicycle | bicycle |
| motorcycle | motorcycle |
| car | car |
| bus | bus |
| truck | truck |
| traffic light | traffic_light |
| traffic sign | traffic_sign |

## Getting the data

BDD100K is not redistributed here. Get your own copy from
[bdd-data.berkeley.edu](https://bdd-data.berkeley.edu/) and check its
license before use. Expected layout:

```text
bdd100k/
├── images/{train,val}/*.jpg
└── labels/det_{train,val}.json
```

## Usage

```bash
python object_detection/scripts/inspect_bdd100k_classes.py \
    /path/to/bdd100k/labels/det_train.json   # sanity-check category coverage

python object_detection/scripts/merge_datasets.py \
    --open-adas /path/to/open_adas \
    --bdd100k /path/to/bdd100k \
    --output ./merged_dataset \
    --bdd-ratio 0.3
```
