# Internship Summary

ML Engineer internship at **Gahan AI Pvt. Ltd.**, on the **Open ADAS**
project — building a perception stack (object detection + lane/drivable-area
segmentation) for Indian road conditions.

## What this repo covers

- **Object detection**: YOLO26 (s and m variants) fine-tuned on a
  custom 12-class taxonomy, built from a CVAT-annotated OpenADAS-derived
  dataset. Includes the full dataset-engineering pipeline — scanning,
  CVAT→YOLO conversion, an optional BDD100K-merge extension, training, and
  Jetson TensorRT export.
- **Lane / drivable-area segmentation**: TwinLiteNetPlus (Large
  configuration), fine-tuned from the upstream open-source TwinLiteNet
  pipeline, plus a from-scratch GhostNet+FPN architecture design report
  (literature survey + proposed multi-task design) documenting a more
  ambitious direction that was researched alongside it.
- **CVAT-based pre-annotation pipeline** — the annotation workflow behind
  both datasets: initial CVAT labeling, format conversion, and class-set
  refinement (17 raw labels down to 12 for object detection, after
  removing segmentation-style and too-sparse classes).

## Honest scope notes

A few things worth stating plainly, since they came up while assembling
this repo from the original project files:

- The object-detection class taxonomy went through at least one earlier
  draft (a different 13-class scheme) before settling on the 12-class set
  actually used for training — see
  [`../object_detection/README.md`](../object_detection/README.md).
- The lane-segmentation GhostNet+FPN design report describes a different,
  more ambitious architecture than the TwinLiteNetPlus model that was
  actually fine-tuned and shipped — see
  [`../lane_segmentation/README.md`](../lane_segmentation/README.md) for
  how the two relate.
- No benchmark metrics from real training runs were available to include
  — see [`evaluation.md`](evaluation.md).

## Repo structure

```text
gahanai-adas-perception/
├── object_detection/    # YOLO26 pipeline: data, training, inference, export
├── lane_segmentation/   # TwinLiteNetPlus pipeline + design report
├── datasets/            # dataset docs + small samples (no raw data)
├── docs/                # this folder
└── assets/              # sample images / diagrams for documentation
```
