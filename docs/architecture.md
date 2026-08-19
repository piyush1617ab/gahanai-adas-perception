# Architecture

Two independent perception models, meant to run side-by-side on a Jetson
Orin NX: a box detector and a segmentation network.

## Object detection — YOLO26

Standard YOLO26 (Ultralytics), fine-tuned from COCO-pretrained weights on
the 12-class ADAS taxonomy. Two variants were trained:

- **YOLO26s** (`object_detection/models/yolo26s_best.pt`) — the smaller,
  faster variant.
- **YOLO26m** (`object_detection/models/yolo26m_refined_best.pt`) — a
  second, refinement pass (run name: `production_12classes_26m_refinement`)
  on the medium variant, trading inference speed for accuracy.

No architectural changes were made to the base YOLO26 network — the work
here is in the dataset (17→12 class taxonomy, CVAT pre-annotation pipeline)
and training configuration (augmentation tuned for Indian road conditions:
wide hue/value jitter for dust and monsoon lighting, rotation/translation
for uneven road surfaces). See
[`../object_detection/README.md`](../object_detection/README.md) and
[`training.md`](training.md) for specifics.

## Lane / drivable-area segmentation — TwinLiteNetPlus (Large)

A dual-head encoder-decoder segmentation network:

- **Shared encoder**: a sequence of `CBR` (conv-BN-PReLU) downsampling
  stages feeding into `ESPBlockLarge` stages — efficient spatial pyramid
  blocks with five parallel dilated depthwise-separable branches (dilation
  rates 1/2/4/8/16), concatenated and fused. Multi-scale downsampled copies
  of the input (`x/2`, `x/4`, `x/8`) are concatenated in at each stage,
  following the original TwinLiteNet design.
- **PCAA attention block**: a position/channel attention module (query/key/value
  self-attention plus a 7×1 + 1×7 factorized conv branch) applied to the
  final encoder features before the two heads split off.
- **Two decoder heads**, each a pair of `TasksUpsampleBlock`s (transposed
  conv + residual conv refinement) followed by an `OutputBlock`:
  - `da` head → 2-channel drivable-area segmentation
  - `ll` head → 2-channel lane-line segmentation

This was fine-tuned from the upstream
[TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet) training
pipeline (MIT licensed), using its "Large" configuration. See
[`../lane_segmentation/README.md`](../lane_segmentation/README.md) for the
full picture, including why the architecture is reconstructed by hand in
`export_onnx.py` rather than imported directly.

## A separate design track: GhostNet + FPN

Alongside the TwinLiteNetPlus work, `lane_segmentation/docs/design_report.pdf`
documents a from-scratch architecture proposal: a GhostNet-1.3 backbone
with an FPN neck and five task heads (drivable area, lane lines, lane type
classification, and two auxiliary heads), plus a literature survey of
SCNN/UFLD/CLRNet/GANet-style approaches. This is real design work, but no
training code for it exists in this repo — treat it as the R&D behind the
lane-detection direction, not as a second trained model.

## Why two separate models instead of one multi-task network

Object detection (boxes, discrete classes) and lane/drivable-area
segmentation (dense masks) have different output structures and loss
landscapes. Running them as two focused models — rather than forcing both
into one multi-task head — kept each pipeline simpler to iterate on
independently during the internship, at the cost of two forward passes
instead of one on the target hardware.
