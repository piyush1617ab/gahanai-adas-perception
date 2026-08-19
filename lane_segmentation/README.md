# Lane & Drivable-Area Segmentation

A dual-head segmentation model — drivable area + lane lines — fine-tuned
for Indian road scenes.

## What's actually implemented and trained

**TwinLiteNetPlus (Large configuration)**, fine-tuned starting from the
upstream [TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet) training
pipeline (Che et al., MIT licensed). The checkpoint is in
`models/twinlitenetplus_large.pth` (Git LFS — run `git lfs pull` after
cloning).

```text
lane_segmentation/
├── docs/
│   └── design_report.pdf      # architecture R&D report — see note below
├── scripts/
│   └── export_onnx.py         # .pth -> ONNX
└── models/
    └── twinlitenetplus_large.pth
```

### Reproducing training

This repo doesn't vendor a full copy of the upstream TwinLiteNet training
code (it's a separate MIT-licensed project — clone it directly rather than
duplicating it here):

```bash
git clone https://github.com/chequanghuy/TwinLiteNet.git
cd TwinLiteNet
# point DataSet.py / train.py at your local lane dataset
# (see ../datasets/lane_detection/README.md), select the "Large" config
```

### Export

```bash
python scripts/export_onnx.py \
    --weights models/twinlitenetplus_large.pth \
    --output twinlitenetplus_large.onnx
```

`export_onnx.py` reconstructs the Large architecture layer-for-layer to
match the checkpoint's `state_dict` (the upstream repo's own module for
this exact config wasn't importable in this setup, so the class definitions
here are a manual, verified reconstruction — loaded with `strict=True`, so
any shape mismatch would fail loudly rather than silently). This has not
yet been benchmark-verified end-to-end on Jetson hardware. An earlier
export attempt in this project produced a non-functional ONNX file (a stub
that returned random noise instead of running the network) — that script
and its output have been removed rather than published.

## The design report — a separate, more ambitious document

`docs/design_report.pdf` is a from-scratch architecture design: a
GhostNet-1.3 + FPN backbone with 5 task heads, including a literature
survey of SCNN/UFLD/CLRNet/GANet-style lane detectors. It's real,
substantial R&D work — but it's a **design document**, not a description
of the trained model above. No training code for that architecture exists
in this repo. Read it as the research/planning behind the lane-detection
work, with TwinLiteNetPlus as the model that actually got fine-tuned,
exported, and shipped within the internship timeline.

## Dataset

~994 images with paired drivable-area / lane-line masks. See
[`../datasets/lane_detection/README.md`](../datasets/lane_detection/README.md).
