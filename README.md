# Gahan AI — ADAS Perception

An ADAS perception stack built for Indian road conditions during an ML
Engineer internship at Gahan AI (Open ADAS project): a 12-class object
detector and a lane/drivable-area segmentation model, both fine-tuned on
custom, CVAT-annotated datasets and exportable to TensorRT for Jetson
deployment.

## What's here

| | Model | Task |
|---|---|---|
| [`object_detection/`](object_detection/README.md) | YOLO26 (s / m) | 12-class detection: people, vehicles, traffic infrastructure |
| [`lane_segmentation/`](lane_segmentation/README.md) | TwinLiteNetPlus (Large) | Drivable-area + lane-line segmentation |

Both pipelines include the full path from raw CVAT annotations to a
trained, exportable checkpoint — not just the final weights.

```text
gahanai-adas-perception/
├── object_detection/
│   ├── configs/data.yaml          # the real 12-class training config
│   ├── scripts/                   # scan → convert → (merge) → train → infer → export
│   ├── models/                    # yolo26s_best.pt, yolo26m_refined_best.pt  (Git LFS)
│   └── test_images/               # sample frames for smoke-testing inference
├── lane_segmentation/
│   ├── docs/design_report.pdf     # GhostNet+FPN architecture R&D — see note below
│   ├── scripts/export_onnx.py
│   └── models/twinlitenetplus_large.pth  (Git LFS)
├── datasets/                      # dataset docs + small, sanitized samples (no raw data)
├── docs/                          # architecture, training, evaluation, deployment, internship summary
└── assets/                        # sample images / diagrams for documentation
```

## Quick start

```bash
git clone https://github.com/piyush1617ab/gahanai-adas-perception.git
cd gahanai-adas-perception
git lfs pull                       # fetch the model checkpoints
pip install -r requirements.txt

# Object detection on the bundled sample images
python object_detection/scripts/inference.py \
    --model object_detection/models/yolo26s_best.pt \
    --source object_detection/test_images \
    --type image
```

## Object detection: 12 classes

`person, rider, bicycle, motorcycle, auto_rickshaw, car, bus, truck,
traffic_light, other, traffic_sign, number_plate`

Trained on a custom dataset derived from **OpenADAS** (58 scenes, 8,565
frames, 4,283 CVAT-annotated at 3 fps) — not a BDD100K merge; an optional,
working BDD100K-augmentation script is included but isn't what produced
the published checkpoints. See
[`docs/dataset.md`](docs/dataset.md) for the full story, including how the
class set was reduced from CVAT's original 17 raw labels.

## Lane segmentation: TwinLiteNetPlus, plus a separate design study

The shipped, fine-tuned model is **TwinLiteNetPlus (Large)**, built on top
of the open-source [TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet)
training pipeline. Alongside it, `lane_segmentation/docs/design_report.pdf`
documents a more ambitious from-scratch design (GhostNet-1.3 + FPN + 5 task
heads, with a full literature survey) — real R&D work, but a design
document rather than a second trained model. See
[`lane_segmentation/README.md`](lane_segmentation/README.md) for exactly
how the two relate.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — both models in detail
- [`docs/dataset.md`](docs/dataset.md) — dataset sourcing and class taxonomy
- [`docs/training.md`](docs/training.md) — hyperparameters and reproduction steps
- [`docs/evaluation.md`](docs/evaluation.md) — how to measure it yourself (no fabricated numbers here)
- [`docs/deployment.md`](docs/deployment.md) — Jetson / TensorRT export, including known limitations
- [`docs/internship_summary.md`](docs/internship_summary.md) — project context and honest scope notes

## Status / license

Private, work-in-progress repository — not licensed for external use or
redistribution at this time.
