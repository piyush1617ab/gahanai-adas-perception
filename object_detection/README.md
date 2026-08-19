# Object Detection

YOLO26, fine-tuned on a custom 12-class taxonomy for Indian road scenes:
`person, rider, bicycle, motorcycle, auto_rickshaw, car, bus, truck,
traffic_light, other, traffic_sign, number_plate`.

## Models

| File | Base model | Notes |
|---|---|---|
| `models/yolo26s_best.pt` | YOLO26s | 20 MB |
| `models/yolo26m_refined_best.pt` | YOLO26m | 175 MB, second-pass refinement run (`production_12classes_26m_refinement`) |

Both are tracked via **Git LFS** — run `git lfs pull` after cloning to fetch
them.

```python
from ultralytics import YOLO
model = YOLO("object_detection/models/yolo26s_best.pt")
results = model.predict("path/to/image.jpg")
```

## Directory layout

```text
object_detection/
├── configs/
│   └── data.yaml              # the actual 12-class training config
├── scripts/
│   ├── scan_dataset.py        # inventory a scene-structured dataset
│   ├── convert_annotations.py # CVAT/JSON -> YOLO labels (17 -> 12 classes)
│   ├── merge_datasets.py      # optional: fold in a BDD100K subset
│   ├── inspect_bdd100k_classes.py
│   ├── train.py               # fine-tune YOLO26 (n/s/m)
│   ├── inference.py           # run a checkpoint on images/video/camera
│   └── export_tensorrt.py     # .pt -> ONNX -> TensorRT engine (Jetson)
├── models/                    # trained checkpoints (Git LFS)
└── test_images/                # a handful of sample frames for smoke-testing inference
```

## Pipeline

1. **Annotate** — CVAT, 17 raw labels (see
   [`../datasets/open_adas/README.md`](../datasets/open_adas/README.md)
   for the full list and how it maps down to 12).
2. **Scan** — `scan_dataset.py` inventories scene folders, image counts,
   and annotation coverage.
3. **Convert** — `convert_annotations.py` turns CVAT XML or JSON into YOLO
   `.txt` labels, dropping the 5 out-of-scope raw labels along the way.
4. **(Optional) Merge** — `merge_datasets.py` can fold in a subset of
   BDD100K remapped onto the same 12 classes. The shipped checkpoints were
   trained on OpenADAS alone; this is a documented, working extension, not
   what produced the published weights.
5. **Train** — `train.py --data configs/data.yaml --base-model yolo26s.pt`.
   Augmentation is tuned for Indian road conditions: wide hue/value jitter
   for dust and monsoon lighting, rotation and translation for uneven road
   surfaces.
6. **Run inference** — `inference.py --model models/yolo26s_best.pt --source test_images --type image`.
7. **Deploy** — `export_tensorrt.py` exports to ONNX then compiles a
   TensorRT engine for Jetson.

## Quick start

```bash
pip install -r ../requirements.txt

# Smoke-test inference on the bundled sample images
python scripts/inference.py \
    --model models/yolo26s_best.pt \
    --source test_images \
    --type image
```

## A note on the scripts here

Earlier drafts of some of these scripts (found while assembling this repo)
hardcoded a different, 13-class taxonomy that doesn't match what's actually
baked into the trained checkpoints — apparently an earlier planning pass
that got superseded. The versions here have been corrected to the verified
12-class scheme (cross-checked directly against the class names embedded in
`models/*.pt`, not just against documentation).
