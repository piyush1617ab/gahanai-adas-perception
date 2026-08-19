# Evaluation

## Status

No benchmark numbers (mAP, IoU, FPS) from actual training runs are
recorded in this repository. The only performance figures found while
assembling it were in a planning document
(`object_detection/scripts/` history) explicitly labeled as illustrative
"expected results," not logged output from a real run — they've been
deliberately left out rather than published as if they were measured.
If you have real numbers from your training logs (Ultralytics writes
`results.csv` / `results.png` per run under `runs/detect/<name>/`), this
is the place to add them.

## How to reproduce metrics

### Object detection

`train.py` calls `model.val()` after training and prints:

```text
mAP50:      <value>
mAP50-95:   <value>
Precision:  <value>
Recall:     <value>
```

To re-evaluate an existing checkpoint against a validation set without
retraining:

```python
from ultralytics import YOLO
model = YOLO("object_detection/models/yolo26s_best.pt")
metrics = model.val(data="object_detection/configs/data.yaml")
print(metrics.box.map50, metrics.box.map)
```

For real-world sanity checks rather than aggregate metrics, run
`object_detection/scripts/inference.py` against the bundled
`object_detection/test_images/` and inspect the annotated outputs directly.

### Lane segmentation

The upstream TwinLiteNet repo ships its own `val.py` and `IOUEval.py` for
computing per-class IoU on drivable-area and lane-line predictions — run
that against `models/twinlitenetplus_large.pth` and your local copy of the
lane dataset (see
[`../lane_segmentation/README.md`](../lane_segmentation/README.md) for the
clone step).

## What to check before trusting a number

- **mAP without a held-out val set** isn't meaningful — confirm
  `configs/data.yaml`'s `val` split doesn't overlap `train`.
- **Class imbalance** — `other`, `number_plate`, and `auto_rickshaw` are
  likely far less frequent than `car`/`person` in most driving footage;
  a high overall mAP can hide a weak minority class. Check per-class AP,
  not just the aggregate.
- **FPS numbers are hardware- and precision-specific** — a TensorRT FP16
  engine on Jetson will differ substantially from a PyTorch `.pt` on a
  desktop GPU. State the hardware and precision alongside any latency
  figure you report.
