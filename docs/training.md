# Training

## Object detection — YOLO26

```bash
python object_detection/scripts/train.py \
    --data object_detection/configs/data.yaml \
    --base-model yolo26s.pt \
    --epochs 100 \
    --batch-size 32 \
    --imgsz 640
```

Key settings (see `object_detection/scripts/train.py` for the full
config):

- **Optimizer**: SGD, `lr0=0.01`, `lrf=0.01`, momentum 0.937, weight decay
  5e-4, 3-epoch warmup.
- **Augmentation**, tuned for Indian road conditions rather than left at
  YOLO defaults:
  - `hsv_v=0.4` — wide brightness jitter for dust/monsoon lighting variation.
  - `degrees=15`, `translate=0.2` — rotation/translation for uneven road
    surfaces and camera mounting variation.
  - `mosaic=1.0`, `close_mosaic=15` — mosaic augmentation for most of
    training, disabled for the last 15 epochs to stabilize convergence.
  - `perspective=0.0`, `erasing=0.0` — deliberately skipped for training
    speed.
- **Loss weights**: `box=7.5`, `cls=0.5`, `dfl=1.5` (YOLO defaults).

`--base-model` accepts `yolo26n.pt` / `yolo26s.pt` / `yolo26m.pt`. The
shipped checkpoints used `yolo26s.pt` and `yolo26m.pt`; `yolo26m_refined_best.pt`
is a second pass on top of an initial YOLO26m run
(`production_12classes_26m` → `production_12classes_26m_refinement`).

`--fast` runs a 20-epoch / small-batch / 416px smoke test to sanity-check
the pipeline before committing to a full run.

## Lane segmentation — TwinLiteNetPlus (Large)

Training itself happens in the upstream
[TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet) repo, not in this
one — see [`../lane_segmentation/README.md`](../lane_segmentation/README.md)
for why, and for the exact clone + fine-tune steps. In short: clone
upstream, point its `DataSet.py` at the dataset described in
[`../datasets/lane_detection/README.md`](../datasets/lane_detection/README.md),
select the "Large" model configuration, and fine-tune from there.

## Hardware

Training was run on a remote Ubuntu machine (RTX A4000). Both YOLO26
variants and the TwinLiteNetPlus fine-tune fit comfortably within a single
16 GB GPU at the batch sizes above.
