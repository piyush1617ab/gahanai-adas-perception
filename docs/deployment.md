# Deployment

Target hardware: NVIDIA Jetson Orin NX, running both models side-by-side.

## Object detection: YOLO26 → ONNX → TensorRT

```bash
python object_detection/scripts/export_tensorrt.py \
    --input object_detection/models/yolo26s_best.pt
```

This runs diagnostics (available RAM → safe TensorRT builder workspace
size, CUDA availability, locates `trtexec`), exports to ONNX via
Ultralytics, then compiles an FP16 TensorRT engine. It auto-detects
TensorRT 10+ vs. legacy flag syntax from `trtexec --version`, so it should
work across JetPack 5.x and 6.x without editing the script.

Status: this export path runs the real model end-to-end (not a stub) and
has previously produced valid engines during the internship. Re-verify
engine build time and inference latency on your specific JetPack version
before treating any number as current.

## Lane segmentation: TwinLiteNetPlus → ONNX

```bash
python lane_segmentation/scripts/export_onnx.py \
    --weights lane_segmentation/models/twinlitenetplus_large.pth \
    --output twinlitenetplus_large.onnx
```

Status: the export script is real, working code — the architecture is
reconstructed to match the checkpoint's `state_dict` and loaded with
`strict=True`, so a shape mismatch fails loudly rather than silently. It
has **not** been benchmark-verified end-to-end on Jetson in this repo. A
separate earlier export attempt produced a non-functional ONNX file that
returned random noise instead of real predictions — that script and its
output were left out of this repo rather than published. Once you've
confirmed `export_onnx.py` produces a working engine on your hardware,
compiling it to TensorRT is the same `trtexec` flow used for the detector
(see `export_tensorrt.py` — it works from any `.onnx` input, not just
YOLO's).

## Running both models together

Both models are independent (no shared weights or preprocessing beyond
"take a camera frame"). A minimal Jetson inference loop would:

1. Grab a frame from the camera pipeline.
2. Run the YOLO26 TensorRT engine → boxes + classes.
3. Run the TwinLiteNetPlus TensorRT engine → drivable-area + lane-line
   masks.
4. Fuse/overlay both outputs for the downstream ADAS logic or visualization.

No fusion/overlay code is included here yet — this repo covers training,
export, and single-model inference for each pipeline independently.
