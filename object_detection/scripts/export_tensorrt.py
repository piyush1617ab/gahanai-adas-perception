#!/usr/bin/env python3
"""
Export a trained YOLO26 checkpoint to ONNX, then compile it into a
TensorRT engine for Jetson deployment.

Requires an NVIDIA Jetson (or any machine with `trtexec` on PATH / at
/usr/src/tensorrt/bin/trtexec) and a CUDA-enabled PyTorch install for the
ONNX export step.

Usage:
    python export_tensorrt.py --input ../models/yolo26s_best.pt
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import psutil
import torch


def run_diagnostics(need_cuda: bool) -> tuple[str, int, str]:
    print("\n[1/3] Environment diagnostics")

    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"  RAM: {total_ram_gb:.2f} GB total")

    # Cap the TensorRT builder workspace at 25% of system RAM (or 4 GiB,
    # whichever is smaller) to avoid OOM on memory-constrained Jetson boards.
    safe_workspace_mb = int(min(4096, (total_ram_gb * 0.25) * 1024))
    print(f"  Builder workspace budget: {safe_workspace_mb} MiB")

    if need_cuda:
        if not torch.cuda.is_available():
            print("  CUDA is not available to PyTorch — required for the ONNX export step.")
            sys.exit(1)
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

    trtexec_path = "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec_path):
        trtexec_path = shutil.which("trtexec")
        if not trtexec_path:
            print("  'trtexec' not found — verify your JetPack / TensorRT installation.")
            sys.exit(1)
    print(f"  trtexec: {trtexec_path}")

    try:
        version_info = subprocess.check_output([trtexec_path, "--version"], stderr=subprocess.STDOUT).decode()
    except (subprocess.CalledProcessError, OSError):
        version_info = ""

    return trtexec_path, safe_workspace_mb, version_info


def export_to_onnx(input_path: Path, onnx_path: Path, imgsz: int) -> None:
    print("\n[2/3] Exporting to ONNX")
    if not input_path.exists():
        print(f"  Source checkpoint not found: {input_path}")
        sys.exit(1)

    from ultralytics import YOLO
    model = YOLO(str(input_path))
    model.export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False)

    default_output = input_path.with_suffix(".onnx")
    if default_output.exists() and default_output != onnx_path:
        shutil.move(str(default_output), str(onnx_path))
    if not onnx_path.exists():
        print(f"  Expected ONNX output not found at {onnx_path}")
        sys.exit(1)
    print(f"  ONNX graph written: {onnx_path}")


def compile_engine(trtexec_bin: str, onnx_path: Path, engine_path: Path,
                    workspace_mb: int, version_info: str, fp16: bool) -> None:
    print("\n[3/3] Compiling TensorRT engine")
    if not onnx_path.exists():
        print(f"  ONNX file not found: {onnx_path}")
        sys.exit(1)

    cmd = [trtexec_bin, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    if fp16:
        cmd.append("--fp16")

    # TensorRT 10+ uses --memPoolSize / --stronglyTyped; earlier versions use --workspace.
    is_modern_trt = ("10." in version_info) or ("11." in version_info) or not version_info
    if is_modern_trt:
        cmd += [f"--memPoolSize=workspace:{workspace_mb}MiB", "--stronglyTyped"]
    else:
        cmd.append(f"--workspace={workspace_mb}")

    print(f"  Command: {' '.join(cmd)}")
    print("  Compiling (this profiles GPU kernels and can take several minutes)...")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        if any(k in line for k in ("Targeting", "Completed", "Engine built", "Error")):
            print(f"    {line.strip()}")
    process.wait()

    if process.returncode == 0 and engine_path.exists():
        size_mb = engine_path.stat().st_size / (1024 ** 2)
        print(f"\nEngine written: {engine_path} ({size_mb:.2f} MB)")
    else:
        print(f"\ntrtexec exited with code {process.returncode}")
        sys.exit(process.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a YOLO26 checkpoint to a TensorRT engine")
    parser.add_argument("--input", required=True, type=Path,
                         help="Path to a .pt checkpoint, or an existing .onnx file to skip straight to compilation")
    parser.add_argument("--engine-out", type=Path, default=None, help="Output .engine path (default: alongside input)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 (use full precision)")
    args = parser.parse_args()

    input_path: Path = args.input
    is_pytorch = input_path.suffix == ".pt"

    if is_pytorch:
        onnx_path = input_path.with_suffix(".onnx")
    elif input_path.suffix in (".onnx", ".xml"):
        onnx_path = input_path
    else:
        print("Input must be a .pt or .onnx file.")
        sys.exit(1)

    engine_path = args.engine_out or input_path.with_suffix(".engine")

    trt_bin, workspace_mb, version_str = run_diagnostics(need_cuda=is_pytorch)
    if is_pytorch:
        export_to_onnx(input_path, onnx_path, args.imgsz)
    compile_engine(trt_bin, onnx_path, engine_path, workspace_mb, version_str, fp16=not args.no_fp16)
