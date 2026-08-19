#!/usr/bin/env python3
"""
Fine-tune a YOLO26 model (n/s/m) on the 12-class ADAS dataset.

The checkpoints shipped in ../models/ were produced with this pipeline:
  - yolo26s_best.pt         -> base model 'yolo26s.pt'
  - yolo26m_refined_best.pt -> base model 'yolo26m.pt', a second refinement
                                 pass on top of an initial run

Usage:
    python train.py --data ../configs/data.yaml --base-model yolo26s.pt --epochs 100
"""

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def print_gpu_info() -> bool:
    if torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("GPU")
        print("=" * 70)
        print(f"CUDA available: yes")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total memory: {total_memory:.1f} GB")
        print("=" * 70 + "\n")
        return True
    print("\nGPU not available — training will be very slow on CPU.\n")
    return False


def validate_data_yaml(yaml_path: Path) -> bool:
    try:
        data = yaml.safe_load(yaml_path.read_text())
    except OSError as e:
        print(f"Error reading data.yaml: {e}")
        return False

    missing = [k for k in ('path', 'train', 'val', 'nc', 'names') if k not in data]
    if missing:
        print(f"data.yaml missing keys: {missing}")
        return False

    print(f"data.yaml OK — {data['nc']} classes, dataset path: {data['path']}")
    return True


def train(data_yaml: Path, base_model: str, epochs: int, batch_size: int, imgsz: int,
          device: int, fast_mode: bool) -> object:
    print("\n" + "=" * 70)
    print(f"FINE-TUNING {base_model} ON THE 12-CLASS ADAS DATASET")
    print("=" * 70)

    if not data_yaml.exists():
        print(f"data.yaml not found: {data_yaml}")
        raise SystemExit(1)
    if not validate_data_yaml(data_yaml):
        raise SystemExit(1)

    print_gpu_info()

    print(f"Loading pretrained {base_model}...")
    model = YOLO(base_model)

    training_config = {
        'data': str(data_yaml),
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': device,
        'patience': 20,
        'save': True,
        'project': 'runs/detect',
        'name': f"{Path(base_model).stem}_adas",
        'exist_ok': False,
        'verbose': True,
        'plots': True,

        'optimizer': 'SGD',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,

        # Augmentation tuned for Indian road conditions (dust/monsoon
        # lighting, uneven road surfaces, dense mixed traffic).
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 15,
        'translate': 0.2,
        'scale': 0.5,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'close_mosaic': 15,
        'perspective': 0.0,
        'erasing': 0.0,

        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
    }

    if fast_mode:
        training_config.update(epochs=20, batch=min(batch_size, 16), imgsz=416)
        print("Fast mode: reduced epochs/batch/imgsz for a quick smoke test.")

    print("\nTraining config:")
    for key, value in training_config.items():
        if key != 'data':
            print(f"  {key:<16} {value}")

    results = model.train(**training_config)

    metrics = model.val()
    print("\nValidation metrics:")
    print(f"  mAP50:     {metrics.box.map50:.3f}")
    print(f"  mAP50-95:  {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall:    {metrics.box.mr:.3f}")

    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    print(f"\nBest checkpoint: {best_model_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune YOLO26 on the 12-class ADAS dataset')
    parser.add_argument('--data', required=True, type=Path, help='Path to data.yaml')
    parser.add_argument('--base-model', default='yolo26s.pt',
                         help="Pretrained base checkpoint, e.g. 'yolo26n.pt' / 'yolo26s.pt' / 'yolo26m.pt'")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--fast', action='store_true', help='Quick smoke-test run (20 epochs, small batch/imgsz)')
    args = parser.parse_args()

    train(args.data, args.base_model, args.epochs, args.batch_size, args.imgsz, args.device, args.fast)
