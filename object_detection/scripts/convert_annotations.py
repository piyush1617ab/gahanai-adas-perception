#!/usr/bin/env python3
"""
Convert CVAT-XML (or JSON) annotations to YOLO format.

Maps the 17 raw CVAT pre-annotation labels down to the final 12-class
detection taxonomy actually used for training (see ../configs/data.yaml).
Five raw labels are intentionally dropped:
  - "lane markings" / "road boundaries" -> segmentation-style labels,
    out of scope for a box detector (handled instead by the lane
    segmentation model, see ../../lane_segmentation/).
  - "animal" / "Emergency vehicle" -> too sparse in the source data to
    train a reliable class; folded into "other" is possible but was
    excluded rather than diluting "other" further.
  - "Image" -> stray CVAT bookkeeping label, not a real object class.

Usage:
    python convert_annotations.py /path/to/dataset_root --format cvat
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

# Final 12-class taxonomy (verified against object_detection/configs/data.yaml
# and against the class names embedded in the shipped checkpoints).
CLASS_NAMES = {
    'person': 0,
    'rider': 1,
    'bicycle': 2,
    'motorcycle': 3,
    'auto_rickshaw': 4,
    'car': 5,
    'bus': 6,
    'truck': 7,
    'traffic_light': 8,
    'other': 9,
    'traffic_sign': 10,
    'number_plate': 11,
}

# Raw CVAT labels use spaces / mixed case; normalize before lookup.
# Labels not present here (lane markings, road boundaries, animal,
# emergency vehicle, image) are intentionally excluded — see module
# docstring.
_LABEL_ALIASES = {
    'auto rickshaw': 'auto_rickshaw',
    'traffic light': 'traffic_light',
    'traffic sign': 'traffic_sign',
    'number plate': 'number_plate',
}


def _normalize_label(label: str) -> str:
    key = label.strip().lower()
    return _LABEL_ALIASES.get(key, key.replace(' ', '_'))


def cvat_to_yolo(xml_file: Path, image_file: Path, output_txt: Path) -> bool:
    """Convert a single CVAT XML annotation to a YOLO-format label file."""
    if not image_file.exists():
        print(f"  [skip] image not found: {image_file}")
        return False

    img = cv2.imread(str(image_file))
    if img is None:
        print(f"  [skip] can't read image: {image_file}")
        return False
    img_height, img_width = img.shape[:2]

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  [error] malformed XML {xml_file}: {e}")
        return False

    yolo_lines = []
    for image_elem in root.findall('.//image'):
        image_name = image_elem.get('name')
        if image_name and image_name != image_file.name:
            continue

        for box in image_elem.findall('box'):
            raw_label = box.get('label', 'Unknown')
            label = _normalize_label(raw_label)

            if label not in CLASS_NAMES:
                continue  # intentionally excluded class — see docstring

            class_id = CLASS_NAMES[label]
            x1, y1 = float(box.get('xtl')), float(box.get('ytl'))
            x2, y2 = float(box.get('xbr')), float(box.get('ybr'))

            x_center = max(0.0, min(1.0, (x1 + x2) / 2 / img_width))
            y_center = max(0.0, min(1.0, (y1 + y2) / 2 / img_height))
            width = max(0.0, min(1.0, (x2 - x1) / img_width))
            height = max(0.0, min(1.0, (y2 - y1) / img_height))

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    output_txt.write_text('\n'.join(yolo_lines))
    return True


def json_to_yolo(json_file: Path, image_file: Path, output_txt: Path) -> bool:
    """Convert a JSON annotation ({'objects': [{'label', 'bbox'}]}) to YOLO format."""
    if not image_file.exists():
        print(f"  [skip] image not found: {image_file}")
        return False

    img = cv2.imread(str(image_file))
    if img is None:
        print(f"  [skip] can't read image: {image_file}")
        return False
    img_height, img_width = img.shape[:2]

    data = json.loads(json_file.read_text())
    yolo_lines = []
    for obj in data.get('objects', []):
        label = _normalize_label(obj.get('label', 'Unknown'))
        if label not in CLASS_NAMES:
            continue

        bbox = obj.get('bbox', [])
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = bbox[:4]

        x_center = max(0.0, min(1.0, (x1 + x2) / 2 / img_width))
        y_center = max(0.0, min(1.0, (y1 + y2) / 2 / img_height))
        width = max(0.0, min(1.0, (x2 - x1) / img_width))
        height = max(0.0, min(1.0, (y2 - y1) / img_height))

        class_id = CLASS_NAMES[label]
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    output_txt.write_text('\n'.join(yolo_lines))
    return True


def convert_dataset(dataset_root: Path, annotation_type: str) -> None:
    scene_folders = sorted(d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith('scene_'))
    total = converted = skipped = 0

    print(f"Converting {len(scene_folders)} scenes to YOLO format ({annotation_type})...\n")

    for scene_idx, scene_folder in enumerate(scene_folders):
        images_folder = scene_folder / 'images'
        source_folder = scene_folder / ('cvat for images 1.1' if annotation_type == 'cvat' else 'annotations')
        if not source_folder.exists():
            continue

        yolo_folder = scene_folder / 'YOLO'
        yolo_folder.mkdir(exist_ok=True)

        pattern = '*.xml' if annotation_type == 'cvat' else '*.json'
        for ann_file in source_folder.glob(pattern):
            total += 1
            image_stem = ann_file.stem
            image_file = next(
                (images_folder / f"{image_stem}{ext}" for ext in ('.jpg', '.jpeg', '.png')
                 if (images_folder / f"{image_stem}{ext}").exists()),
                None,
            )
            if image_file is None:
                skipped += 1
                print(f"  [skip] no image found for {ann_file.name}")
                continue

            output_txt = yolo_folder / f"{image_stem}.txt"
            fn = cvat_to_yolo if annotation_type == 'cvat' else json_to_yolo
            if fn(ann_file, image_file, output_txt):
                converted += 1

        if (scene_idx + 1) % 10 == 0:
            print(f"  Processed {scene_idx + 1}/{len(scene_folders)} scenes...")

    print("\n" + "=" * 70)
    print(f"Conversion complete — total: {total} | converted: {converted} | skipped: {skipped}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert CVAT/JSON annotations to YOLO format')
    parser.add_argument('dataset_root', help='Path to dataset root (contains scene_0000, scene_0001, ...)')
    parser.add_argument('--format', choices=['cvat', 'json'], default='cvat', help='Source annotation format')
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        print(f"Path not found: {root}")
        raise SystemExit(1)

    convert_dataset(root, args.format)
