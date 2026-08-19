#!/usr/bin/env python3
"""
Optional dataset-augmentation path: merge the custom OpenADAS scene dataset
with a subset of BDD100K, remapped onto the same 12-class taxonomy.

NOTE ON PROVENANCE: this script was built and tested while exploring whether
BDD100K could add useful volume/diversity on top of the OpenADAS data. The
checkpoints shipped in ../models/ were trained on the OpenADAS-derived
dataset directly (see ../configs/data.yaml) — this merge path is kept here
as a documented, working option for anyone who wants to extend training
with BDD100K, not as a description of how the shipped models were produced.

BDD100K is not redistributed in this repo; point --bdd100k at your own
local copy (https://bdd-data.berkeley.edu/).

Usage:
    python merge_datasets.py --open-adas /path/to/open_adas \
        --bdd100k /path/to/bdd100k --output ./merged_dataset
"""

import argparse
import json
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Final 12-class taxonomy — must match ../configs/data.yaml exactly.
CLASSES = {
    0: 'person', 1: 'rider', 2: 'bicycle', 3: 'motorcycle', 4: 'auto_rickshaw',
    5: 'car', 6: 'bus', 7: 'truck', 8: 'traffic_light', 9: 'other',
    10: 'traffic_sign', 11: 'number_plate',
}

# BDD100K detection categories -> our class ids. BDD100K has no
# auto-rickshaw, traffic-sign-as-box, or number-plate equivalent, so those
# stay OpenADAS-only. Categories with no reasonable match (train, other
# vehicle/person, trailer) are skipped rather than force-mapped.
BDD100K_TO_CLASSES = {
    'pedestrian': 0,   # person
    'rider': 1,        # rider
    'bicycle': 2,
    'motorcycle': 3,
    'car': 5,
    'bus': 6,
    'truck': 7,
    'traffic light': 8,
    'traffic sign': 10,
}


def copy_open_adas_scenes(open_adas_root: Path, output_images: Path, output_labels: Path,
                           keep_ratio: float = 1.0) -> dict:
    """Copy OpenADAS scene images + YOLO labels into a flat merged pool."""
    stats = {'images_copied': 0, 'labels_copied': 0, 'missing_labels': 0, 'corrupted': 0}
    scene_folders = sorted(d for d in open_adas_root.iterdir() if d.is_dir() and d.name.startswith('scene_'))

    logger.info(f"Processing {len(scene_folders)} OpenADAS scenes...")

    for scene_idx, scene_folder in enumerate(scene_folders):
        images_folder = scene_folder / 'images'
        yolo_folder = scene_folder / 'YOLO'
        if not images_folder.exists() or not yolo_folder.exists():
            continue

        image_files = list(images_folder.glob('*.jpg')) + list(images_folder.glob('*.png'))
        if keep_ratio < 1.0:
            image_files = random.sample(image_files, int(len(image_files) * keep_ratio))

        for img_file in image_files:
            unique_name = f"{scene_folder.name}_{img_file.name}"
            dest_img = output_images / unique_name
            try:
                shutil.copy2(img_file, dest_img)
                stats['images_copied'] += 1
            except OSError as e:
                logger.warning(f"  failed to copy {img_file}: {e}")
                stats['corrupted'] += 1
                continue

            label_file = yolo_folder / f"{img_file.stem}.txt"
            if label_file.exists():
                dest_label = output_labels / f"{unique_name.replace(img_file.suffix, '.txt')}"
                shutil.copy2(label_file, dest_label)
                stats['labels_copied'] += 1
            else:
                stats['missing_labels'] += 1

        if (scene_idx + 1) % 10 == 0:
            logger.info(f"  processed {scene_idx + 1}/{len(scene_folders)} scenes...")

    return stats


def process_bdd100k(bdd100k_root: Path, output_images: Path, output_labels: Path,
                     use_fraction: float = 0.3, split: str = 'train') -> dict:
    """
    Remap a fraction of BDD100K detection labels onto the 12-class taxonomy.

    Expects the standard BDD100K layout:
        bdd100k/images/{train,val}/*.jpg
        bdd100k/labels/det_{train,val}.json
    """
    stats = {'images_copied': 0, 'labels_created': 0, 'unmapped_classes': defaultdict(int), 'skipped': 0}
    logger.info(f"Processing BDD100K ({use_fraction * 100:.0f}% of {split})...")

    labels_file = bdd100k_root / 'labels' / f'det_{split}.json'
    images_dir = bdd100k_root / 'images' / split
    if not labels_file.exists():
        logger.warning(f"  BDD100K labels not found: {labels_file}")
        return stats

    bdd_data = json.loads(labels_file.read_text())
    if use_fraction < 1.0:
        bdd_data = random.sample(bdd_data, int(len(bdd_data) * use_fraction))

    # BDD100K images are a fixed 1280x720.
    img_width, img_height = 1280, 720

    for item_idx, item in enumerate(bdd_data):
        img_name = item['name']
        img_path = images_dir / img_name
        if not img_path.exists():
            stats['skipped'] += 1
            continue

        dest_img = output_images / f"bdd100k_{img_name}"
        try:
            shutil.copy2(img_path, dest_img)
            stats['images_copied'] += 1
        except OSError as e:
            logger.warning(f"  failed to copy {img_path}: {e}")
            stats['skipped'] += 1
            continue

        yolo_labels = []
        for obj in item.get('labels', []):
            category = obj.get('category', 'unknown').lower()
            if category not in BDD100K_TO_CLASSES:
                stats['unmapped_classes'][category] += 1
                continue

            class_id = BDD100K_TO_CLASSES[category]
            bbox = obj.get('box2d', {})
            x1, y1, x2, y2 = bbox.get('x1', 0), bbox.get('y1', 0), bbox.get('x2', 0), bbox.get('y2', 0)
            if x1 >= x2 or y1 >= y2:
                continue

            x_center = max(0.0, min(1.0, (x1 + x2) / 2 / img_width))
            y_center = max(0.0, min(1.0, (y1 + y2) / 2 / img_height))
            width = max(0.0, min(1.0, (x2 - x1) / img_width))
            height = max(0.0, min(1.0, (y2 - y1) / img_height))
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if yolo_labels:
            dest_label = output_labels / f"bdd100k_{img_name.replace(Path(img_name).suffix, '.txt')}"
            dest_label.write_text('\n'.join(yolo_labels))
            stats['labels_created'] += 1

        if (item_idx + 1) % 500 == 0:
            logger.info(f"  processed {item_idx + 1}/{len(bdd_data)} BDD100K images...")

    return stats


def create_train_val_split(images_dir: Path, labels_dir: Path, train_ratio: float = 0.8) -> Path:
    all_images = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_ratio)
    train_images, val_images = all_images[:split_idx], all_images[split_idx:]

    dirs = {
        'train_img': images_dir.parent / 'train' / 'images',
        'train_lbl': images_dir.parent / 'train' / 'labels',
        'val_img': images_dir.parent / 'val' / 'images',
        'val_lbl': images_dir.parent / 'val' / 'labels',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating train/val split ({train_ratio * 100:.0f}% train): "
                f"{len(train_images)} train / {len(val_images)} val")

    for img in train_images:
        shutil.copy2(img, dirs['train_img'] / img.name)
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, dirs['train_lbl'] / lbl.name)
    for img in val_images:
        shutil.copy2(img, dirs['val_img'] / img.name)
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, dirs['val_lbl'] / lbl.name)

    return dirs['train_img'].parent.parent


def create_data_yaml(dataset_root: Path) -> Path:
    output_file = dataset_root / 'data.yaml'
    names = [CLASSES[i] for i in range(len(CLASSES))]
    output_file.write_text(
        "# Merged OpenADAS + BDD100K config (12-class taxonomy)\n"
        f"path: {dataset_root.absolute()}\n"
        "train: train/images\n"
        "val: val/images\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {names}\n"
    )
    logger.info(f"Created data.yaml: {output_file}")
    return output_file


def merge_datasets(open_adas_root: Path, bdd100k_root: Path, output_root: Path,
                    open_adas_ratio: float = 1.0, bdd100k_ratio: float = 0.3) -> None:
    temp_images = output_root / 'temp_images'
    temp_labels = output_root / 'temp_labels'
    for d in (temp_images, temp_labels):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MERGING OPENADAS + BDD100K (12-class taxonomy)")
    logger.info("=" * 70)

    logger.info("\n[1/4] Copying OpenADAS scenes...")
    adas_stats = copy_open_adas_scenes(open_adas_root, temp_images, temp_labels, open_adas_ratio)
    logger.info(f"  images: {adas_stats['images_copied']} | labels: {adas_stats['labels_copied']}")
    if adas_stats['missing_labels']:
        logger.warning(f"  missing labels: {adas_stats['missing_labels']}")

    logger.info("\n[2/4] Processing BDD100K subset...")
    bdd_stats = process_bdd100k(bdd100k_root, temp_images, temp_labels, bdd100k_ratio)
    logger.info(f"  images: {bdd_stats['images_copied']} | labels created: {bdd_stats['labels_created']}")
    if bdd_stats['unmapped_classes']:
        top = sorted(bdd_stats['unmapped_classes'].items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"  unmapped BDD100K categories (kept out of the 12-class set): {top}")

    logger.info("\n[3/4] Creating train/val split...")
    dataset_root = create_train_val_split(temp_images, temp_labels, train_ratio=0.8)
    shutil.rmtree(temp_images, ignore_errors=True)
    shutil.rmtree(temp_labels, ignore_errors=True)

    logger.info("\n[4/4] Writing data.yaml...")
    create_data_yaml(dataset_root)

    total = adas_stats['images_copied'] + bdd_stats['images_copied']
    logger.info("\n" + "=" * 70)
    logger.info(f"Merge complete — {total} images at {dataset_root}")
    logger.info(f"Next: python train.py --data {dataset_root}/data.yaml")
    logger.info("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge OpenADAS + a subset of BDD100K onto the 12-class taxonomy')
    parser.add_argument('--open-adas', required=True, help='Path to OpenADAS dataset root')
    parser.add_argument('--bdd100k', required=True, help='Path to a local BDD100K copy')
    parser.add_argument('--output', default='./merged_dataset', help='Output directory')
    parser.add_argument('--adas-ratio', type=float, default=1.0, help='Fraction of OpenADAS to keep')
    parser.add_argument('--bdd-ratio', type=float, default=0.3, help='Fraction of BDD100K to keep')
    args = parser.parse_args()

    for path in (args.open_adas, args.bdd100k):
        if not Path(path).exists():
            logger.error(f"Path not found: {path}")
            raise SystemExit(1)

    merge_datasets(Path(args.open_adas), Path(args.bdd100k), Path(args.output), args.adas_ratio, args.bdd_ratio)
