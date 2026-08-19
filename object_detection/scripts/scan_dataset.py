#!/usr/bin/env python3
"""
Scan a scene-structured OpenADAS-style dataset to inventory image counts,
annotation formats present (YOLO / CVAT XML / JSON), and any scenes
missing labels. Run this first before conversion or training.

Usage:
    python scan_dataset.py /path/to/dataset_root
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def scan_dataset(root_path: Path) -> dict:
    """Scan the entire dataset structure."""
    root = Path(root_path)

    stats = {
        'total_images': 0,
        'total_scenes': 0,
        'annotation_formats': defaultdict(int),
        'scenes_detail': {},
        'missing_labels': [],
        'yolo_labels_found': 0,
        'cvat_xml_found': 0,
        'annotation_json_found': 0,
    }

    scene_folders = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith('scene_'))
    stats['total_scenes'] = len(scene_folders)

    print(f"Scanning {stats['total_scenes']} scenes...\n")

    for scene_idx, scene_folder in enumerate(scene_folders):
        scene_name = scene_folder.name
        scene_stats = {
            'images': 0,
            'yolo_labels': 0,
            'cvat_xml': 0,
            'annotation_json': 0,
            'formats_found': set(),
        }

        images_folder = scene_folder / 'images'
        if images_folder.exists():
            image_files = list(images_folder.glob('*.jpg')) + list(images_folder.glob('*.png'))
            scene_stats['images'] = len(image_files)
            stats['total_images'] += len(image_files)

        yolo_folder = scene_folder / 'YOLO'
        if yolo_folder.exists():
            yolo_labels = list(yolo_folder.glob('*.txt'))
            scene_stats['yolo_labels'] = len(yolo_labels)
            stats['yolo_labels_found'] += len(yolo_labels)
            scene_stats['formats_found'].add('YOLO')

        cvat_folder = scene_folder / 'cvat for images 1.1'
        if cvat_folder.exists():
            cvat_files = list(cvat_folder.glob('*.xml'))
            scene_stats['cvat_xml'] = len(cvat_files)
            stats['cvat_xml_found'] += len(cvat_files)
            scene_stats['formats_found'].add('CVAT_XML')

        annotations_folder = scene_folder / 'annotations'
        if annotations_folder.exists():
            json_files = list(annotations_folder.glob('*.json'))
            scene_stats['annotation_json'] = len(json_files)
            stats['annotation_json_found'] += len(json_files)
            scene_stats['formats_found'].add('JSON')

        if scene_stats['images'] > 0 and scene_stats['yolo_labels'] == 0:
            if scene_stats['cvat_xml'] == 0 and scene_stats['annotation_json'] == 0:
                stats['missing_labels'].append(scene_name)

        stats['scenes_detail'][scene_name] = scene_stats

        if (scene_idx + 1) % 10 == 0:
            print(f"  Processed {scene_idx + 1}/{stats['total_scenes']} scenes...")

    return stats


def print_report(stats: dict) -> None:
    print("\n" + "=" * 70)
    print("DATASET INVENTORY REPORT")
    print("=" * 70)

    print(f"\nTotal images: {stats['total_images']}")
    print(f"Total scenes: {stats['total_scenes']}")

    print("\nAnnotation formats found:")
    print(f"   YOLO format (.txt):  {stats['yolo_labels_found']} labels")
    print(f"   CVAT XML format:     {stats['cvat_xml_found']} annotations")
    print(f"   JSON format:         {stats['annotation_json_found']} annotations")

    if stats['missing_labels']:
        print(f"\nScenes without any labels ({len(stats['missing_labels'])}):")
        for scene in stats['missing_labels'][:5]:
            print(f"   - {scene}")
        if len(stats['missing_labels']) > 5:
            print(f"   ... and {len(stats['missing_labels']) - 5} more")
    else:
        print("\nAll images have labels.")

    if stats['yolo_labels_found'] > 0 and stats['total_images'] > 0:
        coverage = (stats['yolo_labels_found'] / stats['total_images']) * 100
        print(f"\nYOLO label coverage: {coverage:.1f}%")
        if coverage < 100:
            print(f"   {stats['total_images'] - stats['yolo_labels_found']} images still need labels")
    if stats['cvat_xml_found'] > 0:
        print(f"\nCVAT XML annotations found ({stats['cvat_xml_found']}) — convert with convert_annotations.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inventory a scene-structured ADAS dataset')
    parser.add_argument('dataset_root', help='Path to the dataset root (contains scene_0000, scene_0001, ...)')
    parser.add_argument('--report-out', default=None, help='Optional path to write dataset_inventory.json')
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        print(f"Path not found: {root}")
        raise SystemExit(1)

    stats = scan_dataset(root)
    print_report(stats)

    report_file = Path(args.report_out) if args.report_out else root.parent / 'dataset_inventory.json'
    with open(report_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nDetailed report saved to: {report_file}")
