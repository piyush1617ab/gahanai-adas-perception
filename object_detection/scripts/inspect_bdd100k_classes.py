#!/usr/bin/env python3
"""
Small helper for the optional BDD100K merge path (merge_datasets.py):
print the distinct object categories present in a BDD100K detection
labels file, so you can sanity-check BDD100K_TO_CLASSES coverage.

Usage:
    python inspect_bdd100k_classes.py /path/to/bdd100k_labels_images_train.json
"""

import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List distinct BDD100K detection categories")
    parser.add_argument("labels_json", type=Path, help="Path to a BDD100K det_*.json labels file")
    parser.add_argument("--sample", type=int, default=200, help="Number of items to sample (0 = all)")
    args = parser.parse_args()

    data = json.loads(args.labels_json.read_text())
    items = data if args.sample == 0 else data[:args.sample]

    classes = set()
    for item in items:
        for label in item.get('labels', []):
            classes.add(label.get('category', 'unknown'))

    print(f"BDD100K categories found (from {len(items)} items):")
    for cls in sorted(classes):
        print(f"  - {cls}")
