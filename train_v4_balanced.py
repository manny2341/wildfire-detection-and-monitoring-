"""
train_v4_balanced.py — Class-balanced oversampling for the Extreme class.

Problem
-------
v3 per-class mAP50:
    Unburned   99.3%
    Low        96.8%
    Moderate   92.1%
    High       71.4%
    Extreme    48.4%   <-- the rare class is the weak class

Cause
-----
Extreme severity tiles make up ~22% of training tiles vs Unburned at 93%.
Each epoch the model sees Unburned 4x more often than Extreme — so it
under-learns the extreme burn pattern.

Fix
---
Class-aware oversampling. Duplicate every Extreme-containing tile 4 extra
times. YOLO's per-epoch augmentation (flipud, fliplr, rotation, hsv) means
each duplicate is augmented differently during training, giving the model
genuine variety on the rare class.

Output
------
yolo_dataset_v4_balanced/   — same structure as v3 but with Extreme tiles
                              duplicated 4x in the train set.

Run
---
    # Just build the balanced dataset:
    python3 train_v4_balanced.py

    # Build + train (slow on CPU — hours; use a GPU box if possible):
    python3 train_v4_balanced.py --train
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

BASE = Path(__file__).parent
SOURCE = BASE / "yolo_dataset_v3"
DEST = BASE / "yolo_dataset_v4_balanced"

CLASS_NAMES = ["Unburned", "Low", "Moderate", "High", "Extreme"]
EXTREME_CLASS_ID = 4
OVERSAMPLE_FACTOR = 4   # Each Extreme-containing tile gets 4 extra copies


# ─── Helpers ─────────────────────────────────────────────────────────────────
def has_class(label_path: Path, class_id: int) -> bool:
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if int(line.split()[0]) == class_id:
                    return True
            except (ValueError, IndexError):
                pass
    return False


def class_distribution(labels_dir: Path) -> tuple[int, Counter]:
    counts = Counter()
    label_files = sorted(labels_dir.glob("*.txt"))
    for label_path in label_files:
        classes_in_tile = set()
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        classes_in_tile.add(int(line.split()[0]))
                    except (ValueError, IndexError):
                        pass
        for c in classes_in_tile:
            counts[c] += 1
    return len(label_files), counts


def print_distribution(label_dir: Path, header: str) -> None:
    n_tiles, counts = class_distribution(label_dir)
    print(f"\n{header}  ({n_tiles} train tiles)")
    print(f"   {'Class':<10} {'Tiles':>6}  {'%':>6}")
    print(f"   {'-'*10} {'-'*6}  {'-'*6}")
    for class_id in range(5):
        n = counts[class_id]
        pct = (n / n_tiles * 100) if n_tiles else 0
        print(f"   {CLASS_NAMES[class_id]:<10} {n:>6}  {pct:>5.1f}%")


# ─── Pipeline ────────────────────────────────────────────────────────────────
def copy_dataset(src: Path, dst: Path) -> None:
    print(f"→ Copying base v3 dataset: {src.name} → {dst.name}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def oversample_class(dataset_dir: Path, target_class: int, factor: int) -> int:
    """Duplicate every train tile containing target_class — `factor` extra copies."""
    train_labels = dataset_dir / "labels" / "train"
    train_images = dataset_dir / "images" / "train"

    target_tiles = [
        lp.stem for lp in sorted(train_labels.glob("*.txt"))
        if has_class(lp, target_class)
    ]
    print(f"→ Found {len(target_tiles)} train tiles containing class "
          f"{target_class} ({CLASS_NAMES[target_class]})")
    print(f"→ Creating {factor} duplicates per tile "
          f"= {len(target_tiles) * factor} new tiles")

    written = 0
    for tile_name in target_tiles:
        src_img = train_images / f"{tile_name}.png"
        src_lbl = train_labels / f"{tile_name}.txt"
        if not src_img.exists() or not src_lbl.exists():
            continue
        for k in range(factor):
            dst_img = train_images / f"{tile_name}_aug{k+1}.png"
            dst_lbl = train_labels / f"{tile_name}_aug{k+1}.txt"
            shutil.copy(src_img, dst_img)
            shutil.copy(src_lbl, dst_lbl)
            written += 1
    print(f"   Wrote {written} duplicate tile pairs")
    return written


def write_dataset_yaml(dataset_dir: Path) -> Path:
    config = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 5,
        "names": CLASS_NAMES,
    }
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)
    return yaml_path


def train(yaml_path: Path, epochs: int) -> None:
    from ultralytics import YOLO
    print("\n" + "=" * 60)
    print(f" Starting YOLOv8s training (class-balanced) — {epochs} epochs")
    print("=" * 60)
    model = YOLO("yolov8s.pt")
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=8,
        name="wildfire_severity_v4_balanced",
        patience=20,
        device="cpu",
        verbose=True,
        flipud=0.5,
        fliplr=0.5,
        degrees=45.0,
        scale=0.3,
        hsv_s=0.3,
        hsv_v=0.3,
        mosaic=0.8,
        mixup=0.1,
    )
    best = BASE / "runs" / "detect" / "wildfire_severity_v4_balanced" / "weights" / "best.pt"
    print(f"\n✓ Training complete. Best weights: {best}")


def main(do_train: bool, epochs: int) -> None:
    print("=" * 60)
    print(" Wildfire YOLO v4 — Class-Balanced Retraining")
    print("=" * 60)

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE}\n"
            "Run retrain_v3.py first to build the multi-region dataset."
        )

    # 1. Copy v3 → v4_balanced
    copy_dataset(SOURCE, DEST)

    # 2. Show pre-augmentation distribution
    print_distribution(DEST / "labels" / "train", "[Before oversampling]")

    # 3. Oversample Extreme class
    oversample_class(DEST, EXTREME_CLASS_ID, OVERSAMPLE_FACTOR)

    # 4. Show post-augmentation distribution
    print_distribution(DEST / "labels" / "train", "[After oversampling]")

    # 5. Write dataset config
    yaml_path = write_dataset_yaml(DEST)
    print(f"\n→ Dataset config: {yaml_path}")

    # 6. Train (optional)
    if do_train:
        train(yaml_path, epochs)
    else:
        print("\n" + "─" * 60)
        print(" Dataset built. Skipping training (no --train flag).")
        print(" To train when ready:")
        print(f"   python3 {Path(__file__).name} --train")
        print(" Or run YOLO directly:")
        print(f"   yolo train data={yaml_path} model=yolov8s.pt epochs=80 imgsz=640")
        print("─" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true",
                   help="Run YOLO training after building the balanced dataset")
    p.add_argument("--epochs", type=int, default=80,
                   help="Training epochs (default 80, used only with --train)")
    args = p.parse_args()
    main(do_train=args.train, epochs=args.epochs)
