"""
Wildfire YOLO v3 Retraining Script
-----------------------------------
- Generates NBR labels for Evros + Tenerife from their pre/post TIF files
- Merges with existing Rhodes dataset
- Retrains with YOLOv8s (upgrade from nano)
"""

import os
import shutil
import glob
import cv2
import numpy as np
import rasterio
import yaml
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))

REGIONS = [
    {
        "name": "Rhodes",
        "pre_tif":  None,   # Already has tiles + labels in yolo_dataset
        "post_tif": None,
        "rgb_tif":  None,
        "use_existing": True,
    },
    {
        "name": "Evros_Greece",
        "pre_tif":  os.path.join(BASE, "Evros_Greece_pre.tif"),
        "post_tif": os.path.join(BASE, "Evros_Greece_post.tif"),
        "rgb_tif":  None,   # Evros_Greece_rgb.tif is corrupted — use false colour
    },
    {
        "name": "Tenerife_Spain",
        "pre_tif":  os.path.join(BASE, "Tenerife_Spain_pre.tif"),
        "post_tif": os.path.join(BASE, "Tenerife_Spain_post.tif"),
        "rgb_tif":  None,   # No RGB tif — use false colour from NBR bands
    },
]

TILE_SIZE  = 640
DATASET_V3 = os.path.join(BASE, "yolo_dataset_v3")
CLASS_NAMES = ["Unburned", "Low", "Moderate", "High", "Extreme"]


# ── HELPERS ──────────────────────────────────────────────────────────────────

def compute_dnbr(pre_tif, post_tif):
    """Load 2-band pre/post TIF files and return dNBR array."""
    with rasterio.open(pre_tif) as src:
        b08_pre = src.read(1).astype(float)
        b12_pre = src.read(2).astype(float)
    with rasterio.open(post_tif) as src:
        b08_post = src.read(1).astype(float)
        b12_post = src.read(2).astype(float)

    b08_pre[b08_pre == 0]   = np.nan
    b12_pre[b12_pre == 0]   = np.nan
    b08_post[b08_post == 0] = np.nan
    b12_post[b12_post == 0] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        nbr_pre  = np.where((b08_pre  + b12_pre)  != 0,
                            (b08_pre  - b12_pre)  / (b08_pre  + b12_pre),  np.nan)
        nbr_post = np.where((b08_post + b12_post) != 0,
                            (b08_post - b12_post) / (b08_post + b12_post), np.nan)

    dnbr = nbr_pre - nbr_post
    return dnbr, b08_post, b12_post


def classify_dnbr(dnbr):
    """Convert dNBR array to 5-class severity map."""
    c = np.zeros_like(dnbr, dtype=np.uint8)
    c[dnbr < 0.1]                     = 0
    c[(dnbr >= 0.10) & (dnbr < 0.27)] = 1
    c[(dnbr >= 0.27) & (dnbr < 0.44)] = 2
    c[(dnbr >= 0.44) & (dnbr < 0.66)] = 3
    c[dnbr >= 0.66]                    = 4
    c[np.isnan(dnbr)]                  = 255
    return c


def make_false_colour(b08_post, b12_post, b08_pre):
    """Create a false-colour RGB: R=B12post G=B08post B=B08pre (burned = bright red)."""
    def norm(a):
        return np.clip(a / 5000.0, 0, 1)
    R = norm(b12_post)
    G = norm(b08_post)
    B = norm(b08_pre)
    return (np.dstack([R, G, B]) * 255).astype(np.uint8)


def load_rgb_tif(rgb_tif):
    """Load a standard 3-band RGB TIF."""
    with rasterio.open(rgb_tif) as src:
        R = np.clip(src.read(1).astype(float), 0, 3000) / 3000
        G = np.clip(src.read(2).astype(float), 0, 3000) / 3000
        B = np.clip(src.read(3).astype(float), 0, 3000) / 3000
    return (np.dstack([R, G, B]) * 255).astype(np.uint8)


def tile_and_label(rgb_image, classified, region_name, counters):
    """
    Slice rgb_image + classified into TILE_SIZE tiles, write PNG + YOLO label.
    Returns number of tiles written.
    """
    h, w    = rgb_image.shape[:2]
    written = 0

    for y in range(0, h - TILE_SIZE, TILE_SIZE):
        for x in range(0, w - TILE_SIZE, TILE_SIZE):
            img_tile = rgb_image[y:y+TILE_SIZE, x:x+TILE_SIZE]
            cls_tile = classified[y:y+TILE_SIZE, x:x+TILE_SIZE]

            labels = []
            for class_id in range(5):
                mask = (cls_tile == class_id)
                pixel_count = np.sum(mask)
                # Lower threshold for burned classes
                threshold = 0.02 if class_id > 0 else 0.30
                if pixel_count > (TILE_SIZE * TILE_SIZE * threshold):
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if rows.any() and cols.any():
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        cx = (cmin + cmax) / 2 / TILE_SIZE
                        cy = (rmin + rmax) / 2 / TILE_SIZE
                        bw = (cmax - cmin) / TILE_SIZE
                        bh = (rmax - rmin) / TILE_SIZE
                        labels.append(
                            f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                        )

            if not labels:
                continue

            n      = counters[0]
            split  = "val" if n % 5 == 0 else "train"
            img_p  = os.path.join(DATASET_V3, "images", split, f"tile_{n:05d}.png")
            lbl_p  = os.path.join(DATASET_V3, "labels", split, f"tile_{n:05d}.txt")

            cv2.imwrite(img_p, cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR))
            with open(lbl_p, "w") as f:
                f.write("\n".join(labels))

            counters[0] += 1
            written += 1

    return written


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Build fresh dataset folder
    print("=" * 55)
    print("  Wildfire YOLO v3 — Multi-Region Retraining")
    print("=" * 55)

    shutil.rmtree(DATASET_V3, ignore_errors=True)
    for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(DATASET_V3, folder), exist_ok=True)

    counters = [0]   # mutable counter passed into tile_and_label

    # 2. Copy existing Rhodes tiles + labels
    print("\n[1/4] Copying existing Rhodes tiles...")
    for split in ["train", "val"]:
        imgs   = sorted(glob.glob(os.path.join(BASE, "yolo_dataset", "images", split, "*.png")))
        labels = sorted(glob.glob(os.path.join(BASE, "yolo_dataset", "labels", split, "*.txt")))
        paired = [(i, l) for i, l in zip(imgs, labels)]
        for img_p, lbl_p in paired:
            n     = counters[0]
            split_ = "val" if n % 5 == 0 else "train"
            shutil.copy(img_p, os.path.join(DATASET_V3, "images", split_, f"tile_{n:05d}.png"))
            shutil.copy(lbl_p, os.path.join(DATASET_V3, "labels", split_, f"tile_{n:05d}.txt"))
            counters[0] += 1
    print(f"   Rhodes tiles copied: {counters[0]}")

    # 3. Generate Evros tiles + labels from TIF
    print("\n[2/4] Generating Evros Greece tiles + labels...")
    evros = REGIONS[1]
    dnbr, b08_post, b12_post = compute_dnbr(evros["pre_tif"], evros["post_tif"])

    # Load b08_pre for false colour
    with rasterio.open(evros["pre_tif"]) as src:
        b08_pre = src.read(1).astype(float)

    classified = classify_dnbr(dnbr)

    if evros["rgb_tif"] and os.path.exists(evros["rgb_tif"]):
        rgb_image = load_rgb_tif(evros["rgb_tif"])
        # Resize to match classified if needed
        if rgb_image.shape[:2] != classified.shape[:2]:
            h, w = classified.shape[:2]
            rgb_image = cv2.resize(rgb_image, (w, h))
    else:
        rgb_image = make_false_colour(b08_post, b12_post, b08_pre)

    before = counters[0]
    n_evros = tile_and_label(rgb_image, classified, "Evros", counters)
    print(f"   Evros tiles generated: {n_evros}  (total so far: {counters[0]})")

    # 4. Generate Tenerife tiles + labels from TIF
    print("\n[3/4] Generating Tenerife Spain tiles + labels...")
    ten = REGIONS[2]
    dnbr, b08_post, b12_post = compute_dnbr(ten["pre_tif"], ten["post_tif"])

    with rasterio.open(ten["pre_tif"]) as src:
        b08_pre = src.read(1).astype(float)

    classified  = classify_dnbr(dnbr)
    rgb_image   = make_false_colour(b08_post, b12_post, b08_pre)

    n_tenerife = tile_and_label(rgb_image, classified, "Tenerife", counters)
    print(f"   Tenerife tiles generated: {n_tenerife}  (total so far: {counters[0]})")

    # 5. Dataset summary
    train_count = len(os.listdir(os.path.join(DATASET_V3, "images", "train")))
    val_count   = len(os.listdir(os.path.join(DATASET_V3, "images", "val")))
    print(f"\n[4/4] Dataset ready — Train: {train_count}  Val: {val_count}  Total: {counters[0]}")

    # 6. Write dataset.yaml
    config = {
        "path":  DATASET_V3,
        "train": "images/train",
        "val":   "images/val",
        "nc":    5,
        "names": CLASS_NAMES,
    }
    yaml_path = os.path.join(DATASET_V3, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

    # 7. Train
    print("\n" + "=" * 55)
    print("  Starting YOLOv8s training — this will take a while")
    print("=" * 55 + "\n")

    model = YOLO("yolov8s.pt")

    results = model.train(
        data=yaml_path,
        epochs=80,
        imgsz=640,
        batch=8,
        name="wildfire_severity_v3",
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

    best = os.path.join(BASE, "runs", "detect", "wildfire_severity_v3", "weights", "best.pt")
    print("\n" + "=" * 55)
    print("  Training complete!")
    print(f"  Best model: {best}")
    print("=" * 55)


if __name__ == "__main__":
    main()
