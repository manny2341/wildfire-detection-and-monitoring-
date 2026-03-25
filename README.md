# Wildfire Detection & Monitoring System

A complete wildfire detection and monitoring system using YOLOv8 deep 
learning and Sentinel-2 satellite imagery. The system detects active 
fire zones, classifies burn severity, and maps total damage area.

---

## System Flow
```
Sentinel-2 Satellite Data (Copernicus)
              |
    +---------+---------+
    |                   |
YOLO Detection      NBR Analysis
    |                   |
Fire Zones +        Pixel-level
Severity Class      Fire Map
    |                   |
    +---------+---------+
              |
      Burn Severity Map
              |
      NDVI Vegetation Loss
              |
      Final Damage Report
```

---

## Detection Methods

### Method 1 — YOLOv8 (Detection + Severity Classification)
- Fine-tuned YOLOv8 nano model trained on our own satellite data
- Uses transfer learning from NBR labels — no manual labelling needed
- Splits satellite image into 640x640 pixel tiles (6.4km x 6.4km each)
- Detects fire zones AND classifies severity in one single pass
- Outputs colour-coded bounding boxes with confidence scores
- Trained for 41 epochs on 476 labelled tiles
- Achieved 94.1% mAP accuracy

### Method 2 — NBR/NBFI Spectral Index
- Uses Near-Infrared (B8A) and SWIR (B12) Sentinel-2 bands
- NBFI = (B12 - B8A) / (B12 + B8A)
- Classifies every pixel as FIRE or NO FIRE
- Maps active fire front during peak burning days
- 62 km² of active fire detected on Rhodes

---

## YOLO Training Pipeline

| Step | Action | Result |
|---|---|---|
| 1 | Download Sentinel-2 post-fire image | 6712x5464 pixel image |
| 2 | Generate NBR severity labels | 5 class labels per pixel |
| 3 | Split into 640x640 tiles | 80 base tiles |
| 4 | Augment burned class tiles | 476 training tiles total |
| 5 | Fine-tune YOLOv8 | 41 epochs, early stopping |
| 6 | Evaluate performance | 94.1% mAP achieved |

---

## YOLO Model Performance

| Metric | Score |
|---|---|
| Precision | 93.0% |
| Recall | 88.1% |
| mAP50 | 94.1% |
| mAP50-95 | 80.9% |
| Training tiles | 476 |
| Epochs | 41 |

---

## Severity Classes

| Class | Label | dNBR Range | Colour |
|---|---|---|---|
| 0 | Unburned | < 0.1 | Green |
| 1 | Low Severity | 0.1 - 0.27 | Yellow |
| 2 | Moderate Severity | 0.27 - 0.44 | Orange |
| 3 | High Severity | 0.44 - 0.66 | Red |
| 4 | Extreme Severity | > 0.66 | Black |

---

## Monitoring Components

### Burn Severity Mapping (dNBR)
- Compares pre and post fire images using NASA/USGS thresholds
- Five severity classes mapped pixel by pixel
- Total burned area calculated in km²

### Vegetation Analysis (NDVI)
- Pre and post fire NDVI comparison
- Quantifies total vegetation loss
- Average NDVI loss: 0.114 across burned zones on Rhodes

---

## Validation Results — Summer 2023

| Wildfire | YOLO Zones | YOLO mAP | NBR Burned | Official | Accuracy |
|---|---|---|---|---|---|
| Rhodes, Greece | 219 zones | 94.1% | 801 km² | ~750-800 km² | Excellent |
| Evros, Greece | 101 zones | 94.1% | 1,987 km² | ~2,000 km² | Excellent |
| Tenerife, Spain | 10 zones | 94.1% | 719 km² | ~700 km² | Excellent |

---

## Method Comparison

| Feature | YOLO | NBR/NBFI |
|---|---|---|
| Type | Deep Learning AI | Physics Formula |
| Output | Bounding boxes + severity | Pixel-level map |
| Accuracy | 94.1% mAP | Within 2-5% of official |
| Training | Fine-tuned on NBR labels | No training needed |
| Best for | Real-time camera/drone | Satellite imagery |

---

## Output Files

| File | Description |
|---|---|
| yolo_severity_detection.png | YOLO detection + severity map |
| yolo_vs_nbr_comparison.png | Side by side method comparison |
| fire_detection_map.png | NBR binary fire map |
| wildfire_classification_map.png | Burn severity map |
| ndvi_analysis.png | Vegetation loss map |
| complete_system.png | Full linked system output |

---

## Tools & Libraries

- Python 3.13
- ultralytics — YOLOv8 deep learning model
- openEO — Copernicus Data Space connection
- rasterio — satellite image processing
- numpy — numerical computation
- matplotlib — visualisation
- opencv-python — image tile processing

---

## Data Source
European Space Agency — Copernicus Data Space Ecosystem
https://dataspace.copernicus.eu
