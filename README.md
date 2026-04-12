# Wildfire Detection & Monitoring System

A complete wildfire detection and monitoring system using YOLOv8 deep learning and Sentinel-2 satellite imagery. The system detects active fire zones, classifies burn severity, maps total damage area, and monitors ongoing wildfire events.

---

## Prerequisites

- Python 3.13
- A free [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) account (required to download Sentinel-2 imagery)

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd wildfire-detection-and-monitoring

# Create and activate a virtual environment (recommended)
python3 -m venv wildfire_env
source wildfire_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Streamlit Web App
```bash
streamlit run wildfire_app.py
```
Open your browser at `http://localhost:8501`

### Jupyter Notebooks
```bash
jupyter notebook
```

| Notebook | Purpose |
|---|---|
| `index.ipynb` | Main analysis — downloads Sentinel-2 data, computes NBR/NDVI, trains YOLO |
| `wildfire_notebook2.ipynb` | Extended pipeline — multi-region validation and YOLO v2 retraining |
| `wildfire_notebook_final.ipynb` | Final summary with all outputs and visualisations |

### Standalone Scripts
```bash
# Run wildfire detection on a satellite image
python wildfire_detection.py

# Run the automatic monitoring system
python wildfire_monitor.py
```

---

## Project Structure

```
wildfire-detection-and-monitoring/
├── wildfire_app.py              # Streamlit web application
├── wildfire_detection.py        # Standalone detection script
├── wildfire_monitor.py          # Automatic monitoring script
├── index.ipynb                  # Main analysis notebook
├── wildfire_notebook2.ipynb     # Extended pipeline notebook
├── wildfire_notebook_final.ipynb# Final results notebook
├── requirements.txt             # Python dependencies
├── monitor_config.json          # Monitoring configuration
├── monitor_regions.json         # Regions being monitored
├── monitoring_log.csv           # Monitoring event log
├── yolov8n.pt                   # Base YOLOv8 model weights
├── runs/                        # YOLO training outputs
├── yolo_dataset/                # Generated training tiles
└── monitoring_maps/             # Saved monitoring outputs
```

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
- Splits satellite image into 640×640 pixel tiles (6.4 km × 6.4 km each)
- Detects fire zones AND classifies severity in one single pass
- Outputs colour-coded bounding boxes with confidence scores
- Trained for 41 epochs on 476 labelled tiles
- Achieved 94.1% mAP accuracy

### Method 2 — NBR Spectral Index
- Uses Near-Infrared (B08) and SWIR (B12) Sentinel-2 bands
- NBR = (B08 - B12) / (B08 + B12)
- dNBR = pre-fire NBR − post-fire NBR
- Classifies every pixel into a burn severity class
- Maps burned area pixel by pixel across the entire scene
- 62 km² of active fire detected on Rhodes

---

## YOLO Training Pipeline

| Step | Action | Result |
|---|---|---|
| 1 | Download Sentinel-2 post-fire image | 6712×5464 pixel image |
| 2 | Generate NBR severity labels | 5 class labels per pixel |
| 3 | Split into 640×640 tiles | 80 base tiles |
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
| 1 | Low Severity | 0.10 – 0.27 | Yellow |
| 2 | Moderate Severity | 0.27 – 0.44 | Orange |
| 3 | High Severity | 0.44 – 0.66 | Red |
| 4 | Extreme Severity | > 0.66 | Black |

---

## Validation Results — Summer 2023

| Wildfire | YOLO Zones | YOLO mAP | NBR Burned | Official | Accuracy |
|---|---|---|---|---|---|
| Rhodes, Greece | 219 zones | 94.1% | 801 km² | ~750–800 km² | Excellent |
| Evros, Greece | 101 zones | 94.1% | 1,987 km² | ~2,000 km² | Excellent |
| Tenerife, Spain | 10 zones | 94.1% | 719 km² | ~700 km² | Excellent |

---

## Method Comparison

| Feature | YOLO | NBR |
|---|---|---|
| Type | Deep Learning AI | Physics Formula |
| Output | Bounding boxes + severity | Pixel-level map |
| Accuracy | 94.1% mAP | Within 2–5% of official |
| Training | Fine-tuned on NBR labels | No training needed |
| Best for | Real-time camera/drone | Satellite imagery |

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

## Output Files

| File | Description |
|---|---|
| `yolo_severity_detection.png` | YOLO detection + severity map |
| `yolo_vs_nbr_comparison.png` | Side-by-side method comparison |
| `fire_detection_map.png` | NBR binary fire map |
| `wildfire_classification_map.png` | Burn severity map |
| `ndvi_analysis.png` | Vegetation loss map |
| `complete_system.png` | Full linked system output |

---

## Tools & Libraries

| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv8 deep learning model |
| `openeo` | Copernicus Data Space connection |
| `rasterio` | Satellite image processing |
| `numpy` | Numerical computation |
| `matplotlib` | Visualisation |
| `opencv-python` | Image tile processing |
| `geopandas` | Geospatial data handling |
| `streamlit` | Web application interface |

---

## Data Source

European Space Agency — Copernicus Data Space Ecosystem
https://dataspace.copernicus.eu

---

## Licence

MIT Licence — free to use, modify, and distribute with attribution.
