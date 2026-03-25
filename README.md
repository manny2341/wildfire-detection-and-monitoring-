# Wildfire Detection & Monitoring System

A complete wildfire detection and monitoring system built using Sentinel-2 
satellite imagery from the European Space Agency Copernicus programme.
Two detection methods are implemented and compared — YOLOv8 Deep Learning 
and NBR Spectral Index Analysis.

---

## System Overview
```
Satellite Data (Sentinel-2)
        ↓
┌───────────────────────────────────────┐
│         DETECTION PHASE               │
│  ┌─────────────┐  ┌─────────────────┐ │
│  │ YOLO v8     │  │ NBR/NBFI Index  │ │
│  │ Deep Learning│  │ Spectral Method │ │
│  └─────────────┘  └─────────────────┘ │
└───────────────────────────────────────┘
        ↓
┌───────────────────────────────────────┐
│         MONITORING PHASE              │
│  Burn Severity Map (dNBR)             │
│  NDVI Vegetation Loss Analysis        │
└───────────────────────────────────────┘
        ↓
   Final Report + Maps
```

---

## Detection Methods

### Method 1 — YOLOv8 (Deep Learning)
- Uses YOLOv8 nano model for object detection
- Splits satellite image into 640x640 pixel tiles
- Detects fire zones with confidence scores
- Returns bounding boxes around detected fire areas
- 8 fire zones detected on Rhodes with up to 74% confidence

### Method 2 — NBR/NBFI (Spectral Index)
- Uses Near-Infrared (B8A) and SWIR (B12) bands
- Calculates Normalized Burned Fire Index (NBFI)
- Classifies every pixel as FIRE or NO FIRE
- Maps active fire front during peak burning days
- 62 km² of active fire detected on Rhodes

### Method Comparison

| Feature | YOLO | NBR/NBFI |
|---|---|---|
| Type | Deep Learning AI | Physics Formula |
| Output | Bounding boxes | Pixel map |
| Confidence | Up to 74% | Within 2-5% of official |
| Best for | Real-time camera/drone | Satellite imagery |
| Training needed | Pre-trained | No training needed |

---

## Monitoring Components

### Burn Severity Mapping (dNBR)
- Compares pre and post fire satellite images
- Classifies every pixel using NASA/USGS thresholds
- Five severity classes: Unburned, Low, Moderate, High, Extreme
- Calculates total burned area in km²

### Vegetation Analysis (NDVI)
- Measures vegetation health before and after fire
- Quantifies total vegetation loss
- Maps areas of greatest ecological damage

---

## Case Studies — Summer 2023

| Wildfire | Active Fire | Total Burned | Official Report | Accuracy |
|---|---|---|---|---|
| Rhodes, Greece | 62 km² | 801 km² | ~750-800 km² | Excellent |
| Evros, Greece | - | 1,987 km² | ~2,000 km² | Excellent |
| Tenerife, Spain | - | 719 km² | ~700 km² | Excellent |

---

## Key Results

- YOLO detected 8 fire zones with max confidence of 74%
- NBR mapped 62 km² of active fire during peak burning
- Total burned area: 801 km² (within 1% of official figure)
- Average NDVI loss: 0.114 across burned zones
- Model validated across 3 wildfires in 2 countries

---

## Output Files

| File | Description |
|---|---|
| yolo_detection_map.png | YOLO bounding box detections |
| yolo_vs_nbr_comparison.png | Side by side method comparison |
| fire_detection_map.png | NBR binary fire map |
| wildfire_classification_map.png | Burn severity map |
| ndvi_analysis.png | Vegetation loss map |
| complete_system.png | Full linked system output |
| Wildfire_Detection_Report.pdf | Full analysis report |

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
