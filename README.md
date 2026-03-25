# Wildfire Detection & Monitoring System

A complete wildfire detection and monitoring system built using Sentinel-2 satellite imagery from the European Space Agency Copernicus programme.

## What It Does

### Part 1 — Fire Detection
- Downloads satellite images taken DURING an active fire
- Uses Near-Infrared (B8A) and Shortwave Infrared (B11, B12) bands
- Calculates the Normalized Burned Fire Index (NBFI)
- Classifies every pixel as FIRE or NO FIRE
- Maps the active fire front in real time

### Part 2 — Burn Severity Monitoring
- Compares satellite images before and after the fire
- Calculates the Normalized Burn Ratio (NBR)
- Classifies burn severity using NASA/USGS thresholds
- Measures total burned area in km²
- Produces full burn severity maps

### Part 3 — Vegetation Analysis
- Calculates NDVI before and after the fire
- Measures how much vegetation was destroyed
- Maps vegetation loss across the burned area

### Part 4 — Linked System
- Detection automatically feeds into monitoring
- One function runs the complete pipeline
- Outputs unified Detection + Monitoring + Report map

## System Flow
```
Satellite Data → Fire Detection → Burn Severity → NDVI Loss → Final Report
```

## Case Studies — Summer 2023

| Wildfire | Active Fire | Total Burned | Official Report | Accuracy |
|---|---|---|---|---|
| Rhodes, Greece | 62 km² | 801 km² | ~750-800 km² | Excellent |
| Evros, Greece | - | 1,987 km² | ~2,000 km² | Excellent |
| Tenerife, Spain | - | 719 km² | ~700 km² | Excellent |

## Tools & Libraries
- Python 3.13
- openEO — Copernicus Data Space connection
- rasterio — satellite image processing
- numpy — numerical computation
- matplotlib — visualisation
- geopandas — geospatial data

## Data Source
European Space Agency — Copernicus Data Space Ecosystem
https://dataspace.copernicus.eu

## Key Results
- Active fire detection: 62 km² detected during peak burning
- Total burned area: 801 km² (within 1% of official figure)
- Average NDVI loss: 0.114 across burned zones
- Model validated across 3 wildfires in 2 countries
