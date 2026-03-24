# 🔥 Wildfire Detection & Monitoring System

A wildfire detection and monitoring system built using **Sentinel-2 satellite imagery** from the European Space Agency's Copernicus programme.

## What It Does
- Downloads real satellite images from the Copernicus Data Space Ecosystem
- Calculates the **Normalized Burn Ratio (NBR)** to detect burned areas
- Classifies burn severity using **NASA/USGS thresholds**
- Measures **vegetation loss using NDVI**
- Calculates total burned area in km²

## Case Studies — Summer 2023
| Wildfire | My Model | Official Report | Accuracy |
|---|---|---|---|
| Rhodes, Greece | 801 km² | ~750-800 km² | Excellent |
| Evros, Greece | 1,987 km² | ~2,000 km² | Excellent |
| Tenerife, Spain | 719 km² | ~700 km² | Excellent |

## Tools & Libraries
- Python 3.13
- openEO (Copernicus Data Space)
- rasterio, numpy, matplotlib, geopandas

## Data Source
European Space Agency — Copernicus Data Space Ecosystem
https://dataspace.copernicus.eu
