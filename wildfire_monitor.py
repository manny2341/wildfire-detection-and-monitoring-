#!/usr/bin/env python3
# ============================================================
# WILDFIRE AUTOMATIC MONITORING SYSTEM
# Runs daily — checks any region for new fire activity
# Sends email alerts + logs results to CSV for dashboard
# ============================================================

import openeo
import rasterio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
import smtplib
import csv
import shutil
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── PATHS ─────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
CONFIG_FILE  = BASE_DIR / "monitor_config.json"
REGIONS_FILE = BASE_DIR / "monitor_regions.json"
LOG_FILE     = BASE_DIR / "monitoring_log.csv"
MAPS_DIR     = BASE_DIR / "monitoring_maps"
TEMP_DIR     = BASE_DIR / "_temp"
MAPS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ── LOAD CONFIG ───────────────────────────────────────────────
def load_config():
    with open(CONFIG_FILE) as f:
        return json.load(f)

def load_regions():
    with open(REGIONS_FILE) as f:
        return json.load(f)["regions"]

# ── CONNECT TO COPERNICUS ─────────────────────────────────────
def connect_copernicus():
    print("Connecting to Copernicus Data Space...")
    conn = openeo.connect("openeo.dataspace.copernicus.eu")
    conn.authenticate_oidc()
    print("Connected and authenticated.")
    return conn

# ── DATE HELPERS ──────────────────────────────────────────────
def get_date_ranges():
    today = datetime.now(timezone.utc)
    # Baseline: 20–14 days ago (stable, cloud-free reference)
    baseline_start = (today - timedelta(days=20)).strftime("%Y-%m-%d")
    baseline_end   = (today - timedelta(days=14)).strftime("%Y-%m-%d")
    # Latest: last 5 days
    latest_start   = (today - timedelta(days=5)).strftime("%Y-%m-%d")
    latest_end     = today.strftime("%Y-%m-%d")
    return baseline_start, baseline_end, latest_start, latest_end

# ── DOWNLOAD NBR ──────────────────────────────────────────────
def download_nbr(conn, bbox, period, output_path):
    cube = conn.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=period,
        spatial_extent=bbox,
        bands=["B08", "B12"],
        max_cloud_cover=20
    ).reduce_dimension(dimension="t", reducer="mean")

    nbr = (cube.band("B08") - cube.band("B12")) / \
          (cube.band("B08") + cube.band("B12"))
    nbr.download(str(output_path))

# ── COMPUTE dNBR ──────────────────────────────────────────────
def compute_dnbr(baseline_path, latest_path):
    with rasterio.open(baseline_path) as src:
        baseline = src.read(1).astype(float)
        nodata_b = src.nodata
    with rasterio.open(latest_path) as src:
        latest = src.read(1).astype(float)
        nodata_l = src.nodata

    # Mask nodata and zero values
    if nodata_b is not None:
        baseline[baseline == nodata_b] = np.nan
    baseline[baseline == 0] = np.nan

    if nodata_l is not None:
        latest[latest == nodata_l] = np.nan
    latest[latest == 0] = np.nan

    # Clip to valid NBR range — anything outside [-1, 1] is clouds/noise
    baseline = np.clip(baseline, -1, 1)
    latest   = np.clip(latest,   -1, 1)

    # Positive dNBR = vegetation loss = fire/burn
    return baseline - latest

# ── DETECT FIRE ───────────────────────────────────────────────
def detect_fire(dnbr, config):
    threshold  = config["thresholds"]["dnbr_threshold"]
    min_pixels = config["thresholds"]["fire_pixel_min"]
    pixel_area = 0.0001  # km2 per 10m pixel

    low      = int(np.sum((dnbr >= 0.1)  & (dnbr < 0.27)))
    moderate = int(np.sum((dnbr >= 0.27) & (dnbr < 0.44)))
    high     = int(np.sum((dnbr >= 0.44) & (dnbr < 0.66)))
    extreme  = int(np.sum(dnbr >= 0.66))
    total_burned = (low + moderate + high + extreme) * pixel_area

    # Fire alert triggers when moderate+ pixels exceed minimum count
    # AND dNBR exceeds threshold
    significant_pixels = moderate + high + extreme
    fire_detected = (significant_pixels >= min_pixels) and \
                    (float(np.nanmax(dnbr)) >= threshold)

    return {
        "fire_detected":    bool(fire_detected),
        "total_burned_km2": round(total_burned, 2),
        "low_km2":          round(low      * pixel_area, 2),
        "moderate_km2":     round(moderate * pixel_area, 2),
        "high_km2":         round(high     * pixel_area, 2),
        "extreme_km2":      round(extreme  * pixel_area, 2),
        "max_dnbr":         round(float(np.nanmax(dnbr)) if np.any(~np.isnan(dnbr)) else 0.0, 3),
    }

# ── GENERATE SEVERITY MAP ─────────────────────────────────────
def generate_map(dnbr, region_name, timestamp):
    classified = np.zeros_like(dnbr, dtype=np.uint8)
    classified[dnbr < 0.1]                     = 0
    classified[(dnbr >= 0.1) & (dnbr < 0.27)]  = 1
    classified[(dnbr >= 0.27) & (dnbr < 0.44)] = 2
    classified[(dnbr >= 0.44) & (dnbr < 0.66)] = 3
    classified[dnbr >= 0.66]                    = 4
    classified[np.isnan(dnbr)]                  = 255

    colors = {
        0:   (0.2, 0.6, 0.2),
        1:   (1.0, 1.0, 0.0),
        2:   (1.0, 0.6, 0.0),
        3:   (1.0, 0.2, 0.0),
        4:   (0.1, 0.0, 0.0),
        255: (0.05, 0.05, 0.1),
    }
    rgb = np.zeros((*classified.shape, 3), dtype=float)
    for v, c in colors.items():
        rgb[classified == v] = c

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.imshow(rgb)
    ax.set_title(
        f"FIRE ALERT — {region_name}\n{timestamp}",
        fontsize=14, fontweight="bold", color="red"
    )
    ax.axis("off")

    legend = [
        mpatches.Patch(color=(0.2, 0.6, 0.2), label="Unburned"),
        mpatches.Patch(color=(1.0, 1.0, 0.0), label="Low Severity"),
        mpatches.Patch(color=(1.0, 0.6, 0.0), label="Moderate Severity"),
        mpatches.Patch(color=(1.0, 0.2, 0.0), label="High Severity"),
        mpatches.Patch(color=(0.1, 0.0, 0.0), label="Extreme Severity"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=10,
              facecolor="#2a2a3e", labelcolor="white")
    plt.tight_layout()

    safe_name = region_name.replace(" ", "_")
    map_path  = MAPS_DIR / f"alert_{safe_name}_{timestamp}.png"
    plt.savefig(map_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Map saved: {map_path}")
    return str(map_path)

# ── SEND EMAIL ALERT ──────────────────────────────────────────
def send_email(region_name, results, map_path, config, timestamp):
    email_cfg = config["email"]

    # Load password from .env (takes priority over config file)
    password = os.getenv("WILDFIRE_EMAIL_PASSWORD") or email_cfg.get("password", "")

    # Skip if email not configured
    if (not email_cfg["sender"]
            or email_cfg["sender"] == "your-email@gmail.com"
            or not password):
        print("  Email not configured — skipping email alert")
        print("  Edit monitor_config.json and .env to enable email alerts")
        return

    subject = (f"WILDFIRE ALERT: {region_name} — "
               f"{results['total_burned_km2']} km2 detected")

    body = f"""
WILDFIRE MONITORING SYSTEM — AUTOMATIC ALERT
=============================================
Region    : {region_name}
Timestamp : {timestamp}
Status    : FIRE ACTIVITY DETECTED

BURN SEVERITY BREAKDOWN
-----------------------
Low severity      : {results['low_km2']} km2
Moderate severity : {results['moderate_km2']} km2
High severity     : {results['high_km2']} km2
Extreme severity  : {results['extreme_km2']} km2
-----------------------
TOTAL BURNED      : {results['total_burned_km2']} km2

Peak dNBR value   : {results['max_dnbr']}

A burn severity map is attached to this email.

This is an automated daily alert from your Wildfire Monitoring System.
"""

    msg = MIMEMultipart()
    msg["From"]    = email_cfg["sender"]
    msg["To"]      = email_cfg["recipient"]
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if map_path and os.path.exists(map_path):
        with open(map_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header(
                "Content-Disposition", "attachment",
                filename=os.path.basename(map_path)
            )
            msg.attach(img)

    try:
        with smtplib.SMTP(email_cfg["smtp_server"],
                          email_cfg["smtp_port"]) as server:
            server.starttls()
            server.login(email_cfg["sender"], password)
            server.send_message(msg)
        print(f"  Email alert sent to {email_cfg['recipient']}")
    except Exception as e:
        print(f"  Email failed: {e}")

# ── LOG RESULT TO CSV ─────────────────────────────────────────
def log_result(region_name, results, timestamp):
    fieldnames = [
        "timestamp", "region", "fire_detected",
        "total_burned_km2", "low_km2", "moderate_km2",
        "high_km2", "extreme_km2", "max_dnbr"
    ]
    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp":        timestamp,
            "region":           region_name,
            "fire_detected":    results["fire_detected"],
            "total_burned_km2": results["total_burned_km2"],
            "low_km2":          results["low_km2"],
            "moderate_km2":     results["moderate_km2"],
            "high_km2":         results["high_km2"],
            "extreme_km2":      results["extreme_km2"],
            "max_dnbr":         results["max_dnbr"],
        })
    print(f"  Logged to {LOG_FILE.name}")

# ── MAIN ──────────────────────────────────────────────────────
def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    print(f"\n{'='*50}")
    print(f" WILDFIRE MONITOR — {timestamp} UTC")
    print(f"{'='*50}")

    config  = load_config()
    regions = load_regions()
    print(f"Monitoring {len(regions)} region(s)\n")

    baseline_start, baseline_end, latest_start, latest_end = get_date_ranges()
    print(f"Baseline period : {baseline_start} to {baseline_end}")
    print(f"Latest period   : {latest_start}   to {latest_end}")

    conn        = connect_copernicus()
    alerts_sent = 0
    errors      = 0

    for region in regions:
        name = region["name"]
        bbox = region["bbox"]
        safe = name.replace(" ", "_")

        print(f"\n--- Checking: {name} ---")

        baseline_path = TEMP_DIR / f"baseline_{safe}.tif"
        latest_path   = TEMP_DIR / f"latest_{safe}.tif"

        try:
            print("  Downloading baseline imagery...")
            download_nbr(conn, bbox,
                         [baseline_start, baseline_end],
                         baseline_path)

            print("  Downloading latest imagery...")
            download_nbr(conn, bbox,
                         [latest_start, latest_end],
                         latest_path)

            print("  Computing dNBR...")
            dnbr    = compute_dnbr(baseline_path, latest_path)
            results = detect_fire(dnbr, config)

            status = "FIRE DETECTED" if results["fire_detected"] else "All clear"
            print(f"  Status         : {status}")
            print(f"  Total burned   : {results['total_burned_km2']} km2")
            print(f"  Max dNBR       : {results['max_dnbr']}")

            log_result(name, results, timestamp)

            if results["fire_detected"]:
                map_path = generate_map(dnbr, name, timestamp)
                send_email(name, results, map_path, config, timestamp)
                alerts_sent += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1
            log_result(name, {
                "fire_detected": False,
                "total_burned_km2": 0,
                "low_km2": 0, "moderate_km2": 0,
                "high_km2": 0, "extreme_km2": 0,
                "max_dnbr": 0,
            }, timestamp)
        finally:
            # Always clean up temp files to keep project folder tidy
            for p in [baseline_path, latest_path]:
                if p.exists():
                    p.unlink()

    print(f"\n{'='*50}")
    print(f" Run complete — {alerts_sent} alert(s) sent, {errors} error(s)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
