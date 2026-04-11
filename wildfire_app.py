# ============================================================
# WILDFIRE DETECTION & MONITORING SYSTEM
# Streamlit Web Application — TIF Support
# ============================================================

import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO
import cv2
import os
import csv
import json
import tempfile
import shutil
from io import BytesIO
from pathlib import Path
from datetime import datetime

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Wildfire Detection & Monitoring",
    page_icon="🔥",
    layout="wide"
)

# ── SESSION STATE ────────────────────────────────────────────
if "post_path" not in st.session_state:
    st.session_state.post_path = None
if "pre_path" not in st.session_state:
    st.session_state.pre_path = None

# ── STYLING ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #cc4400 100%);
        padding: 2rem; border-radius: 10px;
        text-align: center; margin-bottom: 2rem;
    }
    .main-header h1 { color: white; font-size: 2.5rem; margin: 0; }
    .main-header p  { color: #ffcc00; font-size: 1.1rem; margin: 0.5rem 0 0 0; }
    .fire-detected {
        background: #fff0f0; padding: 1rem; border-radius: 8px;
        border: 2px solid #cc2200; text-align: center;
        font-size: 1.3rem; font-weight: bold; color: #cc2200;
    }
    .no-fire {
        background: #f0fff0; padding: 1rem; border-radius: 8px;
        border: 2px solid #1a6e3c; text-align: center;
        font-size: 1.3rem; font-weight: bold; color: #1a6e3c;
    }
    .metric-box {
        background: #1a1a2e; color: white;
        padding: 1rem; border-radius: 8px; text-align: center;
    }
    .metric-box h3 { color: #ffcc00; margin: 0; font-size: 1.8rem; }
    .metric-box p  { color: #aaaaaa; margin: 0.3rem 0 0 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔥 Wildfire Detection & Monitoring System</h1>
    <p>YOLOv8 Deep Learning + Sentinel-2 NBR Spectral Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── LOAD MODEL ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "runs/detect/wildfire_severity_v2/weights/best.pt"
    if os.path.exists(model_path):
        return YOLO(model_path), True
    return YOLO("yolov8n.pt"), False

model, using_trained_model = load_model()

# ── HELPER FUNCTIONS ─────────────────────────────────────────
def load_tif_rgb(tif_path):
    with rasterio.open(tif_path) as src:
        if src.count >= 3:
            R = np.clip(src.read(1).astype(float), 0, 3000) / 3000
            G = np.clip(src.read(2).astype(float), 0, 3000) / 3000
            B = np.clip(src.read(3).astype(float), 0, 3000) / 3000
        else:
            R = np.clip(src.read(1).astype(float), 0, 3000) / 3000
            G = R.copy()
            B = R.copy()
    return np.dstack([R, G, B])

def load_tif_bands(tif_path):
    with rasterio.open(tif_path) as src:
        return [src.read(i+1).astype(float) for i in range(src.count)]

def save_upload(uploaded_file):
    suffix = "." + uploaded_file.name.split(".")[-1]
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name

def run_yolo(rgb_image, conf_threshold=0.25):
    rgb_uint8  = (rgb_image * 255).astype(np.uint8)
    tile_size  = 640
    height, width = rgb_uint8.shape[:2]
    detections = []
    conf_scores = []
    class_names = ["Unburned", "Low", "Moderate", "High", "Extreme"]
    tile_dir   = tempfile.mkdtemp()
    tile_paths = []
    tile_num   = 0

    total_tiles = (len(range(0, height - tile_size, tile_size)) *
                   len(range(0, width  - tile_size, tile_size)))
    progress = st.progress(0, text="Processing tiles...")
    processed = 0

    for y in range(0, height - tile_size, tile_size):
        for x in range(0, width - tile_size, tile_size):
            tile = rgb_uint8[y:y+tile_size, x:x+tile_size]
            tp   = os.path.join(tile_dir, f"t_{tile_num:04d}.png")
            cv2.imwrite(tp, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            tile_paths.append((tp, x, y))
            tile_num += 1

    for tp, xo, yo in tile_paths:
        results = model(tp, verbose=False, conf=conf_threshold)
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls  = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    if cls < len(class_names):
                        detections.append({
                            "x1": xo+x1, "y1": yo+y1,
                            "x2": xo+x2, "y2": yo+y2,
                            "confidence": conf,
                            "class_id": cls,
                            "class_name": class_names[cls]
                        })
                        conf_scores.append(conf)
        processed += 1
        if total_tiles > 0:
            progress.progress(processed / total_tiles,
                              text=f"Processing tile {processed}/{total_tiles}...")

    progress.empty()
    shutil.rmtree(tile_dir, ignore_errors=True)
    return detections, conf_scores

def calc_nbr(pre_bands, post_bands):
    b08_pre  = pre_bands[0].copy()
    b12_pre  = pre_bands[1].copy() if len(pre_bands) > 1 else pre_bands[0].copy()
    b08_post = post_bands[0].copy()
    b12_post = post_bands[1].copy() if len(post_bands) > 1 else post_bands[0].copy()

    for a in [b08_pre, b12_pre, b08_post, b12_post]:
        a[a == 0] = np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        pre  = np.where((b08_pre  + b12_pre)  != 0,
                        (b08_pre  - b12_pre)  / (b08_pre  + b12_pre),  np.nan)
        post = np.where((b08_post + b12_post) != 0,
                        (b08_post - b12_post) / (b08_post + b12_post), np.nan)
    return pre, post, pre - post

def classify(dnbr):
    c = np.zeros_like(dnbr, dtype=np.uint8)
    c[dnbr < 0.1]                    = 0
    c[(dnbr >= 0.1) & (dnbr < 0.27)] = 1
    c[(dnbr >= 0.27) & (dnbr < 0.44)]= 2
    c[(dnbr >= 0.44) & (dnbr < 0.66)]= 3
    c[dnbr >= 0.66]                   = 4
    c[np.isnan(dnbr)]                 = 255
    return c

def fig_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.info("YOLOv8 + Sentinel-2 NBR analysis for wildfire detection and monitoring.")

    st.markdown("### Model Info")
    if using_trained_model:
        st.success("Custom YOLOv8 — mAP: 94.1%")
    else:
        st.warning("Custom model not found. Using generic YOLOv8 — results may vary.")
    st.success("NBR Accuracy: Within 2-5%")

    st.markdown("### Detection Settings")
    conf_threshold = st.slider(
        "YOLO Confidence Threshold", 0.10, 0.90, 0.25, 0.05,
        help="Lower = detect more zones. Higher = only high-confidence detections."
    )

    st.markdown("### How to Use")
    st.markdown("""
    **Option A — Upload files:**
    Upload .tif files directly

    **Option B — Existing data:**
    Select from already downloaded satellite data
    """)

    st.markdown("### System Health")
    checks = [
        ("Trained Model",   os.path.exists("runs/detect/wildfire_severity_v2/weights/best.pt")),
        ("Monitor Config",  os.path.exists("monitor_config.json")),
        ("Monitor Regions", os.path.exists("monitor_regions.json")),
        ("Monitoring Maps", os.path.exists("monitoring_maps")),
        ("Monitor Log",     os.path.exists("monitoring_log.csv")),
    ]
    for label, ok in checks:
        if ok:
            st.success(label)
        else:
            st.error(f"{label} — missing")

# ── MAIN APP ─────────────────────────────────────────────────
st.markdown("## Select Satellite Images")

tab1, tab2, tab3 = st.tabs(["Analyse Images", "Existing Data", "Dashboard"])

with tab1:
    st.markdown("#### Upload your own .tif satellite images")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Post-Fire Image (Required)**")
        st.caption("Satellite image taken AFTER the fire. Bands: B04, B03, B02 (RGB) or B08, B12.")
        post_upload = st.file_uploader(
            "Post-fire .tif", type=["tif", "tiff", "png", "jpg"],
            key="post_upload"
        )
        if post_upload:
            st.session_state.post_path = save_upload(post_upload)
            st.success(f"Uploaded: {post_upload.name}")

    with col2:
        st.markdown("**Pre-Fire Image (For NBR severity analysis)**")
        st.caption("Satellite image taken BEFORE the fire. Needs bands B08 and B12 for NBR calculation.")
        pre_upload = st.file_uploader(
            "Pre-fire .tif", type=["tif", "tiff", "png", "jpg"],
            key="pre_upload"
        )
        if pre_upload:
            st.session_state.pre_path = save_upload(pre_upload)
            st.success(f"Uploaded: {pre_upload.name}")

with tab2:
    st.markdown("#### Use satellite files already downloaded to your project folder")

    # Sample data button
    sample_post = "post_fire_rgb.tif"
    sample_pre  = "pre_fire_nbr.tif"
    if os.path.exists(sample_post) and os.path.exists(sample_pre):
        if st.button("Use Rhodes 2023 Sample Data", type="secondary"):
            st.session_state.post_path = sample_post
            st.session_state.pre_path  = sample_pre
            st.success("Rhodes 2023 sample data loaded — click Analyse below.")
            st.rerun()

    tif_files_all = [f for f in os.listdir(".") if f.endswith(".tif")]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Post-Fire Image**")
        st.caption("Select the RGB or multi-band TIF taken after the fire.")
        post_options = ["-- Select --"] + tif_files_all
        post_select  = st.selectbox("Select post-fire file", post_options, key="post_sel")
        if post_select != "-- Select --":
            st.session_state.post_path = post_select
            st.success(f"Selected: {post_select}")

    with col2:
        st.markdown("**Pre-Fire Image (For NBR)**")
        st.caption("Select the NBR TIF taken before the fire (contains B08 + B12 bands).")
        pre_options = ["-- Select --"] + tif_files_all
        pre_select  = st.selectbox("Select pre-fire file", pre_options, key="pre_sel")
        if pre_select != "-- Select --":
            st.session_state.pre_path = pre_select
            st.success(f"Selected: {pre_select}")

    st.info("Recommended: Post = post_fire_rgb.tif  |  Pre = pre_fire_nbr.tif")

st.markdown("---")

# ── ANALYSE BUTTON ───────────────────────────────────────────
post_path = st.session_state.post_path
pre_path  = st.session_state.pre_path

if post_path:
    if st.button("🔥 Analyse for Wildfire",
                 type="primary", use_container_width=True):

        # STEP 1: Load image
        with st.spinner("Loading satellite image..."):
            try:
                rgb  = load_tif_rgb(post_path)
                h, w = rgb.shape[:2]
                st.success(f"Image loaded — {w}x{h} pixels")
            except Exception as e:
                st.error(f"Error loading image: {e}")
                st.stop()

        st.markdown("### Satellite Image")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(rgb)
        ax.axis("off")
        image_label = os.path.basename(post_path).replace(".tif", "").replace("_", " ").title()
        ax.set_title(f"Post-Fire Satellite Image — {image_label}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # STEP 2: YOLO Detection
        st.markdown("### Step 1 — Fire Detection (YOLOv8)")
        with st.spinner("Running YOLO detection..."):
            detections, conf_scores = run_yolo(rgb, conf_threshold)

        fire_classes  = [d for d in detections if d["class_id"] > 0]
        fire_detected = len(fire_classes) > 0

        if fire_detected:
            st.markdown(
                f'<div class="fire-detected">🔥 FIRE DETECTED — '
                f'{len(detections)} zones classified</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="no-fire">NO FIRE DETECTED</div>',
                unsafe_allow_html=True
            )

        # YOLO map
        class_colors = {
            0: (0.2, 0.8, 0.2), 1: (1.0, 1.0, 0.0),
            2: (1.0, 0.6, 0.0), 3: (1.0, 0.2, 0.0),
            4: (0.1, 0.0, 0.0)
        }
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        ax.imshow(rgb)
        for det in detections:
            color = class_colors.get(det["class_id"], (1, 0, 0))
            rect  = plt.Rectangle(
                (det["x1"], det["y1"]),
                det["x2"] - det["x1"],
                det["y2"] - det["y1"],
                linewidth=2, edgecolor=color,
                facecolor=(*color, 0.2)
            )
            ax.add_patch(rect)
            ax.text(
                det["x1"], det["y1"] - 5,
                f"{det['class_name']} {det['confidence']:.0%}",
                color="white", fontsize=7, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.85, boxstyle="round")
            )
        ax.set_title(
            f"YOLO Detection — {len(detections)} zones | {image_label}",
            fontsize=13, fontweight="bold", color="white"
        )
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button(
            "Download YOLO Map",
            fig_bytes(fig), "yolo_detection.png", "image/png"
        )
        plt.close()

        # Metrics
        if conf_scores:
            col1, col2, col3, col4 = st.columns(4)
            for col, label, val in [
                (col1, "Zones",     str(len(detections))),
                (col2, "Max Conf",  f"{max(conf_scores):.0%}"),
                (col3, "Avg Conf",  f"{np.mean(conf_scores):.0%}"),
                (col4, "Model mAP", "94.1%" if using_trained_model else "N/A"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="metric-box">'
                        f'<h3>{val}</h3><p>{label}</p></div>',
                        unsafe_allow_html=True
                    )

        # STEP 3: NBR Analysis
        if fire_detected and pre_path:
            st.markdown("---")
            st.markdown("### Step 2 — NBR Burn Severity Monitoring")
            st.info("Fire confirmed — running full NBR pipeline...")

            with st.spinner("Running NBR analysis..."):
                try:
                    pre_bands  = load_tif_bands(pre_path)
                    post_bands = load_tif_bands(post_path)
                    pre_nbr, post_nbr, dnbr = calc_nbr(pre_bands, post_bands)
                    classified = classify(dnbr)
                    px   = 0.0001
                    low  = np.sum(classified == 1)
                    mod  = np.sum(classified == 2)
                    high = np.sum(classified == 3)
                    ext  = np.sum(classified == 4)
                    total = (low + mod + high + ext) * px

                    # NBR maps
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    axes[0].imshow(pre_nbr, cmap="RdYlGn", vmin=-1, vmax=1)
                    axes[0].set_title("Pre-Fire NBR")
                    axes[0].axis("off")
                    axes[1].imshow(post_nbr, cmap="RdYlGn", vmin=-1, vmax=1)
                    axes[1].set_title("Post-Fire NBR")
                    axes[1].axis("off")
                    im = axes[2].imshow(dnbr, cmap="hot", vmin=-0.5, vmax=1)
                    axes[2].set_title("Burn Severity (dNBR)")
                    axes[2].axis("off")
                    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
                    plt.suptitle(f"NBR Analysis — {image_label}",
                                 fontsize=14, fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.download_button(
                        "Download NBR Map",
                        fig_bytes(fig), "nbr_analysis.png", "image/png"
                    )
                    plt.close()

                    # Classification map
                    sev_cols = {
                        0: (0.2, 0.6, 0.2), 1: (1.0, 1.0, 0.0),
                        2: (1.0, 0.6, 0.0), 3: (1.0, 0.2, 0.0),
                        4: (0.2, 0.0, 0.0), 255: (0.0, 0.0, 0.0)
                    }
                    rgb_sev = np.zeros((*classified.shape, 3), dtype=float)
                    for v, c in sev_cols.items():
                        rgb_sev[classified == v] = c

                    fig, ax = plt.subplots(figsize=(10, 12))
                    ax.imshow(rgb_sev)
                    ax.set_title(
                        f"Burn Severity Classification — {image_label}",
                        fontsize=14, fontweight="bold"
                    )
                    ax.axis("off")
                    legend = [
                        mpatches.Patch(color=(0.2, 0.6, 0.2), label="Unburned"),
                        mpatches.Patch(color=(1.0, 1.0, 0.0), label=f"Low: {low*px:.0f} km2"),
                        mpatches.Patch(color=(1.0, 0.6, 0.0), label=f"Moderate: {mod*px:.0f} km2"),
                        mpatches.Patch(color=(1.0, 0.2, 0.0), label=f"High: {high*px:.0f} km2"),
                        mpatches.Patch(color=(0.2, 0.0, 0.0), label=f"Extreme: {ext*px:.0f} km2"),
                    ]
                    ax.legend(handles=legend, loc="lower left", fontsize=11)
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.download_button(
                        "Download Classification Map",
                        fig_bytes(fig), "classification.png", "image/png"
                    )
                    plt.close()

                    # Results
                    st.markdown("### Burn Severity Results")
                    cols = st.columns(5)
                    data = [
                        ("Low",      f"{low*px:.0f} km2",   "#ffff00", "#333"),
                        ("Moderate", f"{mod*px:.0f} km2",   "#ff9900", "white"),
                        ("High",     f"{high*px:.0f} km2",  "#ff3300", "white"),
                        ("Extreme",  f"{ext*px:.0f} km2",   "#330000", "white"),
                        ("TOTAL",    f"{total:.0f} km2",    "#cc4400", "white"),
                    ]
                    for col, (label, val, bg, fg) in zip(cols, data):
                        with col:
                            st.markdown(
                                f'<div style="background:{bg};color:{fg};'
                                f'padding:1rem;border-radius:8px;text-align:center;">'
                                f'<h3 style="margin:0;color:{fg}">{val}</h3>'
                                f'<p style="margin:0.3rem 0 0 0;color:{fg}">{label}</p>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                except Exception as e:
                    st.error(f"NBR analysis error: {e}")

        elif fire_detected and not pre_path:
            st.warning(
                "Fire detected! Select a pre-fire image to run the full NBR analysis."
            )
        else:
            st.success("No fire detected. NBR analysis skipped.")

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem;
    background:#f8f9fa;border-radius:10px;">
        <h2>Select a satellite image to begin</h2>
        <p style="color:#666;font-size:1.1rem;">
        Use the tabs above to upload a file or select from existing downloaded data
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### What this app does")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Step 1 — Detection**\nYOLOv8 classifies zones as FIRE or NO FIRE with confidence scores")
    with col2:
        st.markdown("**Step 2 — Monitoring**\nNBR maps burn severity across 5 classes using NASA standards")
    with col3:
        st.markdown("**Step 3 — Results**\nDownload maps and view burned area in km2")

# ── TAB 3: MONITORING DASHBOARD ───────────────────────────────
with tab3:
    st.markdown("## Automatic Monitoring Dashboard")
    st.markdown("Showing results from the daily `wildfire_monitor.py` runs.")

    LOG_FILE     = Path(__file__).parent / "monitoring_log.csv"
    REGIONS_FILE = Path(__file__).parent / "monitor_regions.json"
    MAPS_DIR     = Path(__file__).parent / "monitoring_maps"

    # ── REGION CONFIG EDITOR ──────────────────────────────────
    with st.expander("Manage Monitored Regions"):
        if REGIONS_FILE.exists():
            with open(REGIONS_FILE) as f:
                regions_data = json.load(f)
            regions = regions_data["regions"]

            st.markdown("**Currently monitoring:**")
            for r in regions:
                b = r["bbox"]
                st.markdown(
                    f"- **{r['name']}** — "
                    f"W:{b['west']} E:{b['east']} "
                    f"S:{b['south']} N:{b['north']}"
                )

            st.markdown("---")
            st.markdown("**Add a new region:**")
            new_name = st.text_input("Region name", placeholder="e.g. California USA")
            c1, c2 = st.columns(2)
            with c1:
                new_west  = st.number_input("West longitude", value=-120.0, format="%.4f")
                new_east  = st.number_input("East longitude", value=-118.0, format="%.4f")
            with c2:
                new_south = st.number_input("South latitude", value=35.0, format="%.4f")
                new_north = st.number_input("North latitude", value=37.0, format="%.4f")

            if st.button("Add Region"):
                if not new_name:
                    st.error("Please enter a region name.")
                elif new_west >= new_east:
                    st.error("West longitude must be less than East longitude.")
                elif new_south >= new_north:
                    st.error("South latitude must be less than North latitude.")
                else:
                    regions.append({
                        "name": new_name,
                        "bbox": {
                            "west":  new_west,
                            "east":  new_east,
                            "south": new_south,
                            "north": new_north,
                        }
                    })
                    with open(REGIONS_FILE, "w") as f:
                        json.dump({"regions": regions}, f, indent=2)
                    st.success(f"Region '{new_name}' added!")
                    st.rerun()

            if len(regions) > 0:
                remove_name = st.selectbox(
                    "Remove a region",
                    ["-- Select --"] + [r["name"] for r in regions]
                )
                if st.button("Remove Region") and remove_name != "-- Select --":
                    regions = [r for r in regions if r["name"] != remove_name]
                    with open(REGIONS_FILE, "w") as f:
                        json.dump({"regions": regions}, f, indent=2)
                    st.success(f"Region '{remove_name}' removed.")
                    st.rerun()
        else:
            st.warning("monitor_regions.json not found.")

    st.markdown("---")

    # ── LOG DATA ──────────────────────────────────────────────
    if not LOG_FILE.exists():
        st.info(
            "No monitoring runs yet. "
            "Run `wildfire_monitor.py` to start collecting data."
        )
        st.code("python wildfire_monitor.py", language="bash")
    else:
        rows = []
        with open(LOG_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            st.info("Log file exists but contains no entries yet.")
        else:
            # ── SUMMARY METRICS ───────────────────────────────
            total_runs   = len(rows)
            total_alerts = sum(1 for r in rows if r["fire_detected"] == "True")
            regions_seen = list({r["region"] for r in rows})
            latest_run   = rows[-1]["timestamp"]

            col1, col2, col3, col4 = st.columns(4)
            for col, label, val, color in [
                (col1, "Total Runs",      str(total_runs),        "#1a6e3c"),
                (col2, "Fire Alerts",     str(total_alerts),      "#cc2200"),
                (col3, "Regions Watched", str(len(regions_seen)), "#1a1a8e"),
                (col4, "Last Run",        latest_run[:10],        "#554400"),
            ]:
                with col:
                    st.markdown(
                        f'<div style="background:{color};color:white;'
                        f'padding:1rem;border-radius:8px;text-align:center;">'
                        f'<h3 style="margin:0;color:white">{val}</h3>'
                        f'<p style="margin:0.3rem 0 0 0;color:#ddd">{label}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("---")

            # ── FIRE ALERTS CHART ─────────────────────────────
            alert_rows = [r for r in rows if r["fire_detected"] == "True"]
            if alert_rows:
                st.markdown("### Fire Alerts History")
                dates           = [r["timestamp"][:10] for r in alert_rows]
                regions_alerted = [r["region"] for r in alert_rows]
                burned          = [float(r["total_burned_km2"]) for r in alert_rows]

                fig, ax = plt.subplots(figsize=(12, 4))
                fig.patch.set_facecolor("#1a1a2e")
                ax.set_facecolor("#1a1a2e")
                ax.bar(range(len(alert_rows)), burned,
                       color="#cc4400", edgecolor="#ff6600", linewidth=0.8)
                ax.set_xticks(range(len(alert_rows)))
                ax.set_xticklabels(
                    [f"{d}\n{r}" for d, r in zip(dates, regions_alerted)],
                    rotation=30, ha="right", fontsize=8, color="white"
                )
                ax.set_ylabel("Burned Area (km2)", color="white")
                ax.set_title("Detected Burned Area Over Time",
                             color="white", fontsize=13)
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#444")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.success("No fire alerts recorded — all regions clear.")

            # ── FULL LOG TABLE ────────────────────────────────
            st.markdown("### Full Monitoring Log")
            all_regions   = ["All"] + sorted({r["region"] for r in rows})
            filter_region = st.selectbox("Filter by region", all_regions)
            filtered      = rows if filter_region == "All" else \
                            [r for r in rows if r["region"] == filter_region]

            for row in reversed(filtered[-50:]):
                is_fire = row["fire_detected"] == "True"
                bg    = "#3a0a0a" if is_fire else "#0a2a0a"
                icon  = "FIRE"    if is_fire else "CLEAR"
                badge = "#cc2200" if is_fire else "#1a6e3c"
                st.markdown(
                    f'<div style="background:{bg};padding:0.6rem 1rem;'
                    f'border-radius:6px;margin:0.3rem 0;'
                    f'display:flex;justify-content:space-between;align-items:center;">'
                    f'<span style="color:#aaa;font-size:0.85rem">{row["timestamp"]}</span>'
                    f'<span style="color:white;font-weight:bold">{row["region"]}</span>'
                    f'<span style="color:white">{row["total_burned_km2"]} km2</span>'
                    f'<span style="background:{badge};color:white;'
                    f'padding:0.2rem 0.6rem;border-radius:4px;'
                    f'font-size:0.85rem;font-weight:bold">{icon}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ── LATEST ALERT MAPS ─────────────────────────────
            if MAPS_DIR.exists():
                map_files = sorted(MAPS_DIR.glob("alert_*.png"), reverse=True)
                if map_files:
                    st.markdown("---")
                    st.markdown("### Latest Alert Maps")
                    cols = st.columns(min(3, len(map_files)))
                    for i, mf in enumerate(map_files[:3]):
                        with cols[i]:
                            st.image(str(mf), caption=mf.stem,
                                     use_container_width=True)

            # ── DOWNLOAD LOG ──────────────────────────────────
            st.markdown("---")
            with open(LOG_FILE, "rb") as f:
                st.download_button(
                    "Download Full Log (CSV)",
                    f.read(), "monitoring_log.csv", "text/csv"
                )
