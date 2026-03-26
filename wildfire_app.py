# ============================================================
# WILDFIRE DETECTION & MONITORING SYSTEM
# Streamlit Web Application
# ============================================================

import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO
import cv2
import os
import tempfile
from io import BytesIO

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Wildfire Detection & Monitoring",
    page_icon="🔥",
    layout="wide"
)

# ── CUSTOM STYLING ───────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #cc4400 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        color: #ffcc00;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    .result-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #cc4400;
        margin: 1rem 0;
    }
    .fire-detected {
        background: #fff0f0;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #cc2200;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        color: #cc2200;
    }
    .no-fire {
        background: #f0fff0;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #1a6e3c;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        color: #1a6e3c;
    }
    .metric-box {
        background: #1a1a2e;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .metric-box h3 {
        color: #ffcc00;
        margin: 0;
        font-size: 1.8rem;
    }
    .metric-box p {
        color: #aaaaaa;
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔥 Wildfire Detection & Monitoring System</h1>
    <p>YOLOv8 Deep Learning + Sentinel-2 NBR Spectral Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── LOAD YOLO MODEL ──────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "runs/detect/wildfire_severity_v2/weights/best.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        return YOLO("yolov8n.pt")

model = load_model()

# ── HELPER FUNCTIONS ─────────────────────────────────────────
def load_tif_as_rgb(uploaded_file):
    """Load a .tif file and convert to displayable RGB"""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with rasterio.open(tmp_path) as src:
        bands = src.count
        if bands >= 3:
            R = np.clip(src.read(1).astype(float), 0, 3000) / 3000
            G = np.clip(src.read(2).astype(float), 0, 3000) / 3000
            B = np.clip(src.read(3).astype(float), 0, 3000) / 3000
        else:
            R = np.clip(src.read(1).astype(float), 0, 3000) / 3000
            G = R.copy()
            B = R.copy()
    os.unlink(tmp_path)
    return np.dstack([R, G, B])

def load_tif_bands(uploaded_file):
    """Load all bands from a .tif file"""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with rasterio.open(tmp_path) as src:
        bands = [src.read(i+1).astype(float) for i in range(src.count)]
    os.unlink(tmp_path)
    return bands

def run_yolo_detection(rgb_image):
    """Run YOLO on image tiles and return detections"""
    rgb_uint8 = (rgb_image * 255).astype(np.uint8)
    tile_size = 640
    height, width = rgb_uint8.shape[:2]
    detections = []
    confidence_scores = []
    class_names = ["Unburned","Low","Moderate","High","Extreme"]

    tile_dir = tempfile.mkdtemp()
    tile_paths = []
    tile_num = 0

    for y in range(0, height - tile_size, tile_size):
        for x in range(0, width - tile_size, tile_size):
            tile = rgb_uint8[y:y+tile_size, x:x+tile_size]
            tile_path = os.path.join(tile_dir, f"tile_{tile_num:04d}.png")
            cv2.imwrite(tile_path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            tile_paths.append((tile_path, x, y))
            tile_num += 1

    for tile_path, x_off, y_off in tile_paths:
        results = model(tile_path, verbose=False, conf=0.25)
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls  = int(box.cls[0])
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    if cls < len(class_names):
                        detections.append({
                            "x1":x_off+x1,"y1":y_off+y1,
                            "x2":x_off+x2,"y2":y_off+y2,
                            "confidence":conf,
                            "class_id":cls,
                            "class_name":class_names[cls]
                        })
                        confidence_scores.append(conf)

    import shutil
    shutil.rmtree(tile_dir, ignore_errors=True)
    return detections, confidence_scores

def calculate_nbr(pre_bands, post_bands):
    """Calculate NBR and dNBR from band arrays"""
    pre_B08  = pre_bands[0].copy()
    pre_B12  = pre_bands[1].copy()
    post_B08 = post_bands[0].copy()
    post_B12 = post_bands[1].copy()

    for arr in [pre_B08, pre_B12, post_B08, post_B12]:
        arr[arr == 0] = np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        pre_nbr  = np.where((pre_B08+pre_B12)!=0,
                            (pre_B08-pre_B12)/(pre_B08+pre_B12), np.nan)
        post_nbr = np.where((post_B08+post_B12)!=0,
                            (post_B08-post_B12)/(post_B08+post_B12), np.nan)
    return pre_nbr, post_nbr, pre_nbr - post_nbr

def classify_severity(dnbr):
    """Classify dNBR into severity classes"""
    classified = np.zeros_like(dnbr, dtype=np.uint8)
    classified[dnbr < 0.1]                     = 0
    classified[(dnbr>=0.1) & (dnbr<0.27)]      = 1
    classified[(dnbr>=0.27) & (dnbr<0.44)]     = 2
    classified[(dnbr>=0.44) & (dnbr<0.66)]     = 3
    classified[dnbr >= 0.66]                    = 4
    classified[np.isnan(dnbr)]                  = 255
    return classified

def fig_to_bytes(fig):
    """Convert matplotlib figure to bytes for download"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/"
             "ESA_logo_simple.svg/200px-ESA_logo_simple.svg.png", width=100)
    st.markdown("### About")
    st.info(
        "This system uses **YOLOv8** deep learning and "
        "**Sentinel-2 NBR** spectral analysis to detect "
        "and monitor wildfires from satellite imagery."
    )
    st.markdown("### How to Use")
    st.markdown("""
    1. Upload a **satellite image** (.tif or .png)
    2. For full NBR analysis, upload **pre and post fire** band files
    3. Click **Analyse**
    4. View results and download maps
    """)
    st.markdown("### Model Info")
    st.success("YOLOv8 — mAP: 94.1%")
    st.success("NBR Accuracy: Within 2-5%")

# ── MAIN APP ─────────────────────────────────────────────────
st.markdown("## Upload Satellite Image")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Post-Fire Image (Required)")
    post_file = st.file_uploader(
        "Upload post-fire satellite image",
        type=["tif","tiff","png","jpg"],
        key="post"
    )

with col2:
    st.markdown("#### Pre-Fire Image (For NBR Analysis)")
    pre_file = st.file_uploader(
        "Upload pre-fire satellite image",
        type=["tif","tiff","png","jpg"],
        key="pre"
    )

st.markdown("---")

# ── ANALYSE BUTTON ───────────────────────────────────────────
if post_file is not None:
    if st.button("🔥 Analyse for Wildfire", type="primary",
                 use_container_width=True):

        # ── STEP 1: LOAD IMAGE ───────────────────────────────
        with st.spinner("Loading satellite image..."):
            post_file.seek(0)
            rgb_image = load_tif_as_rgb(post_file)
            st.success(f"Image loaded — {rgb_image.shape[1]}x"
                      f"{rgb_image.shape[0]} pixels")

        # Show the uploaded image
        st.markdown("### Satellite Image")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(rgb_image)
        ax.axis("off")
        ax.set_title("Post-Fire Satellite Image", fontsize=14,
                    fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── STEP 2: YOLO DETECTION ───────────────────────────
        st.markdown("### Step 1 — Fire Detection (YOLOv8)")
        with st.spinner("Running YOLO fire detection..."):
            detections, conf_scores = run_yolo_detection(rgb_image)

        fire_classes  = [d for d in detections if d["class_id"] > 0]
        fire_detected = len(fire_classes) > 0

        # Show detection result
        if fire_detected:
            st.markdown(
                f'<div class="fire-detected">🔥 FIRE DETECTED — '
                f'{len(detections)} zones classified</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="no-fire">✅ NO FIRE DETECTED</div>',
                unsafe_allow_html=True
            )

        # Show YOLO detection map
        class_colors = {
            0:(0.2,0.8,0.2), 1:(1.0,1.0,0.0),
            2:(1.0,0.6,0.0), 3:(1.0,0.2,0.0),
            4:(0.1,0.0,0.0)
        }
        class_names = ["Unburned","Low","Moderate","High","Extreme"]

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(rgb_image)
        for det in detections:
            color = class_colors.get(det["class_id"], (1,0,0))
            rect  = plt.Rectangle(
                (det["x1"],det["y1"]),
                det["x2"]-det["x1"],
                det["y2"]-det["y1"],
                linewidth=2, edgecolor=color,
                facecolor=(*color,0.2)
            )
            ax.add_patch(rect)
            ax.text(
                det["x1"], det["y1"]-5,
                f"{det['class_name']} {det['confidence']:.0%}",
                color="white", fontsize=7, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.85, boxstyle="round")
            )
        ax.set_title(
            f"YOLO Detection — {len(detections)} zones detected",
            fontsize=13, fontweight="bold"
        )
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)

        yolo_buf = fig_to_bytes(fig)
        st.download_button(
            "Download YOLO Detection Map",
            yolo_buf, "yolo_detection.png", "image/png"
        )
        plt.close()

        # Show metrics
        if conf_scores:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(
                    f'<div class="metric-box"><h3>'
                    f'{len(detections)}</h3>'
                    f'<p>Zones Detected</p></div>',
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f'<div class="metric-box"><h3>'
                    f'{max(conf_scores):.0%}</h3>'
                    f'<p>Max Confidence</p></div>',
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    f'<div class="metric-box"><h3>'
                    f'{np.mean(conf_scores):.0%}</h3>'
                    f'<p>Avg Confidence</p></div>',
                    unsafe_allow_html=True
                )
            with col4:
                st.markdown(
                    f'<div class="metric-box"><h3>'
                    f'94.1%</h3>'
                    f'<p>Model mAP</p></div>',
                    unsafe_allow_html=True
                )

        # ── STEP 3: NBR ANALYSIS (only if fire detected) ─────
        if fire_detected and pre_file is not None:
            st.markdown("---")
            st.markdown("### Step 2 — NBR Burn Severity Analysis")
            st.info("Fire detected — running full NBR monitoring pipeline...")

            with st.spinner("Running NBR analysis..."):
                pre_file.seek(0)
                post_file.seek(0)
                pre_bands  = load_tif_bands(pre_file)
                post_bands = load_tif_bands(post_file)

                if len(pre_bands) >= 2 and len(post_bands) >= 2:
                    pre_nbr, post_nbr, dnbr = calculate_nbr(
                        pre_bands, post_bands
                    )
                    classified = classify_severity(dnbr)
                    pixel_km2  = 0.0001

                    low      = np.sum(classified==1)
                    moderate = np.sum(classified==2)
                    high     = np.sum(classified==3)
                    extreme  = np.sum(classified==4)
                    total    = (low+moderate+high+extreme)*pixel_km2

                    # NBR maps
                    fig, axes = plt.subplots(1, 3, figsize=(18,6))
                    axes[0].imshow(pre_nbr, cmap="RdYlGn",
                                  vmin=-1, vmax=1)
                    axes[0].set_title("Pre-Fire NBR")
                    axes[0].axis("off")
                    axes[1].imshow(post_nbr, cmap="RdYlGn",
                                  vmin=-1, vmax=1)
                    axes[1].set_title("Post-Fire NBR")
                    axes[1].axis("off")
                    im = axes[2].imshow(dnbr, cmap="hot",
                                       vmin=-0.5, vmax=1)
                    axes[2].set_title("Burn Severity (dNBR)")
                    axes[2].axis("off")
                    plt.colorbar(im, ax=axes[2],
                                fraction=0.046, pad=0.04)
                    plt.suptitle("NBR Analysis",
                                fontsize=14, fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig)
                    nbr_buf = fig_to_bytes(fig)
                    st.download_button(
                        "Download NBR Analysis Map",
                        nbr_buf, "nbr_analysis.png", "image/png"
                    )
                    plt.close()

                    # Classification map
                    sev_colors = {
                        0:(0.2,0.6,0.2), 1:(1.0,1.0,0.0),
                        2:(1.0,0.6,0.0), 3:(1.0,0.2,0.0),
                        4:(0.2,0.0,0.0), 255:(0.0,0.0,0.0)
                    }
                    rgb_sev = np.zeros(
                        (*classified.shape,3), dtype=float
                    )
                    for val, col in sev_colors.items():
                        rgb_sev[classified==val] = col

                    fig, ax = plt.subplots(figsize=(10,12))
                    ax.imshow(rgb_sev)
                    ax.set_title(
                        "Burn Severity Classification Map",
                        fontsize=14, fontweight="bold"
                    )
                    ax.axis("off")
                    legend_items = [
                        mpatches.Patch(
                            color=(0.2,0.6,0.2),
                            label=f"Unburned"
                        ),
                        mpatches.Patch(
                            color=(1.0,1.0,0.0),
                            label=f"Low: {low*pixel_km2:.0f} km2"
                        ),
                        mpatches.Patch(
                            color=(1.0,0.6,0.0),
                            label=f"Moderate: "
                                  f"{moderate*pixel_km2:.0f} km2"
                        ),
                        mpatches.Patch(
                            color=(1.0,0.2,0.0),
                            label=f"High: {high*pixel_km2:.0f} km2"
                        ),
                        mpatches.Patch(
                            color=(0.2,0.0,0.0),
                            label=f"Extreme: "
                                  f"{extreme*pixel_km2:.0f} km2"
                        ),
                    ]
                    ax.legend(
                        handles=legend_items,
                        loc="lower left", fontsize=11
                    )
                    plt.tight_layout()
                    st.pyplot(fig)
                    cls_buf = fig_to_bytes(fig)
                    st.download_button(
                        "Download Classification Map",
                        cls_buf, "classification_map.png",
                        "image/png"
                    )
                    plt.close()

                    # Results metrics
                    st.markdown("### Burn Severity Results")
                    col1,col2,col3,col4,col5 = st.columns(5)
                    metrics = [
                        (col1,"Low",f"{low*pixel_km2:.0f} km2",
                         "#ffff00","#333"),
                        (col2,"Moderate",
                         f"{moderate*pixel_km2:.0f} km2",
                         "#ff9900","white"),
                        (col3,"High",f"{high*pixel_km2:.0f} km2",
                         "#ff3300","white"),
                        (col4,"Extreme",
                         f"{extreme*pixel_km2:.0f} km2",
                         "#330000","white"),
                        (col5,"TOTAL",f"{total:.0f} km2",
                         "#cc4400","white"),
                    ]
                    for col,label,val,bg,fg in metrics:
                        with col:
                            st.markdown(
                                f'<div style="background:{bg};'
                                f'color:{fg};padding:1rem;'
                                f'border-radius:8px;'
                                f'text-align:center;">'
                                f'<h3 style="margin:0;'
                                f'color:{fg}">{val}</h3>'
                                f'<p style="margin:0.3rem 0 0 0;'
                                f'color:{fg}">{label}</p>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                else:
                    st.warning(
                        "Pre-fire image needs at least 2 bands "
                        "for NBR analysis. Please upload a "
                        "multi-band .tif file."
                    )

        elif fire_detected and pre_file is None:
            st.warning(
                "Fire detected! Upload a pre-fire image in the "
                "sidebar to run the full NBR burn severity analysis."
            )

        elif not fire_detected:
            st.success(
                "No fire detected in this image. "
                "NBR analysis skipped."
            )

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align:center; padding: 3rem;
    background:#f8f9fa; border-radius:10px;">
        <h2>Upload a satellite image to begin</h2>
        <p style="color:#666; font-size:1.1rem;">
        Supports .tif, .tiff, .png, .jpg files<br>
        For full analysis upload both pre-fire
        and post-fire images
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show example info
    st.markdown("### What this app does")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Step 1 — Detection**
        YOLOv8 AI analyses the image
        and classifies each zone as
        FIRE or NO FIRE with a
        confidence score
        """)
    with col2:
        st.markdown("""
        **Step 2 — Monitoring**
        If fire is detected, NBR
        spectral analysis maps the
        burn severity across 5
        classes using NASA standards
        """)
    with col3:
        st.markdown("""
        **Step 3 — Results**
        Download maps, view burned
        area in km2, and get a full
        breakdown of severity levels
        """)