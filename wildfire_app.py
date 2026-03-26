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
import tempfile
import shutil
from io import BytesIO

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Wildfire Detection & Monitoring",
    page_icon="🔥",
    layout="wide"
)

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
        return YOLO(model_path)
    return YOLO("yolov8n.pt")

model = load_model()

# ── HELPER FUNCTIONS ─────────────────────────────────────────
def load_tif_rgb(tif_path):
    """Load a .tif file as RGB for display"""
    with rasterio.open(tif_path) as src:
        bands = src.count
        if bands >= 3:
            R = np.clip(src.read(1).astype(float), 0, 3000) / 3000
            G = np.clip(src.read(2).astype(float), 0, 3000) / 3000
            B = np.clip(src.read(3).astype(float), 0, 3000) / 3000
        else:
            R = np.clip(src.read(1).astype(float), 0, 3000) / 3000
            G = R.copy()
            B = R.copy()
    return np.dstack([R, G, B])

def load_tif_bands(tif_path):
    """Load all bands from a .tif file"""
    with rasterio.open(tif_path) as src:
        return [src.read(i+1).astype(float)
                for i in range(src.count)]

def save_upload(uploaded_file):
    """Save uploaded file to temp location and return path"""
    suffix = "." + uploaded_file.name.split(".")[-1]
    tmp = tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False
    )
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name

def run_yolo(rgb_image):
    """Run YOLO detection on image"""
    rgb_uint8 = (rgb_image * 255).astype(np.uint8)
    tile_size = 640
    height, width = rgb_uint8.shape[:2]
    detections = []
    conf_scores = []
    class_names = ["Unburned","Low","Moderate","High","Extreme"]
    tile_dir = tempfile.mkdtemp()
    tile_paths = []
    tile_num = 0

    for y in range(0, height - tile_size, tile_size):
        for x in range(0, width - tile_size, tile_size):
            tile = rgb_uint8[y:y+tile_size, x:x+tile_size]
            tp = os.path.join(tile_dir, f"t_{tile_num:04d}.png")
            cv2.imwrite(tp, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            tile_paths.append((tp, x, y))
            tile_num += 1

    for tp, xo, yo in tile_paths:
        results = model(tp, verbose=False, conf=0.25)
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls  = int(box.cls[0])
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    if cls < len(class_names):
                        detections.append({
                            "x1":xo+x1,"y1":yo+y1,
                            "x2":xo+x2,"y2":yo+y2,
                            "confidence":conf,
                            "class_id":cls,
                            "class_name":class_names[cls]
                        })
                        conf_scores.append(conf)

    shutil.rmtree(tile_dir, ignore_errors=True)
    return detections, conf_scores

def calc_nbr(pre_bands, post_bands):
    """Calculate NBR and dNBR"""
    b08_pre  = pre_bands[0].copy()
    b12_pre  = pre_bands[1].copy() if len(pre_bands)>1 else pre_bands[0].copy()
    b08_post = post_bands[0].copy()
    b12_post = post_bands[1].copy() if len(post_bands)>1 else post_bands[0].copy()

    for a in [b08_pre, b12_pre, b08_post, b12_post]:
        a[a==0] = np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        pre  = np.where((b08_pre+b12_pre)!=0,
                        (b08_pre-b12_pre)/(b08_pre+b12_pre), np.nan)
        post = np.where((b08_post+b12_post)!=0,
                        (b08_post-b12_post)/(b08_post+b12_post), np.nan)
    return pre, post, pre - post

def classify(dnbr):
    """Classify dNBR into severity classes"""
    c = np.zeros_like(dnbr, dtype=np.uint8)
    c[dnbr < 0.1]                   = 0
    c[(dnbr>=0.1) & (dnbr<0.27)]    = 1
    c[(dnbr>=0.27) & (dnbr<0.44)]   = 2
    c[(dnbr>=0.44) & (dnbr<0.66)]   = 3
    c[dnbr >= 0.66]                  = 4
    c[np.isnan(dnbr)]                = 255
    return c

def fig_bytes(fig):
    """Convert figure to downloadable bytes"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.info("YOLOv8 + Sentinel-2 NBR analysis for wildfire detection and monitoring.")
    st.markdown("### How to Use")
    st.markdown("""
    **Option A — Upload files:**
    Upload .tif files directly

    **Option B — Use existing files:**
    Select from already downloaded satellite data
    """)
    st.markdown("### Model Info")
    st.success("YOLOv8 mAP: 94.1%")
    st.success("NBR Accuracy: Within 2-5%")

# ── MAIN APP ─────────────────────────────────────────────────
st.markdown("## Select Satellite Images")

# Two options — upload or use existing
tab1, tab2 = st.tabs(["Upload .tif Files", "Use Existing Downloaded Files"])

post_path = None
pre_path  = None

with tab1:
    st.markdown("#### Upload your own .tif satellite images")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Post-Fire Image (Required)**")
        post_upload = st.file_uploader(
            "Post-fire .tif", type=["tif","tiff","png","jpg"],
            key="post_upload"
        )
        if post_upload:
            post_path = save_upload(post_upload)
            st.success(f"Uploaded: {post_upload.name}")

    with col2:
        st.markdown("**Pre-Fire Image (For NBR)**")
        pre_upload = st.file_uploader(
            "Pre-fire .tif", type=["tif","tiff","png","jpg"],
            key="pre_upload"
        )
        if pre_upload:
            pre_path = save_upload(pre_upload)
            st.success(f"Uploaded: {pre_upload.name}")

with tab2:
    st.markdown("#### Use satellite files already downloaded to your project folder")

    # Find available tif files
    tif_files = [f for f in os.listdir(".")
                 if f.endswith(".tif") and "nbr" not in f.lower()
                 and "ndvi" not in f.lower()]
    tif_files_all = [f for f in os.listdir(".") if f.endswith(".tif")]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Post-Fire Image**")
        post_options = ["-- Select --"] + tif_files_all
        post_select = st.selectbox(
            "Select post-fire file", post_options, key="post_sel"
        )
        if post_select != "-- Select --":
            post_path = post_select
            st.success(f"Selected: {post_select}")

    with col2:
        st.markdown("**Pre-Fire Image (For NBR)**")
        pre_options = ["-- Select --"] + tif_files_all
        pre_select = st.selectbox(
            "Select pre-fire file", pre_options, key="pre_sel"
        )
        if pre_select != "-- Select --":
            pre_path = pre_select
            st.success(f"Selected: {pre_select}")

    st.info(
        "Recommended: Post = post_fire_rgb.tif  |  "
        "Pre = pre_fire_nbr.tif"
    )

st.markdown("---")

# ── ANALYSE BUTTON ───────────────────────────────────────────
if post_path:
    if st.button("🔥 Analyse for Wildfire",
                 type="primary", use_container_width=True):

        # STEP 1: Load image
        with st.spinner("Loading satellite image..."):
            try:
                rgb = load_tif_rgb(post_path)
                h, w = rgb.shape[:2]
                st.success(f"Image loaded — {w}x{h} pixels")
            except Exception as e:
                st.error(f"Error loading image: {e}")
                st.stop()

        # Show image
        st.markdown("### Satellite Image")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title("Post-Fire Satellite Image",
                    fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # STEP 2: YOLO Detection
        st.markdown("### Step 1 — Fire Detection (YOLOv8)")
        with st.spinner("Running YOLO — this may take 1-2 minutes for large images..."):
            detections, conf_scores = run_yolo(rgb)

        fire_classes  = [d for d in detections if d["class_id"]>0]
        fire_detected = len(fire_classes) > 0

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

        # YOLO map
        class_colors = {
            0:(0.2,0.8,0.2), 1:(1.0,1.0,0.0),
            2:(1.0,0.6,0.0), 3:(1.0,0.2,0.0),
            4:(0.1,0.0,0.0)
        }
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        ax.imshow(rgb)
        for det in detections:
            color = class_colors.get(det["class_id"],(1,0,0))
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
                bbox=dict(facecolor=color,alpha=0.85,
                         boxstyle="round")
            )
        ax.set_title(
            f"YOLO Detection — {len(detections)} zones",
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
            col1,col2,col3,col4 = st.columns(4)
            for col, label, val in [
                (col1,"Zones", str(len(detections))),
                (col2,"Max Conf", f"{max(conf_scores):.0%}"),
                (col3,"Avg Conf", f"{np.mean(conf_scores):.0%}"),
                (col4,"Model mAP", "94.1%"),
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
                    pre_nbr, post_nbr, dnbr = calc_nbr(
                        pre_bands, post_bands
                    )
                    classified = classify(dnbr)
                    px = 0.0001
                    low  = np.sum(classified==1)
                    mod  = np.sum(classified==2)
                    high = np.sum(classified==3)
                    ext  = np.sum(classified==4)
                    total = (low+mod+high+ext)*px

                    # NBR maps
                    fig, axes = plt.subplots(1,3,figsize=(18,6))
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
                    plt.suptitle("NBR Analysis — Rhodes 2023",
                                fontsize=14, fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.download_button(
                        "Download NBR Map",
                        fig_bytes(fig),
                        "nbr_analysis.png", "image/png"
                    )
                    plt.close()

                    # Classification map
                    sev_cols = {
                        0:(0.2,0.6,0.2), 1:(1.0,1.0,0.0),
                        2:(1.0,0.6,0.0), 3:(1.0,0.2,0.0),
                        4:(0.2,0.0,0.0), 255:(0.0,0.0,0.0)
                    }
                    rgb_sev = np.zeros(
                        (*classified.shape,3), dtype=float
                    )
                    for v,c in sev_cols.items():
                        rgb_sev[classified==v] = c

                    fig, ax = plt.subplots(figsize=(10,12))
                    ax.imshow(rgb_sev)
                    ax.set_title(
                        "Burn Severity Classification",
                        fontsize=14, fontweight="bold"
                    )
                    ax.axis("off")
                    legend = [
                        mpatches.Patch(color=(0.2,0.6,0.2),
                                      label="Unburned"),
                        mpatches.Patch(color=(1.0,1.0,0.0),
                                      label=f"Low: {low*px:.0f} km2"),
                        mpatches.Patch(color=(1.0,0.6,0.0),
                                      label=f"Moderate: {mod*px:.0f} km2"),
                        mpatches.Patch(color=(1.0,0.2,0.0),
                                      label=f"High: {high*px:.0f} km2"),
                        mpatches.Patch(color=(0.2,0.0,0.0),
                                      label=f"Extreme: {ext*px:.0f} km2"),
                    ]
                    ax.legend(handles=legend, loc="lower left",
                             fontsize=11)
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.download_button(
                        "Download Classification Map",
                        fig_bytes(fig),
                        "classification.png", "image/png"
                    )
                    plt.close()

                    # Results
                    st.markdown("### Burn Severity Results")
                    cols = st.columns(5)
                    data = [
                        ("Low",f"{low*px:.0f} km2","#ffff00","#333"),
                        ("Moderate",f"{mod*px:.0f} km2","#ff9900","white"),
                        ("High",f"{high*px:.0f} km2","#ff3300","white"),
                        ("Extreme",f"{ext*px:.0f} km2","#330000","white"),
                        ("TOTAL",f"{total:.0f} km2","#cc4400","white"),
                    ]
                    for col,(label,val,bg,fg) in zip(cols,data):
                        with col:
                            st.markdown(
                                f'<div style="background:{bg};'
                                f'color:{fg};padding:1rem;'
                                f'border-radius:8px;'
                                f'text-align:center;">'
                                f'<h3 style="margin:0;color:{fg}">'
                                f'{val}</h3>'
                                f'<p style="margin:0.3rem 0 0 0;'
                                f'color:{fg}">{label}</p></div>',
                                unsafe_allow_html=True
                            )

                except Exception as e:
                    st.error(f"NBR analysis error: {e}")

        elif fire_detected and not pre_path:
            st.warning(
                "Fire detected! Select a pre-fire image "
                "to run the full NBR analysis."
            )
        else:
            st.success(
                "No fire detected. NBR analysis skipped."
            )

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem;
    background:#f8f9fa;border-radius:10px;">
        <h2>Select a satellite image to begin</h2>
        <p style="color:#666;font-size:1.1rem;">
        Use the tabs above to upload a file or
        select from existing downloaded data
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### What this app does")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown("**Step 1 — Detection**\nYOLOv8 classifies zones as FIRE or NO FIRE with confidence scores")
    with col2:
        st.markdown("**Step 2 — Monitoring**\nNBR maps burn severity across 5 classes using NASA standards")
    with col3:
        st.markdown("**Step 3 — Results**\nDownload maps and view burned area in km2")