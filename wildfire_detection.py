# ============================================================
# WILDFIRE DETECTION & MONITORING SYSTEM
# Using Sentinel-2 Satellite Imagery & NBR Analysis
# Data: Copernicus Data Space Ecosystem
# ============================================================

import openeo
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import datetime

# ── STEP 1: CONNECT TO COPERNICUS ───────────────────────────
connection = openeo.connect("openeo.dataspace.copernicus.eu")
connection.authenticate_oidc()
print("Connected and authenticated successfully!")

# ── STEP 2: DEFINE STUDY AREA & TIME PERIODS ────────────────
spatial_extent = {
    "west": 27.7, "east": 28.3,
    "south": 35.9, "north": 36.5
}
pre_fire  = ["2023-07-01", "2023-07-17"]
post_fire = ["2023-07-25", "2023-08-15"]
print("Area of interest set: Rhodes, Greece")

# ── STEP 3: DOWNLOAD SATELLITE DATA ─────────────────────────
pre_fire_cube = connection.load_collection(
    "SENTINEL2_L2A",
    temporal_extent=pre_fire,
    spatial_extent=spatial_extent,
    bands=["B08", "B12"],
    max_cloud_cover=20
).reduce_dimension(dimension="t", reducer="mean")

post_fire_cube = connection.load_collection(
    "SENTINEL2_L2A",
    temporal_extent=post_fire,
    spatial_extent=spatial_extent,
    bands=["B08", "B12"],
    max_cloud_cover=20
).reduce_dimension(dimension="t", reducer="mean")

pre_NBR = (pre_fire_cube.band("B08") - pre_fire_cube.band("B12")) / \
          (pre_fire_cube.band("B08") + pre_fire_cube.band("B12"))
post_NBR = (post_fire_cube.band("B08") - post_fire_cube.band("B12")) / \
           (post_fire_cube.band("B08") + post_fire_cube.band("B12"))

print("Downloading pre-fire NBR...")
pre_NBR.download("pre_fire_nbr.tif")
print("Downloading post-fire NBR...")
post_NBR.download("post_fire_nbr.tif")
print("NBR data downloaded!")

# ── STEP 4: LOAD & VISUALISE NBR ────────────────────────────
with rasterio.open("pre_fire_nbr.tif") as src:
    pre_nbr = src.read(1).astype(float)
with rasterio.open("post_fire_nbr.tif") as src:
    post_nbr = src.read(1).astype(float)

pre_nbr[pre_nbr == 0]   = np.nan
post_nbr[post_nbr == 0] = np.nan
dnbr = pre_nbr - post_nbr

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(pre_nbr,  cmap="RdYlGn", vmin=-1, vmax=1)
axes[0].set_title("Pre-Fire NBR\n(Green = Healthy Vegetation)"); axes[0].axis("off")
axes[1].imshow(post_nbr, cmap="RdYlGn", vmin=-1, vmax=1)
axes[1].set_title("Post-Fire NBR\n(Red = Burned Areas)");       axes[1].axis("off")
im = axes[2].imshow(dnbr, cmap="hot", vmin=-0.5, vmax=1)
axes[2].set_title("Burn Severity Map (dNBR)");                   axes[2].axis("off")
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
plt.suptitle("Rhodes, Greece Wildfire 2023 - Sentinel-2 Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("wildfire_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# ── STEP 5: BURN SEVERITY CLASSIFICATION MODEL ───────────────
pixel_area_km2 = 0.0001
classified = np.zeros_like(dnbr, dtype=np.uint8)
classified[dnbr < 0.1]                     = 0
classified[(dnbr >= 0.1) & (dnbr < 0.27)]  = 1
classified[(dnbr >= 0.27) & (dnbr < 0.44)] = 2
classified[(dnbr >= 0.44) & (dnbr < 0.66)] = 3
classified[dnbr >= 0.66]                    = 4
classified[np.isnan(dnbr)]                  = 255

unburned = np.sum(classified == 0)
low      = np.sum(classified == 1)
moderate = np.sum(classified == 2)
high     = np.sum(classified == 3)
extreme  = np.sum(classified == 4)
total_burned = (low + moderate + high + extreme) * pixel_area_km2

print("\nWILDFIRE DETECTION RESULTS")
print("=" * 40)
print(f"Unburned:          {unburned  * pixel_area_km2:.2f} km2")
print(f"Low severity:      {low       * pixel_area_km2:.2f} km2")
print(f"Moderate severity: {moderate  * pixel_area_km2:.2f} km2")
print(f"High severity:     {high      * pixel_area_km2:.2f} km2")
print(f"Extreme severity:  {extreme   * pixel_area_km2:.2f} km2")
print("=" * 40)
print(f"TOTAL BURNED:      {total_burned:.2f} km2")

# ── STEP 6: CLASSIFICATION MAP ───────────────────────────────
colors = {
    0: (0.2, 0.6, 0.2), 1: (1.0, 1.0, 0.0),
    2: (1.0, 0.6, 0.0), 3: (1.0, 0.2, 0.0),
    4: (0.2, 0.0, 0.0), 255: (1.0, 1.0, 1.0),
}
rgb = np.zeros((*classified.shape, 3), dtype=float)
for value, color in colors.items():
    rgb[classified == value] = color

fig, ax = plt.subplots(figsize=(10, 12))
ax.imshow(rgb)
ax.set_title("Rhodes Wildfire 2023 - Burn Severity Classification", fontsize=14, fontweight="bold")
ax.axis("off")
legend_items = [
    mpatches.Patch(color=(0.2,0.6,0.2), label=f"Unburned         ({unburned  * pixel_area_km2:.0f} km2)"),
    mpatches.Patch(color=(1.0,1.0,0.0), label=f"Low Severity     ({low       * pixel_area_km2:.0f} km2)"),
    mpatches.Patch(color=(1.0,0.6,0.0), label=f"Moderate Severity ({moderate * pixel_area_km2:.0f} km2)"),
    mpatches.Patch(color=(1.0,0.2,0.0), label=f"High Severity    ({high      * pixel_area_km2:.0f} km2)"),
    mpatches.Patch(color=(0.2,0.0,0.0), label=f"Extreme Severity ({extreme   * pixel_area_km2:.0f} km2)"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=11, title="Burn Severity Classes")
plt.tight_layout()
plt.savefig("wildfire_classification_map.png", dpi=150, bbox_inches="tight")
plt.show()

# ── STEP 7: NDVI VEGETATION ANALYSIS ────────────────────────
pre_ndvi_cube = connection.load_collection(
    "SENTINEL2_L2A", temporal_extent=pre_fire,
    spatial_extent=spatial_extent, bands=["B04", "B08"], max_cloud_cover=20
).reduce_dimension(dimension="t", reducer="mean")
post_ndvi_cube = connection.load_collection(
    "SENTINEL2_L2A", temporal_extent=post_fire,
    spatial_extent=spatial_extent, bands=["B04", "B08"], max_cloud_cover=20
).reduce_dimension(dimension="t", reducer="mean")
pre_ndvi_cube.download("pre_fire_ndvi_bands.tif")
post_ndvi_cube.download("post_fire_ndvi_bands.tif")

with rasterio.open("pre_fire_ndvi_bands.tif") as src:
    pre_B04 = src.read(1).astype(float); pre_B08 = src.read(2).astype(float)
with rasterio.open("post_fire_ndvi_bands.tif") as src:
    post_B04 = src.read(1).astype(float); post_B08 = src.read(2).astype(float)

for arr in [pre_B04, pre_B08, post_B04, post_B08]:
    arr[arr == 0] = np.nan

pre_ndvi  = (pre_B08  - pre_B04)  / (pre_B08  + pre_B04)
post_ndvi = (post_B08 - post_B04) / (post_B08 + post_B04)
ndvi_loss = pre_ndvi - post_ndvi

print(f"\nPre-fire avg NDVI:  {np.nanmean(pre_ndvi):.3f}")
print(f"Post-fire avg NDVI: {np.nanmean(post_ndvi):.3f}")
print(f"Average NDVI loss:  {np.nanmean(ndvi_loss):.3f}")

# ── STEP 8: TRUE COLOUR RGB ──────────────────────────────────
pre_rgb_cube = connection.load_collection(
    "SENTINEL2_L2A", temporal_extent=pre_fire,
    spatial_extent=spatial_extent, bands=["B04","B03","B02"], max_cloud_cover=20
).reduce_dimension(dimension="t", reducer="mean")
post_rgb_cube = connection.load_collection(
    "SENTINEL2_L2A", temporal_extent=post_fire,
    spatial_extent=spatial_extent, bands=["B04","B03","B02"], max_cloud_cover=20
).reduce_dimension(dimension="t", reducer="mean")
pre_rgb_cube.download("pre_fire_rgb.tif")
post_rgb_cube.download("post_fire_rgb.tif")

# ── STEP 9: MODEL VALIDATION - 3 WILDFIRES ───────────────────
def analyze_wildfire(connection, name, spatial_extent, pre_period, post_period):
    print(f"\nAnalysing: {name}")
    pre_cube = connection.load_collection(
        "SENTINEL2_L2A", temporal_extent=pre_period,
        spatial_extent=spatial_extent, bands=["B08","B12"], max_cloud_cover=20
    ).reduce_dimension(dimension="t", reducer="mean")
    post_cube = connection.load_collection(
        "SENTINEL2_L2A", temporal_extent=post_period,
        spatial_extent=spatial_extent, bands=["B08","B12"], max_cloud_cover=20
    ).reduce_dimension(dimension="t", reducer="mean")
    pre_cube.download(f"{name}_pre.tif")
    post_cube.download(f"{name}_post.tif")
    with rasterio.open(f"{name}_pre.tif") as src:
        B08 = src.read(1).astype(float); B12 = src.read(2).astype(float)
    B08[B08==0]=np.nan; B12[B12==0]=np.nan
    pre_nbr = (B08-B12)/(B08+B12)
    with rasterio.open(f"{name}_post.tif") as src:
        B08 = src.read(1).astype(float); B12 = src.read(2).astype(float)
    B08[B08==0]=np.nan; B12[B12==0]=np.nan
    post_nbr = (B08-B12)/(B08+B12)
    dnbr = pre_nbr - post_nbr
    low      = np.sum((dnbr>=0.1)  & (dnbr<0.27))
    moderate = np.sum((dnbr>=0.27) & (dnbr<0.44))
    high     = np.sum((dnbr>=0.44) & (dnbr<0.66))
    extreme  = np.sum(dnbr>=0.66)
    total    = (low+moderate+high+extreme)*0.0001
    print(f"  Total burned: {total:.2f} km2")
    return dnbr, total, low, moderate, high, extreme

analyze_wildfire(connection, "Evros_Greece",
    {"west":26.0,"east":26.8,"south":41.0,"north":41.8},
    ["2023-07-01","2023-08-15"], ["2023-08-25","2023-09-15"])

analyze_wildfire(connection, "Tenerife_Spain",
    {"west":-16.9,"east":-16.1,"south":28.0,"north":28.6},
    ["2023-07-01","2023-08-14"], ["2023-08-20","2023-09-10"])

print("\nAll analysis complete!")
