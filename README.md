# Industrial Environmental Compliance Monitoring System

**ML Course Project — Statement 3 | IIIT Dharwad**

An automated pipeline for detecting green cover loss and unauthorized construction in industrial zones using multi-temporal Sentinel-2 satellite imagery, spectral analysis, and machine learning.

---

## Results Summary

| Zone | Green Cover 2020 | Green Cover 2024 | Change | Risk Level |
|------|:----------------:|:----------------:|:------:|:----------:|
| **Peenya** | 52.57% | 28.51% | −24.06% | MEDIUM |
| **Whitefield** | 71.25% | 43.45% | −27.80% | MEDIUM |

| ML Model | Peenya | Whitefield |
|----------|:------:|:----------:|
| Accuracy | 97.74% | 98.70% |
| F1-Score | 97.83% | 98.73% |

## Sample Outputs

| Before/After Analysis | Change Detection | Confusion Matrix |
|:---------------------:|:----------------:|:----------------:|
| ![Before-After](results/before_after_peenya.png) | ![Violations](results/annotated_violations_peenya.png) | ![CM](results/confusion_matrix_peenya.png) |

---

## Quick Start — View the Dashboard

> **All data, trained models, and results are included in the repository.** You do not need to download satellite data or run the ML pipeline to view the dashboard.

### Prerequisites

- Python 3.9 or higher
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/DagarMihir/industrial-compliance-monitoring.git
cd industrial-compliance-monitoring

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows (Command Prompt):
.venv\Scripts\activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the dashboard
python app.py
```

Open **http://localhost:5000** in your browser to view the interactive dashboard.

### What you'll see

- Interactive **Leaflet.js map** with color-coded violation markers
- **Zone switching** between Peenya and Whitefield industrial areas
- **6 metric cards**: Green Cover %, Net Change, Vegetation Loss, New Construction, Risk Level, Violations
- **Violations table** with geo-coordinates and area in hectares
- **ML performance panel** with feature importance visualization
- **Satellite imagery viewer** with 7 image types (RGB, NDVI, Change Map, Land Cover, etc.)
- **Compliance reports** accessible at `/report/peenya` and `/report/whitefield`

---

## Full Replication — Run the Pipeline from Scratch

If you want to reproduce the entire pipeline (re-download satellite data, re-run preprocessing, re-train ML models), follow these steps. This is **not required** to view the dashboard — it's only needed if you want to verify or modify the pipeline.

> **Note:** The full pipeline takes approximately 5 minutes and requires an internet connection to download satellite imagery from Microsoft Planetary Computer.

```bash
# Ensure virtual environment is activated and dependencies are installed (see Quick Start steps 1–4)

# Step 1: Download Sentinel-2 satellite data + KGIS boundaries
#         Downloads ~28 GeoTIFF files from ESA Copernicus via STAC API
#         Also downloads KGIS taluk shapefiles from kgis.ksrsac.in
python download_data.py

# Step 2: Run the preprocessing + analysis pipeline
#         Stage 1: Cloud masking, band resampling, KGIS boundary masking
#         Stage 2: NDVI, NDBI, NBI spectral index computation
#         Stage 3: Bitemporal change detection + violation extraction
#         Stage 4: Visualization generation (maps, charts)
python run_pipeline.py

# Step 3: Train ML models (Random Forest + K-Means)
#         70/30 stratified train-test split
#         Saves trained models to models/ and metrics JSON
python -m src.ml_classifier

# Step 4: Generate all result images + HTML compliance reports
#         Creates before/after panels, confusion matrices, feature importance,
#         learning curves, annotated violation maps, and land cover comparisons
python generate_results.py

# Step 5: Launch the dashboard
python app.py
```

### Pipeline Architecture

```
download_data.py            ← Step 1: Sentinel-2 L2A + KGIS boundaries
        │
run_pipeline.py             ← Step 2: Orchestrator (runs stages 1–4)
        │
  src/preprocess.py         ←   Band stacking, cloud masking, resampling
        │
  src/indices.py            ←   NDVI, NDBI, NBI spectral indices
        │
  src/change_detection.py   ←   Bitemporal differencing, risk scoring
        │
  src/visualize.py          ←   Map and chart generation
        │
src/ml_classifier.py        ← Step 3: Random Forest + K-Means classification
        │
generate_results.py         ← Step 4: All result images + compliance reports
        │
app.py                      ← Step 5: Flask dashboard server
```

---

## Project Structure

```
├── app.py                         # Flask web server (dashboard backend)
├── download_data.py               # Data acquisition script
├── run_pipeline.py                # Main pipeline orchestrator
├── generate_results.py            # Result visualization generator
├── requirements.txt               # Python dependencies
│
├── src/                           # Core pipeline modules
│   ├── __init__.py
│   ├── preprocess.py              # Band stacking, cloud masking, resampling
│   ├── indices.py                 # NDVI, NDBI, NBI computation
│   ├── change_detection.py        # Bitemporal change analysis
│   ├── ml_classifier.py           # Random Forest + K-Means + validation
│   └── visualize.py               # Visualization generation
│
├── frontend/                      # Dashboard frontend
│   ├── index.html                 # Single-page dashboard (HTML/JS/CSS)
│   └── assets/                    # Satellite imagery visualizations per zone
│       ├── peenya/                #   RGB, NDVI, change map, land cover PNGs
│       └── whitefield/
│
├── data/
│   ├── raw/                       # Raw Sentinel-2 band GeoTIFFs (7 bands × 2 zones × 2 periods)
│   ├── boundaries/                # Industrial zone GeoJSON boundaries
│   ├── kgis/                      # KGIS taluk + district shapefiles
│   └── processed/                 # Stacked bands, spectral indices, change maps, violations
│
├── models/                        # Trained RF models (.joblib) + metrics JSON
├── results/                       # All result images (13 PNGs)
└── reports/                       # HTML compliance reports + presentation materials
```

---

## Features

- **Data Acquisition**: Programmatic Sentinel-2 L2A retrieval via Microsoft Planetary Computer (STAC API)
- **KGIS Integration**: Karnataka GIS taluk boundaries for precise industrial zone delineation
- **Preprocessing**: Cloud masking (SCL band), 20m→10m bilinear resampling, boundary masking
- **Spectral Analysis**: NDVI (vegetation), NDBI (built-up), NBI (new construction)
- **ML Classification**: Random Forest (100 trees, 6 spectral bands) with non-circular validation
- **Unsupervised Baseline**: K-Means (k=5) clustering for comparison
- **Change Detection**: Bitemporal NDVI/NDBI differencing with adaptive thresholds
- **Violation Extraction**: Morphological processing + connected component analysis with geo-tagging
- **Compliance Reports**: Auto-generated HTML reports with risk assessment and violation alerts
- **Web Dashboard**: Interactive Leaflet.js map with zone switching, metric cards, and imagery viewer

## Dataset

- **Source**: Sentinel-2 L2A (ESA Copernicus) via Microsoft Planetary Computer
- **Resolution**: 10m (B02–B08), 20m resampled to 10m (B11–B12)
- **Zones**: Peenya Industrial Area, Whitefield (Bengaluru, Karnataka)
- **Time Periods**: March 2020 (baseline) vs March 2024 (recent)
- **Total Data Points**: ~11.6 million multispectral pixels
- **Bands**: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR1), B12 (SWIR2), SCL

## Technologies

- **Python 3.9+** — core language
- **Rasterio, GDAL** — geospatial raster I/O
- **GeoPandas, PyProj, Shapely** — vector/spatial operations
- **Scikit-learn** — Random Forest, K-Means
- **NumPy, SciPy** — numerical computing, morphological operations
- **Matplotlib, Seaborn** — visualization
- **PySTAC, Planetary Computer** — satellite data API
- **Flask, Flask-CORS** — web backend
- **Leaflet.js** — interactive map frontend

## Team

- Oishik Kar (23bcs089)
- Mihir Dagar (23bcs080)
- Shouvik Das (23bcs125)
- Yuvansh Chauhan (23bcs139)

Under the guidance of **Dr. Sunil C K**, Assistant Professor, Department of CSE, IIIT Dharwad

## License

Academic use — IIIT Dharwad ML Course Project
