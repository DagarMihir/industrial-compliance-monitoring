"""
ML Classification module.
Adds actual ML models (Random Forest + K-Means) for land cover classification,
replacing pure threshold-based approach. Includes validation metrics.
"""

import os
import json
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import joblib


# Land cover class labels
CLASS_NAMES = {
    0: "Other",
    1: "Dense Vegetation",
    2: "Sparse Vegetation",
    3: "Built-up",
    4: "Bare Soil",
    5: "Water",
}


def prepare_features(stacked_path):
    """
    Extract pixel-level feature vectors from stacked GeoTIFF.
    Features: B02, B03, B04, B08, B11, B12, NDVI, NDBI, NBI
    """
    with rasterio.open(stacked_path) as src:
        data = src.read()  # (7, H, W)
        profile = src.profile.copy()

    h, w = data.shape[1], data.shape[2]

    # Raw bands (scaled to 0-1)
    max_val = 10000.0  # Sentinel-2 L2A surface reflectance scale
    b02 = data[0] / max_val
    b03 = data[1] / max_val
    b04 = data[2] / max_val  # Red
    b08 = data[3] / max_val  # NIR
    b11 = data[4] / max_val  # SWIR1
    b12 = data[5] / max_val  # SWIR2
    cloud = data[6]

    # Compute indices
    eps = 1e-8
    ndvi = (b08 - b04) / (b08 + b04 + eps)
    ndbi = (b11 - b08) / (b11 + b08 + eps)
    nbi = (b04 * b11) / (b08 + eps)

    # Stack features: (9, H, W)
    features = np.stack([b02, b03, b04, b08, b11, b12, ndvi, ndbi, nbi], axis=0)

    # Flatten to (H*W, 9) — each pixel is a sample
    X = features.reshape(9, -1).T
    cloud_flat = cloud.flatten()

    # Mask out cloudy pixels
    valid_mask = cloud_flat > 0.5
    X_valid = X[valid_mask]

    return X, X_valid, valid_mask, h, w, profile


def generate_training_labels(X_valid):
    """
    Generate pseudo-labels using threshold-based rules on NDVI/NDBI.
    These serve as training labels for the Random Forest.

    Feature indices: [B02, B03, B04, B08, B11, B12, NDVI, NDBI, NBI]
                       0     1     2     3     4     5     6      7    8
    """
    ndvi = X_valid[:, 6]
    ndbi = X_valid[:, 7]

    labels = np.zeros(len(X_valid), dtype=np.int32)
    labels[ndvi < 0.0] = 5                               # Water
    labels[(ndvi >= 0.0) & (ndvi < 0.1) & (ndbi < 0.0)] = 4  # Bare Soil
    labels[(ndbi > 0.0) & (ndvi < 0.2)] = 3              # Built-up
    labels[(ndvi >= 0.2) & (ndvi <= 0.4)] = 2             # Sparse Veg
    labels[ndvi > 0.4] = 1                                # Dense Veg

    return labels


def train_random_forest(base_dir, zone, period="T1_2020"):
    """
    Train a Random Forest classifier on spectral features.
    
    IMPORTANT: We train on raw spectral bands ONLY (B02-B12),
    NOT on NDVI/NDBI. Labels are derived from NDVI/NDBI thresholds,
    so if we included those as features, the model would trivially
    learn the threshold rules (circular validation). By training on
    raw bands, the RF must learn spectral signatures from reflectance
    data — this is a proper ML generalization task.
    """
    processed = os.path.join(base_dir, "data", "processed")
    stacked_path = os.path.join(processed, zone, f"{period}_stacked.tif")

    print(f"  Preparing features from {zone}/{period}...")
    X_all, X_valid_all, valid_mask, h, w, profile = prepare_features(stacked_path)
    labels = generate_training_labels(X_valid_all)

    # USE ONLY RAW BANDS for training (indices 0-5: B02, B03, B04, B08, B11, B12)
    # Labels are from NDVI/NDBI (indices 6-7), so this avoids circular validation
    X_valid = X_valid_all[:, :6]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_valid, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  Features: B02, B03, B04, B08, B11, B12 (raw bands only)")
    print(f"  Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Train Random Forest
    print(f"  Training Random Forest (n_trees=100)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=[CLASS_NAMES.get(i, f"Class {i}") for i in sorted(np.unique(y_test))],
        zero_division=0,
    )

    print(f"\n  === VALIDATION RESULTS ===")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n  Classification Report:\n{report}")

    # Feature importance (only raw bands used as features)
    feature_names = ["B02", "B03", "B04", "B08", "B11", "B12"]
    importance = dict(zip(feature_names, rf.feature_importances_.tolist()))
    print(f"  Feature Importance: {json.dumps(importance, indent=2)}")

    # Save model
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"rf_{zone}.joblib")
    joblib.dump(rf, model_path)
    print(f"  Model saved: {model_path}")

    # Save metrics
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "feature_importance": importance,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
    }
    metrics_path = os.path.join(model_dir, f"rf_{zone}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return rf, metrics


def predict_landcover(base_dir, zone, period, rf_model):
    """Apply trained RF model to classify land cover for any time period."""
    processed = os.path.join(base_dir, "data", "processed")
    stacked_path = os.path.join(processed, zone, f"{period}_stacked.tif")

    X, X_valid, valid_mask, h, w, profile = prepare_features(stacked_path)

    # Use only raw bands (matching training features)
    X_valid_raw = X_valid[:, :6]

    # Predict on valid pixels
    pred_valid = rf_model.predict(X_valid_raw)

    # Reconstruct full image
    pred_full = np.zeros(h * w, dtype=np.uint8)
    pred_full[valid_mask] = pred_valid
    pred_map = pred_full.reshape(h, w)

    # Save
    out_dir = os.path.join(processed, zone, f"{period}_ml_classification")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "landcover_rf.tif")

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8")
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(pred_map, 1)

    # Class distribution
    unique, counts = np.unique(pred_map[pred_map > 0], return_counts=True)
    dist = {CLASS_NAMES.get(int(u), f"Class {u}"): int(c) for u, c in zip(unique, counts)}
    print(f"  {zone}/{period} class distribution: {dist}")

    return pred_map


def run_kmeans_clustering(base_dir, zone, period="T1_2020"):
    """Unsupervised K-Means clustering for comparison."""
    processed = os.path.join(base_dir, "data", "processed")
    stacked_path = os.path.join(processed, zone, f"{period}_stacked.tif")

    X, X_valid, valid_mask, h, w, profile = prepare_features(stacked_path)

    print(f"  Running K-Means (k=5) on {zone}/{period}...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(X_valid)

    # Reconstruct full image
    cluster_full = np.zeros(h * w, dtype=np.uint8)
    cluster_full[valid_mask] = cluster_labels + 1  # 0 = masked
    cluster_map = cluster_full.reshape(h, w)

    out_dir = os.path.join(processed, zone, f"{period}_ml_classification")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "clusters_kmeans.tif")

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8")
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(cluster_map, 1)

    print(f"  K-Means clustering saved: {out_path}")
    return cluster_map, kmeans


def extract_violation_coordinates(base_dir, zone):
    """
    Extract geo-coordinates for violation locations from the change mask.
    Uses morphological closing to merge scattered pixels into meaningful
    clusters, then extracts centroids with geo-coordinates.
    """
    processed = os.path.join(base_dir, "data", "processed")
    change_path = os.path.join(processed, zone, "change_detection", "change_mask.tif")

    if not os.path.exists(change_path):
        print(f"  [SKIP] No change mask for {zone}")
        return []

    with rasterio.open(change_path) as src:
        change = src.read(1)
        transform = src.transform
        crs = src.crs

    from pyproj import Transformer
    from scipy.ndimage import label as scipy_label, binary_closing, binary_dilation
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    violations = []

    # Structuring element: 5x5 square = merges pixels within 50m
    struct = np.ones((5, 5), dtype=bool)

    for change_type, change_name in [(1, "vegetation_loss"), (2, "new_construction"), (3, "both")]:
        mask = (change == change_type)
        if not mask.any():
            continue

        # Morphological closing to merge scattered pixels into clusters
        closed = binary_closing(mask, structure=struct, iterations=3)
        # Then dilate slightly to connect nearby clusters
        closed = binary_dilation(closed, structure=struct, iterations=1)

        labeled, n_regions = scipy_label(closed)

        for region_id in range(1, min(n_regions + 1, 101)):
            region_pixels = np.where(labeled == region_id)
            # Count actual change pixels within this merged region
            actual_count = int(mask[region_pixels].sum())
            if actual_count < 5:
                continue

            # Centroid in pixel coordinates
            cy = float(np.mean(region_pixels[0]))
            cx = float(np.mean(region_pixels[1]))

            # Convert pixel to map coordinates
            map_x, map_y = transform * (cx, cy)

            # Convert to lat/lon
            lon, lat = transformer.transform(map_x, map_y)

            area_sqm = actual_count * 100

            violations.append({
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "type": change_name,
                "area_pixels": actual_count,
                "area_sqm": int(area_sqm),
                "area_hectares": round(area_sqm / 10000, 3),
            })

    # Sort by area descending
    violations.sort(key=lambda v: v["area_sqm"], reverse=True)

    # Enrich violations with KGIS administrative metadata
    _enrich_with_kgis(violations)

    # Save
    out_path = os.path.join(processed, zone, "change_detection", "violations.json")
    with open(out_path, "w") as f:
        json.dump(violations, f, indent=2)

    print(f"  {zone}: Found {len(violations)} violation zones")
    for v in violations[:5]:
        kgis_info = f" [{v.get('kgis_ward', '?')}]" if 'kgis_ward' in v else ""
        print(f"    [{v['type']}] ({v['lat']}, {v['lon']}) - {v['area_hectares']} ha{kgis_info}")

    return violations


def _enrich_with_kgis(violations, max_queries=10):
    """
    Query KGIS Location Details API to get administrative hierarchy
    (ward, zone, town, district) for each violation coordinate.
    Only queries the top violations to avoid hammering the API.
    """
    import requests
    api_url = "https://kgis.ksrsac.in:9000/genericwebservices/ws/getlocationdetails"

    enriched = 0
    for v in violations[:max_queries]:
        try:
            r = requests.get(
                api_url,
                params={"coordinates": f"{v['lat']},{v['lon']}", "type": "dd"},
                verify=False, timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                if data and isinstance(data, list) and len(data) > 0:
                    info = data[0]
                    v["kgis_type"] = info.get("type", "")
                    v["kgis_district"] = info.get("districtName", "")
                    v["kgis_ward"] = info.get("wardName", "")
                    v["kgis_zone"] = info.get("zoneName", "")
                    v["kgis_town"] = info.get("townName", "")
                    enriched += 1
        except Exception:
            pass  # API may be slow/down, don't block pipeline

    if enriched > 0:
        print(f"  KGIS API: enriched {enriched}/{min(len(violations), max_queries)} violations with ward data")


def run_ml_pipeline(base_dir):
    """Run the complete ML classification + validation pipeline."""
    all_metrics = {}

    for zone in ["peenya", "whitefield"]:
        print(f"\n{'='*60}")
        print(f"  ML PIPELINE: {zone.upper()}")
        print(f"{'='*60}")

        # Train RF on T1 (baseline) data
        print(f"\n  --- Training Random Forest ---")
        rf, metrics = train_random_forest(base_dir, zone, "T1_2020")
        all_metrics[zone] = metrics

        # Apply RF to both time periods
        print(f"\n  --- Applying RF to T1 ---")
        predict_landcover(base_dir, zone, "T1_2020", rf)
        print(f"\n  --- Applying RF to T2 ---")
        predict_landcover(base_dir, zone, "T2_2024", rf)

        # K-Means for comparison
        print(f"\n  --- K-Means Clustering ---")
        run_kmeans_clustering(base_dir, zone, "T1_2020")
        run_kmeans_clustering(base_dir, zone, "T2_2024")

        # Extract violation coordinates
        print(f"\n  --- Extracting Violation Coordinates ---")
        extract_violation_coordinates(base_dir, zone)

    return all_metrics


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 60)
    print("  ML CLASSIFICATION & VALIDATION PIPELINE")
    print("=" * 60)
    metrics = run_ml_pipeline(base_dir)

    print("\n\n" + "=" * 60)
    print("  FINAL VALIDATION SUMMARY")
    print("=" * 60)
    for zone, m in metrics.items():
        print(f"  {zone.upper()}: Acc={m['accuracy']}, F1={m['f1_score']}, "
              f"Prec={m['precision']}, Rec={m['recall']}")
