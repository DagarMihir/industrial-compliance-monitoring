"""
Preprocessing module: Cloud masking, KGIS boundary masking, band stacking, resampling.
Reads raw Sentinel-2 bands, clips to KGIS taluk boundaries, and produces
analysis-ready stacked GeoTIFFs.
"""

import os
import json
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from rasterio.features import rasterize


def load_band(band_path):
    """Load a single band GeoTIFF and return data + metadata."""
    with rasterio.open(band_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    return data, profile, transform, crs, bounds


def create_cloud_mask(scl_path):
    """
    Create a cloud/shadow mask from Scene Classification Layer (SCL).
    
    SCL values:
      0 = No data
      1 = Saturated/defective
      2 = Dark area pixels
      3 = Cloud shadows
      4 = Vegetation
      5 = Not vegetated
      6 = Water
      7 = Unclassified
      8 = Cloud medium probability
      9 = Cloud high probability
      10 = Thin cirrus
      11 = Snow/ice
    
    We mask out: 0, 1, 3, 8, 9, 10 (no data, defective, shadows, clouds)
    """
    with rasterio.open(scl_path) as src:
        scl = src.read(1)
        scl_profile = src.profile.copy()
    
    # Valid pixels: vegetation(4), not-veg(5), water(6), unclassified(7)
    # Also allow dark area (2) and snow (11) as valid
    valid_classes = {2, 4, 5, 6, 7, 11}
    mask = np.isin(scl, list(valid_classes)).astype(np.uint8)
    
    return mask, scl, scl_profile


def resample_to_10m(data_20m, profile_20m, target_shape, target_transform):
    """
    Resample 20m bands (B11, B12) to 10m resolution to match B02-B08.
    Uses bilinear interpolation.
    """
    # Create output array
    resampled = np.zeros(target_shape, dtype=np.float32)
    
    # Simple nearest-neighbor upsampling (2x in each dimension)
    # This is sufficient for our index computation
    from scipy.ndimage import zoom
    zoom_factor_y = target_shape[0] / data_20m.shape[0]
    zoom_factor_x = target_shape[1] / data_20m.shape[1]
    resampled = zoom(data_20m, (zoom_factor_y, zoom_factor_x), order=1)
    
    return resampled.astype(np.float32)


def create_kgis_mask(kgis_geojson_path, ref_transform, ref_shape, ref_crs):
    """
    Create a binary mask from KGIS taluk boundary GeoJSON.
    Pixels inside the taluk polygon = 1, outside = 0.
    """
    if not os.path.exists(kgis_geojson_path):
        print(f"    [WARN] KGIS boundary not found: {kgis_geojson_path}")
        return np.ones(ref_shape, dtype=np.uint8)

    try:
        with open(kgis_geojson_path, 'r') as f:
            geojson = json.load(f)

        # Extract geometry from GeoJSON features
        geometries = []
        for feature in geojson.get('features', []):
            geom = feature.get('geometry')
            if geom:
                geometries.append(geom)

        if not geometries:
            print("    [WARN] No geometries in KGIS boundary, using full extent")
            return np.ones(ref_shape, dtype=np.uint8)

        # Rasterize the KGIS polygon onto the satellite raster grid
        mask = rasterize(
            [(geom, 1) for geom in geometries],
            out_shape=ref_shape,
            transform=ref_transform,
            fill=0,
            dtype=np.uint8,
        )

        inside_pct = (mask.sum() / mask.size) * 100
        print(f"    KGIS taluk mask: {inside_pct:.1f}% of raster inside boundary")
        return mask

    except Exception as e:
        print(f"    [WARN] KGIS mask error: {e}, using full extent")
        return np.ones(ref_shape, dtype=np.uint8)


def stack_bands(raw_dir, output_path, kgis_geojson_path=None):
    """
    Stack all bands into a single multi-band GeoTIFF.
    Resamples 20m bands to 10m.
    Applies cloud mask.
    
    Output band order: B02, B03, B04, B08, B11, B12, CloudMask
    """
    bands_10m = ["B02", "B03", "B04", "B08"]
    bands_20m = ["B11", "B12"]
    
    # Load 10m bands to get reference shape/transform
    ref_data, ref_profile, ref_transform, ref_crs, ref_bounds = load_band(
        os.path.join(raw_dir, "B02.tif")
    )
    target_shape = ref_data.shape
    print(f"    Reference shape (10m): {target_shape}")
    
    # Load and stack 10m bands
    stacked = []
    for band_name in bands_10m:
        band_path = os.path.join(raw_dir, f"{band_name}.tif")
        data, _, _, _, _ = load_band(band_path)
        stacked.append(data)
        print(f"    Loaded {band_name}: shape={data.shape}, "
              f"min={data.min():.0f}, max={data.max():.0f}")
    
    # Load and resample 20m bands
    for band_name in bands_20m:
        band_path = os.path.join(raw_dir, f"{band_name}.tif")
        data, profile, _, _, _ = load_band(band_path)
        resampled = resample_to_10m(data, profile, target_shape, ref_transform)
        stacked.append(resampled)
        print(f"    Loaded {band_name}: {data.shape} -> resampled to {resampled.shape}")
    
    # Load cloud mask
    scl_path = os.path.join(raw_dir, "SCL.tif")
    if os.path.exists(scl_path):
        cloud_mask, _, _ = create_cloud_mask(scl_path)
        # Resample SCL mask to 10m (it's 20m native)
        cloud_mask = resample_to_10m(
            cloud_mask.astype(np.float32), None, target_shape, ref_transform
        )
        cloud_mask = (cloud_mask > 0.5).astype(np.float32)
        valid_pct = (cloud_mask.sum() / cloud_mask.size) * 100
        print(f"    Cloud mask: {valid_pct:.1f}% valid pixels")
    else:
        cloud_mask = np.ones(target_shape, dtype=np.float32)

    # Apply KGIS taluk boundary mask
    if kgis_geojson_path:
        kgis_mask = create_kgis_mask(
            kgis_geojson_path, ref_transform, target_shape, ref_crs
        )
        # Combine: pixel must be cloud-free AND inside KGIS boundary
        cloud_mask = cloud_mask * kgis_mask.astype(np.float32)
        combined_pct = (cloud_mask.sum() / cloud_mask.size) * 100
        print(f"    Combined mask (cloud + KGIS): {combined_pct:.1f}% valid pixels")

    stacked.append(cloud_mask)
    
    # Stack into array: (n_bands, height, width)
    stacked_array = np.array(stacked, dtype=np.float32)
    
    # Write output
    out_profile = ref_profile.copy()
    out_profile.update(
        count=stacked_array.shape[0],
        dtype="float32",
        driver="GTiff",
        compress="deflate",
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, "w", **out_profile) as dst:
        for i in range(stacked_array.shape[0]):
            dst.write(stacked_array[i], i + 1)
        # Write band descriptions
        band_names = bands_10m + bands_20m + ["CloudMask"]
        dst.descriptions = tuple(band_names)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    Stacked output: {output_path} ({file_size:.1f} MB)")
    print(f"    Shape: {stacked_array.shape}")
    
    return stacked_array, out_profile


def preprocess_all(base_dir):
    """Run preprocessing for all zones and time periods."""
    raw_base = os.path.join(base_dir, "data", "raw")
    processed_base = os.path.join(base_dir, "data", "processed")
    kgis_boundary_dir = os.path.join(base_dir, "data", "kgis", "boundaries")

    # KGIS taluk boundary files for each zone
    kgis_paths = {
        "peenya": os.path.join(kgis_boundary_dir, "bangalore_north_taluk.geojson"),
        "whitefield": os.path.join(kgis_boundary_dir, "bangalore_east_taluk.geojson"),
    }

    zones = ["peenya", "whitefield"]
    periods = ["T1_2020", "T2_2024"]
    
    results = {}
    
    for zone in zones:
        kgis_path = kgis_paths.get(zone)
        if kgis_path and os.path.exists(kgis_path):
            print(f"\n  KGIS boundary: {os.path.basename(kgis_path)}")
        else:
            print(f"\n  [WARN] No KGIS boundary for {zone}")
            kgis_path = None

        for period in periods:
            raw_dir = os.path.join(raw_base, zone, period)
            if not os.path.exists(raw_dir):
                print(f"  [SKIP] {zone}/{period} - no raw data")
                continue
            
            print(f"\n  Processing {zone}/{period}...")
            output_path = os.path.join(processed_base, zone, f"{period}_stacked.tif")
            
            try:
                stacked, profile = stack_bands(raw_dir, output_path, kgis_path)
                results[f"{zone}/{period}"] = {
                    "path": output_path,
                    "shape": stacked.shape,
                    "profile": profile,
                }
                print(f"  [OK] {zone}/{period} preprocessed")
            except Exception as e:
                print(f"  [FAIL] {zone}/{period}: {e}")
                import traceback
                traceback.print_exc()
    
    return results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)
    results = preprocess_all(base_dir)
    print(f"\n  Processed {len(results)} datasets")
