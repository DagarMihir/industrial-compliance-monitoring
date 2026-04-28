"""
Build a fully static version of the dashboard for deployment.
Copies all images, compliance reports, and bakes API data into a static JSON file.
"""
import os, json, shutil, re

BASE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.join(BASE, "deploy")

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def build():
    # Clean and create deploy dir
    if os.path.exists(DEPLOY):
        shutil.rmtree(DEPLOY)
    os.makedirs(DEPLOY)

    # --- 1. Build the API data as static JSON ---
    zones = []
    for zone in ["peenya", "whitefield"]:
        cd = os.path.join(BASE, "data", "processed", zone, "change_detection")
        stats = load_json(os.path.join(cd, "change_stats.json"))
        viol_path = os.path.join(cd, "violations.json")
        violations = load_json(viol_path) if os.path.exists(viol_path) else []
        metrics = load_json(os.path.join(BASE, "models", f"rf_{zone}_metrics.json"))
        t1_idx = load_json(os.path.join(BASE, "data", "processed", zone, "T1_2020_indices", "index_stats.json"))
        t2_idx = load_json(os.path.join(BASE, "data", "processed", zone, "T2_2024_indices", "index_stats.json"))

        bbox = {"peenya": [13.01, 77.48, 13.07, 77.56], "whitefield": [12.95, 77.72, 13.01, 77.80]}
        center = {"peenya": [13.04, 77.52], "whitefield": [12.98, 77.76]}

        zones.append({
            "name": zone,
            "label": zone.title() + " Industrial Area",
            "center": center[zone],
            "bbox": bbox[zone],
            "change_stats": stats,
            "ml_metrics": {
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1_score", 0),
                "n_train": metrics.get("n_train", 0),
                "n_test": metrics.get("n_test", 0),
                "feature_importance": metrics.get("feature_importance", {}),
                "confusion_matrix": metrics.get("confusion_matrix", []),
            },
            "index_stats": {"T1_2020": t1_idx, "T2_2024": t2_idx},
            "violations": violations if isinstance(violations, list) else [],
        })

    with open(os.path.join(DEPLOY, "data.json"), "w", encoding="utf-8") as f:
        json.dump(zones, f)
    print(f"  [OK] data.json ({os.path.getsize(os.path.join(DEPLOY, 'data.json'))} bytes)")

    # --- 2. Copy images ---
    img_dir = os.path.join(DEPLOY, "img")

    # results/ images
    results_src = os.path.join(BASE, "results")
    results_dst = os.path.join(img_dir, "results")
    os.makedirs(results_dst, exist_ok=True)
    for f in os.listdir(results_src):
        if f.endswith(".png"):
            shutil.copy2(os.path.join(results_src, f), os.path.join(results_dst, f))
            print(f"  [IMG] results/{f}")

    # frontend/assets/ images
    assets_src = os.path.join(BASE, "frontend", "assets")
    for zone in ["peenya", "whitefield"]:
        zone_src = os.path.join(assets_src, zone)
        zone_dst = os.path.join(img_dir, "frontend", "assets", zone)
        os.makedirs(zone_dst, exist_ok=True)
        if os.path.isdir(zone_src):
            for f in os.listdir(zone_src):
                if f.endswith(".png"):
                    shutil.copy2(os.path.join(zone_src, f), os.path.join(zone_dst, f))
                    print(f"  [IMG] frontend/assets/{zone}/{f}")

    # --- 3. Copy compliance reports ---
    report_dir = os.path.join(DEPLOY, "report")
    os.makedirs(report_dir, exist_ok=True)
    reports_src = os.path.join(BASE, "reports")
    for zone in ["peenya", "whitefield"]:
        src = os.path.join(reports_src, f"compliance_report_{zone}.html")
        if os.path.exists(src):
            # Copy as report/<zone>.html (matching the /report/<zone> route)
            dst = os.path.join(report_dir, f"{zone}.html")
            shutil.copy2(src, dst)
            print(f"  [RPT] report/{zone}.html")

    # --- 4. Create static index.html ---
    with open(os.path.join(BASE, "frontend", "index.html"), "r", encoding="utf-8") as f:
        html = f.read()

    # Replace fetch("/api/zones") with fetch("data.json")
    html = html.replace('fetch("/api/zones")', 'fetch("data.json")')

    # Make all image paths relative (remove leading /)
    html = html.replace("src: `/img/", "src: `img/")
    html = html.replace("src = `/img/", "src = `img/")

    # Fix report paths: /report/${name} -> report/${name}.html
    html = html.replace("report: `/report/${name}`", "report: `report/${name}.html`")

    with open(os.path.join(DEPLOY, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [OK] index.html")

    # --- 5. Summary ---
    total_files = sum(len(files) for _, _, files in os.walk(DEPLOY))
    total_size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(DEPLOY) for f in fn)
    print(f"\n  Deploy folder ready: {DEPLOY}")
    print(f"  Total files: {total_files}")
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    build()
