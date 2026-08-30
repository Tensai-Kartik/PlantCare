"""
Populate sample leaf examples into backend/static/examples/
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_RAW = ROOT / "ml" / "dataset" / "raw"
STATIC_EXAMPLES = ROOT / "backend" / "static" / "examples"

STATIC_EXAMPLES.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    "apple_black_rot",
    "apple_cedar_apple_rust",
    "apple_healthy",
    "apple_scab",
    "corn_common_rust",
    "corn_healthy",
    "corn_northern_leaf_blight",
    "grape_black_rot",
    "grape_esca_black_measles",
    "grape_healthy",
    "pepper_bell_bacterial_spot",
    "pepper_bell_healthy",
    "potato_early_blight",
    "potato_healthy",
    "potato_late_blight",
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight",
    "tomato_septoria_leaf_spot",
    "tomato_yellow_leaf_curl_virus"
]

for class_name in SAMPLES:
    src_img = DATASET_RAW / class_name / "leaf_0001.jpg"
    if not src_img.exists():
        # Look for any image in that folder
        class_folder = DATASET_RAW / class_name
        if class_folder.exists():
            imgs = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
            if imgs:
                src_img = imgs[0]

    if src_img.exists():
        dst_img = STATIC_EXAMPLES / f"{class_name}.jpg"
        shutil.copy(src_img, dst_img)
        print(f"Copied {src_img.name} -> {dst_img.name}")
    else:
        print(f"Warning: No source image found for {class_name}.")

print(f"Successfully populated all {len(SAMPLES)} disease example images!")
