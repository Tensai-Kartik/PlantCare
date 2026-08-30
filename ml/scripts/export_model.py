"""
Model Export Script for PlantCare
Exports trained PyTorch weights and deployment metadata to backend/model_weights/.
"""

import os
import json
import shutil
from pathlib import Path
import torch

from train import build_model, NORM_MEAN, NORM_STD

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BACKEND_WEIGHTS_DIR = PROJECT_ROOT.parent / "backend" / "model_weights"

def export_all():
    BACKEND_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    models = ["efficientnet_b0", "mobilenet_v3_small"]
    registry_meta = {
        "default_model": "efficientnet_b0",
        "models": {}
    }

    for model_name in models:
        ckpt_path = MODELS_DIR / f"{model_name}_best.pth"
        if not ckpt_path.exists():
            print(f"Skipping {model_name}: checkpoint {ckpt_path} not found.")
            continue

        print(f"Exporting {model_name} for production backend...")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Copy weight file to backend
        target_weight_path = BACKEND_WEIGHTS_DIR / f"{model_name}.pth"
        torch.save({
            "model_state_dict": checkpoint["model_state_dict"],
            "model_name": model_name,
            "num_classes": checkpoint["num_classes"],
            "classes": checkpoint["classes"],
            "img_size": checkpoint.get("img_size", 224),
            "norm_mean": NORM_MEAN,
            "norm_std": NORM_STD
        }, target_weight_path)

        # Load evaluation metrics if available
        eval_path = OUTPUTS_DIR / f"{model_name}_evaluation.json"
        eval_data = {}
        if eval_path.exists():
            with open(eval_path, "r") as f:
                eval_data = json.load(f)

        registry_meta["models"][model_name] = {
            "id": model_name,
            "name": "EfficientNet-B0" if model_name == "efficientnet_b0" else "MobileNetV3-Small",
            "architecture": model_name,
            "file": f"{model_name}.pth",
            "img_size": checkpoint.get("img_size", 224),
            "num_classes": checkpoint["num_classes"],
            "classes": checkpoint["classes"],
            "accuracy": eval_data.get("test_accuracy", checkpoint.get("val_acc", 0.0)),
            "weighted_f1": eval_data.get("weighted_f1_score", 0.0),
            "latency_ms": eval_data.get("avg_inference_latency_ms", 0.0),
            "norm_mean": NORM_MEAN,
            "norm_std": NORM_STD
        }

    meta_file = BACKEND_WEIGHTS_DIR / "model_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(registry_meta, f, indent=2)

    print(f"Export completed! Metadata written to {meta_file}")

if __name__ == "__main__":
    export_all()
