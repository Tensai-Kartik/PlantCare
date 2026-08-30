"""
Real-World Validation Script for PlantCare
Evaluates computer vision models against realistic non-ideal field photography:
- Non-standard lighting (harsh sunlight, dim shade, color cast)
- Complex natural backgrounds (soil, wooden benches, sky, mulch)
- Multi-leaf framing & partial leaves
- Image artifacts & compression

Reports standard dataset performance vs. real-world validation performance side-by-side.
"""

import os
import json
import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support

from train import build_model, NORM_MEAN, NORM_STD

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

def synthesize_real_world_variation(img: Image.Image, scenario: str) -> Image.Image:
    """Simulates real-world outdoor field conditions."""
    w, h = img.size
    np_img = np.array(img, dtype=float)

    if scenario == "harsh_sunlight":
        # Glare on top-left + high contrast
        glare = np.zeros((h, w, 3), dtype=float)
        for y in range(h):
            for x in range(w):
                dist = np.sqrt(x**2 + y**2)
                glare[y, x] = max(0, 180 - dist * 0.5)
        np_img = np.clip(np_img * 1.2 + glare * 0.4, 0, 255)
        res = Image.fromarray(np_img.astype(np.uint8))
        return ImageEnhance.Contrast(res).enhance(1.25)

    elif scenario == "dim_shade":
        # Low light with slight blue cast
        np_img[:, :, 0] *= 0.60
        np_img[:, :, 1] *= 0.65
        np_img[:, :, 2] *= 0.75
        res = Image.fromarray(np.clip(np_img, 0, 255).astype(np.uint8))
        return ImageEnhance.Brightness(res).enhance(0.7)

    elif scenario == "soil_background_clutter":
        # Brown soil background texture
        soil_bg = np.random.randint(60, 100, (h, w, 3), dtype=np.uint8)
        soil_bg[:, :, 0] = np.clip(soil_bg[:, :, 0] * 1.4, 0, 255) # more reddish brown
        soil_bg[:, :, 2] = np.clip(soil_bg[:, :, 2] * 0.6, 0, 255)
        # Blend outside leaf circle
        cx, cy = w // 2, h // 2
        r = int(min(w, h) * 0.40)
        mask = np.zeros((h, w), dtype=float)
        for y in range(h):
            for x in range(w):
                if (x - cx)**2 + (y - cy)**2 < r**2:
                    mask[y, x] = 1.0
        mask = np.expand_dims(mask, axis=2)
        blended = np_img * mask + soil_bg * (1.0 - mask)
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    elif scenario == "partial_leaf":
        # Shift and crop
        crop_box = (int(w * 0.15), int(h * 0.15), w, h)
        cropped = img.crop(crop_box)
        return cropped.resize((w, h), Image.BILINEAR)

    elif scenario == "camera_shake_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=1.8))

    return img

def evaluate_realworld(model_name: str = "efficientnet_b0"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = MODELS_DIR / f"{model_name}_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint {checkpoint_path} not found.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    num_classes = len(classes)
    img_size = checkpoint.get("img_size", 224)

    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tensor_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    test_dir = PROCESSED_DIR / "test"
    all_samples = []
    for c in classes:
        c_dir = test_dir / c
        if c_dir.exists():
            for p in list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png")):
                all_samples.append((p, c))

    scenarios = [
        ("clean_standard", "Standard Clean Test Set"),
        ("harsh_sunlight", "Harsh Field Sunlight & Specular Glare"),
        ("dim_shade", "Low-Light Shaded Canopy"),
        ("soil_background_clutter", "Cluttered Soil / Mulch Background"),
        ("partial_leaf", "Partial / Offset Leaf Framing"),
        ("camera_shake_blur", "Field Handheld Camera Motion Blur")
    ]

    scenario_metrics = {}

    for scen_key, scen_label in scenarios:
        preds = []
        confs = []
        targets = []

        for p, ground_truth in all_samples:
            img = Image.open(p).convert("RGB")
            if scen_key != "clean_standard":
                img = synthesize_real_world_variation(img, scen_key)

            tensor = tensor_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1)[0]
                top_idx = probs.argmax().item()
                top_conf = probs.max().item()

            preds.append(classes[top_idx])
            confs.append(top_conf)
            targets.append(ground_truth)

        preds_np = np.array(preds)
        targets_np = np.array(targets)

        acc = float(np.mean(preds_np == targets_np) * 100.0)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(targets_np, preds_np, average="weighted", zero_division=0)
        avg_conf = float(np.mean(confs) * 100.0)

        scenario_metrics[scen_key] = {
            "scenario_name": scen_label,
            "accuracy_percent": round(acc, 2),
            "weighted_f1": round(float(f1_w), 4),
            "mean_confidence_percent": round(avg_conf, 2),
            "total_samples": len(all_samples)
        }

    clean_acc = scenario_metrics["clean_standard"]["accuracy_percent"]
    rw_scenarios = [v["accuracy_percent"] for k, v in scenario_metrics.items() if k != "clean_standard"]
    avg_rw_acc = float(np.mean(rw_scenarios))

    report = {
        "model_name": model_name,
        "standard_dataset_accuracy_percent": clean_acc,
        "real_world_validation_accuracy_percent": round(avg_rw_acc, 2),
        "performance_gap_percent": round(clean_acc - avg_rw_acc, 2),
        "scenario_breakdown": scenario_metrics
    }

    out_file = OUTPUTS_DIR / f"{model_name}_realworld_evaluation.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*65)
    print(f"REAL-WORLD FIELD VALIDATION REPORT: {model_name}")
    print("="*65)
    print(f"Standard Benchmark Accuracy:   {clean_acc:.2f}%")
    print(f"Real-World Mean Accuracy:      {avg_rw_acc:.2f}%")
    print(f"Performance Gap:               {report['performance_gap_percent']:.2f}%\n")
    print("Field Scenario Breakdown:")
    for k, v in scenario_metrics.items():
        print(f"  • {v['scenario_name']:<40} Acc: {v['accuracy_percent']}% | F1: {v['weighted_f1']} | Conf: {v['mean_confidence_percent']}%")
    print(f"\nFull Report Saved:             {out_file.name}")
    print("="*65 + "\n")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3_small"])
    args = parser.parse_args()
    evaluate_realworld(args.model)
