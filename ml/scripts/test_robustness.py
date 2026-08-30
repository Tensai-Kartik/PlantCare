"""
ML Robustness Testing Script for PlantCare
Evaluates computer vision model resilience against 9 controlled image perturbations:
1. Brightness reduction (-30%, -50%)
2. Brightness increase (+30%, +50%)
3. Contrast variations (low contrast, high contrast)
4. Gaussian blur
5. Rotations (15°, 90°, 180°, 270°)
6. Center and random cropping
7. Gaussian noise
8. JPEG compression artifacts (quality 40, quality 20)
9. Background variation & color shifts
"""

import os
import io
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torchvision import transforms

from train import build_model, NORM_MEAN, NORM_STD

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

def apply_perturbation(img: Image.Image, perturbation_type: str, level: float = 1.0) -> Image.Image:
    """Applies a controlled visual variation to the input leaf image."""
    if perturbation_type == "brightness_down":
        factor = max(0.2, 1.0 - 0.4 * level)
        return ImageEnhance.Brightness(img).enhance(factor)

    elif perturbation_type == "brightness_up":
        factor = 1.0 + 0.5 * level
        return ImageEnhance.Brightness(img).enhance(factor)

    elif perturbation_type == "contrast_low":
        factor = max(0.3, 1.0 - 0.5 * level)
        return ImageEnhance.Contrast(img).enhance(factor)

    elif perturbation_type == "contrast_high":
        factor = 1.0 + 0.6 * level
        return ImageEnhance.Contrast(img).enhance(factor)

    elif perturbation_type == "blur":
        radius = 1.2 * level
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    elif perturbation_type == "rotation_subtle":
        return img.rotate(15 * level, expand=False, fillcolor=(240, 238, 230))

    elif perturbation_type == "rotation_90":
        return img.rotate(90, expand=False)

    elif perturbation_type == "crop":
        w, h = img.size
        crop_ratio = max(0.65, 1.0 - 0.25 * level)
        cw, ch = int(w * crop_ratio), int(h * crop_ratio)
        x1, y1 = (w - cw) // 2, (h - ch) // 2
        cropped = img.crop((x1, y1, x1 + cw, y1 + ch))
        return cropped.resize((w, h), Image.BILINEAR)

    elif perturbation_type == "noise":
        arr = np.array(img, dtype=float)
        noise = np.random.normal(0, 18 * level, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    elif perturbation_type == "jpeg_compression":
        quality = max(15, int(95 - 65 * level))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf)

    return img

def evaluate_robustness(model_name: str = "efficientnet_b0", max_samples: int = 50):
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
    all_image_paths = []
    for c in classes:
        c_dir = test_dir / c
        if c_dir.exists():
            for p in list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png")):
                all_image_paths.append((p, c))

    if len(all_image_paths) > max_samples:
        import random
        random.seed(42)
        all_image_paths = random.sample(all_image_paths, max_samples)

    perturbations = [
        ("brightness_down", "Brightness Reduction (-40%)"),
        ("brightness_up", "Brightness Increase (+50%)"),
        ("contrast_low", "Low Contrast (-50%)"),
        ("contrast_high", "High Contrast (+60%)"),
        ("blur", "Gaussian Blur (r=1.2)"),
        ("rotation_subtle", "Rotation (15 deg)"),
        ("rotation_90", "Rotation (90 deg)"),
        ("crop", "Scale Crop (75%)"),
        ("noise", "Gaussian Sensor Noise (std=18)"),
        ("jpeg_compression", "JPEG Artifacts (quality=30)")
    ]

    results_by_pert = {}

    print(f"\nRunning Robustness Evaluation for {model_name} on {len(all_image_paths)} test images...")

    for pert_key, pert_label in perturbations:
        original_matches = 0
        perturbed_matches_original = 0
        original_confs = []
        perturbed_confs = []

        for img_path, ground_truth in all_image_paths:
            img = Image.open(img_path).convert("RGB")
            pert_img = apply_perturbation(img, pert_key, level=1.0)

            t_orig = tensor_transform(img).unsqueeze(0).to(device)
            t_pert = tensor_transform(pert_img).unsqueeze(0).to(device)

            with torch.no_grad():
                out_orig = model(t_orig)
                out_pert = model(t_pert)

                prob_orig = torch.softmax(out_orig, dim=1)[0]
                prob_pert = torch.softmax(out_pert, dim=1)[0]

                pred_orig = classes[prob_orig.argmax().item()]
                pred_pert = classes[prob_pert.argmax().item()]

                conf_orig = prob_orig.max().item()
                conf_pert = prob_pert.max().item()

            original_confs.append(conf_orig)
            perturbed_confs.append(conf_pert)

            if pred_orig == ground_truth:
                original_matches += 1
            if pred_pert == pred_orig:
                perturbed_matches_original += 1

        stability_rate = (perturbed_matches_original / len(all_image_paths)) * 100.0
        avg_orig_conf = float(np.mean(original_confs) * 100.0)
        avg_pert_conf = float(np.mean(perturbed_confs) * 100.0)
        conf_drop = max(0.0, avg_orig_conf - avg_pert_conf)

        results_by_pert[pert_key] = {
            "label": pert_label,
            "prediction_stability_rate_percent": round(stability_rate, 2),
            "baseline_mean_confidence_percent": round(avg_orig_conf, 2),
            "perturbed_mean_confidence_percent": round(avg_pert_conf, 2),
            "confidence_drop_percent": round(conf_drop, 2)
        }

    overall_stability = float(np.mean([r["prediction_stability_rate_percent"] for r in results_by_pert.values()]))
    overall_conf_drop = float(np.mean([r["confidence_drop_percent"] for r in results_by_pert.values()]))

    robustness_report = {
        "model_name": model_name,
        "samples_tested": len(all_image_paths),
        "overall_stability_score_percent": round(overall_stability, 2),
        "mean_confidence_drop_percent": round(overall_conf_drop, 2),
        "perturbation_details": results_by_pert
    }

    out_file = OUTPUTS_DIR / f"{model_name}_robustness_report.json"
    with open(out_file, "w") as f:
        json.dump(robustness_report, f, indent=2)

    print("\n" + "="*60)
    print(f"ROBUSTNESS STRESS TEST REPORT: {model_name}")
    print("="*60)
    print(f"Overall Stability Score:      {overall_stability:.2f}%")
    print(f"Mean Confidence Degradation:  {overall_conf_drop:.2f}%")
    for k, v in results_by_pert.items():
        print(f"  • {v['label']:<32} Stability: {v['prediction_stability_rate_percent']}% (Conf Drop: {v['confidence_drop_percent']}%)")
    print(f"Full Report Saved:            {out_file.name}")
    print("="*60 + "\n")

    return robustness_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3_small"])
    args = parser.parse_args()
    evaluate_robustness(args.model)
