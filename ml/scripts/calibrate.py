"""
Confidence Calibration & Expected Calibration Error (ECE) Evaluation for PlantCare
Evaluates raw Softmax vs. Temperature Scaled Softmax, computes ECE,
and generates reliability diagram data across confidence bins.
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.optimize import minimize

from train import build_model, NORM_MEAN, NORM_STD

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

def compute_ece(confidences: np.ndarray, predictions: np.ndarray, targets: np.ndarray, n_bins: int = 10):
    """
    Computes Expected Calibration Error (ECE) and reliability bin distributions.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_stats = []
    accuracies = (predictions == targets)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_stats.append({
                "bin": f"{bin_lower:.2f} - {bin_upper:.2f}",
                "sample_count": int(np.sum(in_bin)),
                "accuracy": round(float(accuracy_in_bin), 4),
                "confidence": round(float(avg_confidence_in_bin), 4),
                "calibration_gap": round(float(abs(avg_confidence_in_bin - accuracy_in_bin)), 4)
            })
        else:
            bin_stats.append({
                "bin": f"{bin_lower:.2f} - {bin_upper:.2f}",
                "sample_count": 0,
                "accuracy": 0.0,
                "confidence": round(float((bin_lower + bin_upper) / 2.0), 4),
                "calibration_gap": 0.0
            })

    return round(float(ece), 4), bin_stats

def optimize_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Optimizes temperature T using Negative Log-Likelihood (NLL) on validation set.
    """
    logits_np = logits.numpy()
    labels_np = labels.numpy()

    def nll_loss(t):
        temp = t[0]
        scaled = logits_np / temp
        # Subtract max for numerical stability
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
        # NLL loss
        p_target = probs[np.arange(len(labels_np)), labels_np]
        loss = -np.mean(np.log(np.clip(p_target, 1e-12, 1.0)))
        return loss

    res = minimize(nll_loss, x0=[1.1], bounds=[(0.1, 5.0)], method='L-BFGS-B')
    best_temp = float(res.x[0])
    return round(best_temp, 3)

def run_calibration_evaluation(model_name: str = "efficientnet_b0"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = MODELS_DIR / f"{model_name}_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    num_classes = len(classes)
    img_size = checkpoint.get("img_size", 224)

    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    # 1. Validation Set (for temperature optimization)
    val_dataset = datasets.ImageFolder(str(PROCESSED_DIR / "val"), transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    val_logits = []
    val_labels = []

    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            val_logits.append(out.cpu())
            val_labels.append(lbls)

    val_logits_tensor = torch.cat(val_logits, dim=0)
    val_labels_tensor = torch.cat(val_labels, dim=0)

    optimal_temperature = optimize_temperature(val_logits_tensor, val_labels_tensor)

    # 2. Test Set (for unbiased calibration evaluation)
    test_dataset = datasets.ImageFolder(str(PROCESSED_DIR / "test"), transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    test_logits = []
    test_labels = []

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            test_logits.append(out.cpu())
            test_labels.append(lbls)

    test_logits_tensor = torch.cat(test_logits, dim=0)
    test_labels_np = torch.cat(test_labels, dim=0).numpy()

    # Raw Softmax probabilities
    raw_probs = torch.softmax(test_logits_tensor, dim=1).numpy()
    raw_preds = np.argmax(raw_probs, axis=1)
    raw_confs = np.max(raw_probs, axis=1)
    raw_ece, raw_bins = compute_ece(raw_confs, raw_preds, test_labels_np)

    # Calibrated Softmax probabilities (with optimal T)
    calibrated_probs = torch.softmax(test_logits_tensor / optimal_temperature, dim=1).numpy()
    cal_preds = np.argmax(calibrated_probs, axis=1)
    cal_confs = np.max(calibrated_probs, axis=1)
    cal_ece, cal_bins = compute_ece(cal_confs, cal_preds, test_labels_np)

    accuracy = float(np.mean(raw_preds == test_labels_np) * 100.0)

    report = {
        "model_name": model_name,
        "test_samples": len(test_dataset),
        "accuracy_percent": round(accuracy, 2),
        "optimal_temperature_T": optimal_temperature,
        "raw_ece": raw_ece,
        "raw_ece_percent": round(raw_ece * 100.0, 2),
        "calibrated_ece": cal_ece,
        "calibrated_ece_percent": round(cal_ece * 100.0, 2),
        "ece_reduction_percent": round(((raw_ece - cal_ece) / (raw_ece + 1e-8)) * 100.0, 2),
        "raw_reliability_diagram_bins": raw_bins,
        "calibrated_reliability_diagram_bins": cal_bins
    }

    out_file = OUTPUTS_DIR / f"{model_name}_calibration_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*55)
    print(f"CONFIDENCE CALIBRATION REPORT: {model_name}")
    print("="*55)
    print(f"Accuracy:                    {accuracy:.2f}%")
    print(f"Optimal Temperature (T):     {optimal_temperature}")
    print(f"Raw Softmax ECE:             {raw_ece * 100:.2f}%")
    print(f"Calibrated Softmax ECE:      {cal_ece * 100:.2f}%")
    print(f"ECE Error Reduction:         {report['ece_reduction_percent']}%")
    print(f"Calibration Report Saved:    {out_file.name}")
    print("="*55 + "\n")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3_small"])
    args = parser.parse_args()
    run_calibration_evaluation(args.model)
