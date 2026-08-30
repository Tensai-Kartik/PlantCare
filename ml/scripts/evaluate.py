"""
Model Evaluation Script for PlantCare
Computes real classification metrics (Accuracy, Precision, Recall, F1, Per-class, Confusion Matrix)
and measures CPU inference latency.
"""

import os
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from train import build_model, NORM_MEAN, NORM_STD

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

def evaluate_model(model_name: str, batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = MODELS_DIR / f"{model_name}_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")

    print(f"Loading checkpoint from {checkpoint_path}...")
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

    test_dataset = datasets.ImageFolder(str(PROCESSED_DIR / "test"), transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    all_targets = []
    latencies = []

    print(f"Evaluating {model_name} on test set ({len(test_dataset)} images)...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            t0 = time.perf_counter()
            outputs = model(images)
            t1 = time.perf_counter()

            batch_latency = (t1 - t0) / images.size(0) * 1000.0  # ms per image
            latencies.extend([batch_latency] * images.size(0))

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate overall metrics
    acc = np.mean(all_preds == all_targets) * 100.0
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    avg_latency = float(np.mean(latencies))

    # Per-class metrics
    class_p, class_r, class_f1, class_supp = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)
    per_class_results = {}
    for i, c_name in enumerate(classes):
        per_class_results[c_name] = {
            "precision": round(float(class_p[i]), 4),
            "recall": round(float(class_r[i]), 4),
            "f1_score": round(float(class_f1[i]), 4),
            "support": int(class_supp[i])
        }

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds).tolist()

    report_dict = {
        "model_name": model_name,
        "test_accuracy": round(float(acc), 2),
        "weighted_precision": round(float(precision), 4),
        "weighted_recall": round(float(recall), 4),
        "weighted_f1_score": round(float(f1), 4),
        "macro_f1_score": round(float(macro_f1), 4),
        "avg_inference_latency_ms": round(avg_latency, 2),
        "total_test_samples": len(test_dataset),
        "num_classes": num_classes,
        "classes": classes,
        "per_class_metrics": per_class_results,
        "confusion_matrix": cm
    }

    report_path = OUTPUTS_DIR / f"{model_name}_evaluation.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    print("\n" + "="*50)
    print(f"EVALUATION SUMMARY: {model_name}")
    print("="*50)
    print(f"Test Accuracy:          {acc:.2f}%")
    print(f"Weighted F1 Score:      {f1:.4f}")
    print(f"Macro F1 Score:         {macro_f1:.4f}")
    print(f"Avg CPU Latency:        {avg_latency:.2f} ms/image")
    print(f"Full Report Saved to:   {report_path.name}")
    print("="*50 + "\n")

    return report_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3_small"])
    args = parser.parse_args()
    evaluate_model(args.model)
