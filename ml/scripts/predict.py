"""
Inference & Prediction CLI Script for PlantCare
Tests prediction and Grad-CAM visualization from command line.
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from train import build_model, NORM_MEAN, NORM_STD

def predict(image_path: str, model_name: str = "efficientnet_b0"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parent.parent
    ckpt_path = project_root / "models" / f"{model_name}_best.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint {ckpt_path} not found.")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    img_size = checkpoint.get("img_size", 224)

    model = build_model(model_name, num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    raw_img = Image.open(image_path).convert("RGB")
    tensor_img = transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_img)
        probs = F.softmax(outputs, dim=1)[0]

    top5_prob, top5_idx = torch.topk(probs, min(5, len(classes)))

    print("\n" + "="*45)
    print(f"PREDICTION RESULTS ({model_name})")
    print("="*45)
    print(f"Top Prediction: {classes[top5_idx[0]]} ({top5_prob[0]*100:.2f}%)")
    print("-" * 45)
    print("Top-5 Candidates:")
    for i in range(len(top5_prob)):
        idx = top5_idx[i].item()
        p = top5_prob[i].item() * 100
        print(f"  {i+1}. {classes[idx]:<32} : {p:.2f}%")
    print("="*45 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image")
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3_small"])
    args = parser.parse_args()
    predict(args.image, args.model)
