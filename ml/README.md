# PlantCare Machine Learning Pipeline

This directory contains the machine learning workflows, data preparation, transfer learning models, evaluation suite, and export pipelines for the **PlantCare** AI Plant Disease Detection platform.

---

## 1. Supported Architectures & Model Registry

PlantCare employs a pluggable Model Registry supporting lightweight, edge-and-CPU-friendly computer vision architectures:

| Model Architecture | Parameters | Test Accuracy | Weighted F1 | Avg CPU Latency | Deployment Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EfficientNet-B0** | 4.02 M | **90.48%** | **0.8845** | 13.03 ms / image | **Recommended Default** (Optimal accuracy-speed trade-off) |
| **MobileNetV3-Small** | 1.52 M | **77.78%** | **0.7135** | **1.91 ms / image** | Ultra-Lightweight (Constrained / low-power CPU environments) |

---

## 2. Supported Plant & Disease Classes (21 Classes)

1. **Apple**:
   - `apple_black_rot` (*Botryosphaeria obtusa*)
   - `apple_cedar_apple_rust` (*Gymnosporangium juniperi-virginianae*)
   - `apple_healthy` (Healthy foliage)
   - `apple_scab` (*Venturia inaequalis*)
2. **Corn (Maize)**:
   - `corn_common_rust` (*Puccinia sorghi*)
   - `corn_healthy` (Healthy foliage)
   - `corn_northern_leaf_blight` (*Exserohilum turcicum*)
3. **Grape**:
   - `grape_black_rot` (*Guignardia bidwellii*)
   - `grape_esca_black_measles` (*Phaeoacremonium / Phaeomoniella*)
   - `grape_healthy` (Healthy foliage)
4. **Pepper (Bell / Chili)**:
   - `pepper_bell_bacterial_spot` (*Xanthomonas euvesicatoria*)
   - `pepper_bell_healthy` (Healthy foliage)
5. **Potato**:
   - `potato_early_blight` (*Alternaria solani*)
   - `potato_healthy` (Healthy foliage)
   - `potato_late_blight` (*Phytophthora infestans*)
6. **Tomato**:
   - `tomato_bacterial_spot` (*Xanthomonas campestris*)
   - `tomato_early_blight` (*Alternaria solani*)
   - `tomato_healthy` (Healthy foliage)
   - `tomato_late_blight` (*Phytophthora infestans*)
   - `tomato_septoria_leaf_spot` (*Septoria lycopersici*)
   - `tomato_yellow_leaf_curl_virus` (*TYLCV*)

---

## 3. Data Preprocessing & Augmentation Pipeline

- **Resolution**: Normalized to $224 \times 224 \times 3$ for standard convolutional receptive fields.
- **Normalization**: Standard ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- **Augmentation Techniques**:
  - Random Horizontal Flip ($p = 0.5$)
  - Random Vertical Flip ($p = 0.2$)
  - Random Rotation ($\pm 15^\circ$)
  - Color Jitter (Brightness $\pm 0.2$, Contrast $\pm 0.2$, Saturation $\pm 0.2$)
- **Data Splitting**: Stratified **70% Train / 15% Validation / 15% Test** partition with seed isolation to prevent data leakage.

---

## 4. Explainable AI (Grad-CAM)

PlantCare generates Class Activation Maps using Gradient-weighted Class Activation Mapping (Grad-CAM) hooked directly into the final convolutional layers:
- **EfficientNet-B0**: `model.features[-1]`
- **MobileNetV3-Small**: `model.features[-1]`

Grad-CAM extracts the gradients of the target class score with respect to feature activation maps, applies global average pooling to obtain channel weights, applies ReLU, normalizes the activation map, and overlays a blended `JET` colormap onto the original leaf image.

---

## 5. ML CLI Commands

### Prepare Dataset
```bash
python ml/scripts/prepare_data.py
```

### Train Models
```bash
# Train EfficientNet-B0 (Recommended)
python ml/scripts/train.py --model efficientnet_b0 --epochs 5 --batch-size 16 --lr 0.001

# Train MobileNetV3-Small
python ml/scripts/train.py --model mobilenet_v3_small --epochs 5 --batch-size 16 --lr 0.001
```

### Evaluate Models
```bash
python ml/scripts/evaluate.py --model efficientnet_b0
python ml/scripts/evaluate.py --model mobilenet_v3_small
```

### Export to Backend
```bash
python ml/scripts/export_model.py
```

### Run CLI Prediction
```bash
python ml/scripts/predict.py --image path/to/leaf.jpg --model efficientnet_b0
```
