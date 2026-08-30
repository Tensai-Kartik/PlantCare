# PlantCare Computer Vision Preprocessing Pipeline

This document defines the strict, transparent preprocessing pipeline utilized during both model training and live production inference in PlantCare.

---

## Preprocessing Pipeline Architecture

```
                    ┌─────────────────────────┐
                    │     Uploaded Image      │
                    │   (JPEG / PNG / WebP)   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Optical Quality Check   │
                    │ • Resolution >= 150px   │
                    │ • Blur (Laplacian var)  │
                    │ • Brightness / Contrast │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Multi-Signal Validator  │
                    │ • Bio-Pigment Masks     │
                    │ • MobileNetV3 Semantics │
                    │ • Straight Edge Density │
                    │ • Multi-Leaf Clustered  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   RGB Color Conversion  │
                    │   Image.convert("RGB")  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Geometric Normalization │
                    │ • Resize to (224, 224)  │
                    │   (Bilinear / Lanczos)  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     Tensor Cast         │
                    │ • ToTensor() [0.0, 1.0] │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Statistical Normalization│
                    │ Mean: [0.485,0.456,0.406]│
                    │ Std:  [0.229,0.224,0.225]│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Model Input Tensor      │
                    │ Shape: [1, 3, 224, 224] │
                    └─────────────────────────┘
```

---

## 1. Training vs Inference Consistency

To eliminate training-serving skew, the validation transforms during training and the live inference pipeline share the exact same parameters:

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Input Channels** | 3 (RGB) | Standard 3-channel optical color space |
| **Target Resolution** | $224 \times 224$ px | Optimal trade-off between lesion resolution and CPU latency (<15 ms) |
| **Interpolation** | Bilinear (`PIL.Image.BILINEAR`) | Smooth gradient retention across leaf margins |
| **Tensor Scaling** | $[0.0, 1.0]$ | Standard PyTorch float tensor representation |
| **Normalization Mean** | `[0.485, 0.456, 0.406]` | ImageNet standard channel mean |
| **Normalization Std** | `[0.229, 0.224, 0.225]` | ImageNet standard channel standard deviation |

---

## 2. Bio-Pigment & Spectral Feature Extraction

Before deep neural inference, multi-spectral vegetative indexes are computed on the full-resolution RGB image:

1. **Excess Green Index (ExG)**:
   $$\text{ExG} = 2G - R - B$$
   Isolates active chlorophyll reflectance against neutral background soils.

2. **Chlorotic / Senescent Tissue Mask**:
   $$\text{HSV Range: } H \in [10, 25], S \in [25, 255], V \in [25, 255]$$
   Isolates yellowing viral and nutrient deficiency symptoms.

3. **Necrotic / Rust Tissue Mask**:
   $$\text{HSV Range: } H \in [0, 16] \cup [165, 180], S \in [18, 255], V \in [18, 200]$$
   Isolates dark fungal blight and rust pustules.
