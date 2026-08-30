"""
Explainable AI (Grad-CAM) Service for PlantCare
Computes Class Activation Maps and overlays heatmap attention onto the input leaf image.
"""

import base64
from io import BytesIO
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAMService:
    def __init__(self):
        pass

    def generate_gradcam(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        input_tensor: torch.save,
        raw_pil_image: Image.Image,
        target_class_idx: Optional[int] = None
    ) -> Optional[str]:
        """
        Generates Grad-CAM heatmap overlay as a base64-encoded JPEG.
        """
        model.eval()

        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        f_handle = target_layer.register_forward_hook(forward_hook)
        b_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            # Enable gradient calculation for Grad-CAM
            input_tensor.requires_grad_(True)
            output = model(input_tensor)

            if target_class_idx is None:
                target_class_idx = output.argmax(dim=1).item()

            score = output[0, target_class_idx]
            model.zero_grad()
            score.backward(retain_graph=True)

            if not gradients or not activations:
                return None

            grad = gradients[0].cpu().data.numpy()[0]  # [C, H, W]
            act = activations[0].cpu().data.numpy()[0]   # [C, H, W]

            # Global average pooling of gradients
            weights = np.mean(grad, axis=(1, 2))  # [C]

            # Weighted combination of activation maps
            cam = np.zeros(act.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * act[i]

            # ReLU on activation map
            cam = np.maximum(cam, 0)

            # Normalize between 0 and 1
            if np.max(cam) > 0:
                cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
            else:
                cam = np.zeros_like(cam)

            # Resize CAM to match original image size
            orig_w, orig_h = raw_pil_image.size
            cam_resized = cv2.resize(cam, (orig_w, orig_h))

            # Convert to 8-bit heatmap with JET colormap
            heatmap = np.uint8(255 * cam_resized)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            # Overlay onto original image
            orig_np = np.array(raw_pil_image)
            alpha = 0.45
            overlay = np.uint8(orig_np * (1.0 - alpha) + heatmap_colored * alpha)

            # Encode as base64 JPEG
            result_img = Image.fromarray(overlay)
            buf = BytesIO()
            result_img.save(buf, format="JPEG", quality=90)
            base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{base64_str}"

        except Exception as e:
            print(f"Grad-CAM generation error: {e}")
            return None
        finally:
            f_handle.remove()
            b_handle.remove()

gradcam_service = GradCAMService()
