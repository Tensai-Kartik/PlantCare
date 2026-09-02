"""
Computer Vision Model Registry for PlantCare
Provides architecture abstraction, lazy loading, caching, runtime model switching,
confidence calibration parameters, and multi-model consensus verification.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from app.core.config import settings
from app.schemas.analysis import (
    ModelMetadata,
    ModelListResponse,
    ModelDisagreementResult,
    ModelComparisonEntry
)

class BasePlantModel(ABC):
    def __init__(self, model_id: str, name: str, weight_path: Path, metadata: Dict):
        self.model_id = model_id
        self.name = name
        self.weight_path = weight_path
        self.meta_dict = metadata
        self.classes: List[str] = metadata.get("classes", [])
        self.num_classes: int = len(self.classes)
        self.img_size: int = metadata.get("img_size", 224)
        self.version: str = metadata.get("version", "1.2.0")
        self.dataset: str = metadata.get("dataset", "PlantVillage + FieldAug")
        self.dataset_version: str = metadata.get("dataset_version", "2.0")
        self.class_count: int = metadata.get("class_count", self.num_classes)
        self.training_date: str = metadata.get("training_date", "2026-08-26")
        self.temperature: float = metadata.get("temperature", settings.DEFAULT_CALIBRATION_TEMPERATURE)
        self.ece: Optional[float] = metadata.get("ece", 0.038)
        self.norm_mean = metadata.get("norm_mean", [0.485, 0.456, 0.406])
        self.norm_std = metadata.get("norm_std", [0.229, 0.224, 0.225])

        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        ])

    @abstractmethod
    def _construct_architecture(self) -> nn.Module:
        pass

    @abstractmethod
    def get_target_layer_for_gradcam(self) -> nn.Module:
        pass

    def load(self):
        if self.model is not None:
            return

        print(f"Loading {self.name} weights from {self.weight_path} onto {self.device}...")
        self.model = self._construct_architecture()

        if self.weight_path.exists():
            checkpoint = torch.load(self.weight_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Successfully loaded weights for {self.name}.")
        else:
            print(f"Warning: Model weight file {self.weight_path} not found. Running with baseline architecture.")

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        return self.transform(pil_image).unsqueeze(0).to(self.device)

    def to_metadata_schema(self, is_default: bool = False) -> ModelMetadata:
        return ModelMetadata(
            id=self.model_id,
            name=self.name,
            architecture=self.meta_dict.get("architecture", self.model_id),
            version=self.version,
            dataset=self.dataset,
            dataset_version=self.dataset_version,
            image_size=self.img_size,
            class_count=self.class_count,
            training_date=self.training_date,
            temperature=self.temperature,
            ece=self.ece,
            accuracy=self.meta_dict.get("accuracy", 92.5),
            weighted_f1=self.meta_dict.get("weighted_f1", 0.92),
            latency_ms=self.meta_dict.get("latency_ms", 18.5),
            is_default=is_default
        )


class EfficientNetB0Model(BasePlantModel):
    def _construct_architecture(self) -> nn.Module:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, self.num_classes)
        )
        return model

    def get_target_layer_for_gradcam(self) -> nn.Module:
        self.load()
        return self.model.features[-1]


class MobileNetV3SmallModel(BasePlantModel):
    def _construct_architecture(self) -> nn.Module:
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, self.num_classes)
        return model

    def get_target_layer_for_gradcam(self) -> nn.Module:
        self.load()
        return self.model.features[-1]


class ResNet18Model(BasePlantModel):
    def _construct_architecture(self) -> nn.Module:
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)
        return model

    def get_target_layer_for_gradcam(self) -> nn.Module:
        self.load()
        return self.model.layer4[-1]


class ModelRegistry:
    def __init__(self, weights_dir: Optional[Path] = None):
        self.weights_dir = weights_dir or settings.MODEL_WEIGHTS_DIR
        self.models: Dict[str, BasePlantModel] = {}
        self.default_model_id = settings.DEFAULT_MODEL
        self._active_model_id: Optional[str] = None
        self._initialize_registry()

    def _initialize_registry(self):
        meta_file = self.weights_dir / "model_metadata.json"
        metadata = {}

        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load model_metadata.json: {e}")

        models_meta = metadata.get("models", {})
        if "default_model" in metadata:
            self.default_model_id = metadata["default_model"]

        # Default class fallback list if metadata isn't generated yet
        default_classes = [
            "apple_black_rot", "apple_cedar_apple_rust", "apple_healthy", "apple_scab",
            "corn_common_rust", "corn_healthy", "corn_northern_leaf_blight",
            "grape_black_rot", "grape_esca_black_measles", "grape_healthy",
            "pepper_bell_bacterial_spot", "pepper_bell_healthy",
            "potato_early_blight", "potato_healthy", "potato_late_blight",
            "tomato_bacterial_spot", "tomato_early_blight", "tomato_healthy",
            "tomato_late_blight", "tomato_septoria_leaf_spot", "tomato_yellow_leaf_curl_virus"
        ]

        # Register EfficientNet-B0
        eff_meta = models_meta.get("efficientnet_b0", {
            "name": "EfficientNet-B0",
            "architecture": "efficientnet_b0",
            "version": "1.2.0",
            "dataset": "PlantVillage + FieldAug",
            "dataset_version": "2.0",
            "classes": default_classes,
            "img_size": 224,
            "class_count": 21,
            "training_date": "2026-08-26",
            "temperature": 1.15,
            "ece": 0.038,
            "accuracy": 90.48,
            "weighted_f1": 0.8845,
            "latency_ms": 13.03
        })
        self.models["efficientnet_b0"] = EfficientNetB0Model(
            model_id="efficientnet_b0",
            name="EfficientNet-B0",
            weight_path=self.weights_dir / "efficientnet_b0.pth",
            metadata=eff_meta
        )

        # Register MobileNetV3-Small
        mob_meta = models_meta.get("mobilenet_v3_small", {
            "name": "MobileNetV3-Small",
            "architecture": "mobilenet_v3_small",
            "version": "1.2.0",
            "dataset": "PlantVillage + FieldAug",
            "dataset_version": "2.0",
            "classes": default_classes,
            "img_size": 224,
            "class_count": 21,
            "training_date": "2026-08-26",
            "temperature": 1.20,
            "ece": 0.045,
            "accuracy": 77.78,
            "weighted_f1": 0.7135,
            "latency_ms": 1.91
        })
        self.models["mobilenet_v3_small"] = MobileNetV3SmallModel(
            model_id="mobilenet_v3_small",
            name="MobileNetV3-Small",
            weight_path=self.weights_dir / "mobilenet_v3_small.pth",
            metadata=mob_meta
        )

        # Register ResNet-18
        res_meta = models_meta.get("resnet18", {
            "name": "ResNet-18",
            "architecture": "resnet18",
            "version": "1.2.0",
            "dataset": "PlantVillage + FieldAug",
            "dataset_version": "2.0",
            "classes": default_classes,
            "img_size": 224,
            "class_count": 21,
            "training_date": "2026-08-26",
            "temperature": 1.10,
            "ece": 0.035,
            "accuracy": 92.86,
            "weighted_f1": 0.9120,
            "latency_ms": 11.45
        })
        self.models["resnet18"] = ResNet18Model(
            model_id="resnet18",
            name="ResNet-18",
            weight_path=self.weights_dir / "resnet18.pth",
            metadata=res_meta
        )

    def get_model(self, model_id: Optional[str] = None) -> BasePlantModel:
        target_id = model_id or self.default_model_id
        if target_id not in self.models:
            print(f"Requested model '{target_id}' not found; falling back to default '{self.default_model_id}'")
            target_id = self.default_model_id

        model_instance = self.models[target_id]
        model_instance.load()
        self._active_model_id = target_id
        return model_instance

    def list_models(self) -> ModelListResponse:
        model_items = [
            m.to_metadata_schema(is_default=(m.model_id == self.default_model_id))
            for m in self.models.values()
        ]
        return ModelListResponse(
            models=model_items,
            default=self.default_model_id
        )

    def run_model_comparison(self, pil_image: Image.Image) -> ModelDisagreementResult:
        """
        Runs multiple lightweight models (EfficientNet-B0, MobileNetV3-Small, ResNet-18)
        to check for diagnosis consensus / disagreement.
        """
        entries: List[ModelComparisonEntry] = []
        preds_list = []

        for m_id, model_wrapper in self.models.items():
            try:
                model_wrapper.load()
                tensor = model_wrapper.preprocess(pil_image)
                with torch.no_grad():
                    logits = model_wrapper.model(tensor)
                    # Apply calibrated softmax
                    probs = F.softmax(logits / model_wrapper.temperature, dim=1)[0]
                    top_prob, top_idx = torch.topk(probs, 1)

                top_idx_val = top_idx[0].item()
                top_p_val = top_prob[0].item()
                c_id = model_wrapper.classes[top_idx_val]
                c_name = c_id.replace("_", " ").title()

                entries.append(ModelComparisonEntry(
                    model_id=m_id,
                    model_name=model_wrapper.name,
                    predicted_class_id=c_id,
                    predicted_name=c_name,
                    confidence_percent=round(top_p_val * 100.0, 1)
                ))
                preds_list.append((c_id, top_p_val))
            except Exception as e:
                print(f"Model comparison error for {m_id}: {e}")

        if len(entries) < 2:
            return ModelDisagreementResult(
                enabled=True,
                agreement_status="AGREED",
                models_agree=True,
                consensus_prediction=entries[0].predicted_name if entries else None,
                message="Single model evaluated.",
                comparison=entries
            )

        # Compare top predicted classes across models using true majority voting
        from collections import Counter
        class_counts = Counter(p[0] for p in preds_list)
        most_common_class, top_count = class_counts.most_common(1)[0]
        majority_agreed = (top_count >= (len(preds_list) + 1) // 2)

        if majority_agreed:
            consensus_name = next((entry.predicted_name for entry in entries if entry.predicted_class_id == most_common_class), entries[0].predicted_name)
            avg_conf = sum(p[1] for p in preds_list if p[0] == most_common_class) / top_count * 100.0
            return ModelDisagreementResult(
                enabled=True,
                agreement_status="AGREED",
                models_agree=True,
                consensus_prediction=consensus_name,
                message=f"Model Consensus Confirmed: {consensus_name} (Confidence ~{avg_conf:.1f}%).",
                comparison=entries
            )
        else:
            return ModelDisagreementResult(
                enabled=True,
                agreement_status="DISAGREED",
                models_agree=False,
                consensus_prediction=None,
                message="Model Disagreement: Multi-model architectures produced divergent predictions.",
                comparison=entries
            )

model_registry = ModelRegistry()
