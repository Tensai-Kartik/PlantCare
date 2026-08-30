"""
Future-Ready Architectural Abstractions for PlantCare
Defines clean interfaces for modular extensions without requiring major structural rewrites:
1. Leaf Segmentation & Multi-Leaf Detection (e.g., YOLOv8-Seg, Mask R-CNN)
2. Advanced Uncertainty Estimation (e.g., MC Dropout, Deep Ensembles, Evidential Deep Learning)
3. Model Drift & Distribution Monitoring
4. Continuous User Feedback & Active Learning
5. Batch Leaf Processing
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np

class BaseLeafDetector(ABC):
    """
    Interface for future leaf localization, multi-leaf counting, and instance segmentation.
    """
    @abstractmethod
    def detect_leaves(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Returns bounding boxes, segmentation masks, and confidence scores for each leaf instance.
        [
            {"bbox": [x1, y1, x2, y2], "mask": np.ndarray, "confidence": float, "is_primary": bool}
        ]
        """
        pass

    @abstractmethod
    def estimate_severity(self, leaf_mask: np.ndarray, lesion_mask: np.ndarray) -> float:
        """
        Computes disease severity percentage = (lesion_area / total_leaf_area) * 100.
        """
        pass


class BaseUncertaintyEstimator(ABC):
    """
    Interface for advanced uncertainty quantification (Monte Carlo Dropout, Ensembles, etc.).
    """
    @abstractmethod
    def estimate_uncertainty(
        self,
        model: Any,
        input_tensor: Any,
        n_samples: int = 10
    ) -> Dict[str, float]:
        """
        Returns epistemic (model) and aleatoric (data) uncertainty metrics.
        """
        pass


class BaseDriftMonitor(ABC):
    """
    Interface for real-world input distribution drift tracking and out-of-distribution detection.
    """
    @abstractmethod
    def track_input(self, image_features: np.ndarray, prediction: str, confidence: float):
        """
        Records feature embeddings to detect distribution drift over time.
        """
        pass

    @abstractmethod
    def get_drift_metrics(self) -> Dict[str, Any]:
        """
        Calculates Population Stability Index (PSI) or Wasserstein distance against baseline.
        """
        pass


class BaseUserFeedbackHandler(ABC):
    """
    Interface for recording agronomist/grower feedback and false-positive reports.
    """
    @abstractmethod
    def record_feedback(
        self,
        request_id: str,
        predicted_class: str,
        user_confirmed_class: str,
        notes: Optional[str] = None
    ) -> bool:
        pass


class BaseBatchProcessor(ABC):
    """
    Interface for batch field image processing and farm canopy survey analysis.
    """
    @abstractmethod
    def process_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        pass
