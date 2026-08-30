"""
Plant vs. Non-Plant Verification Service for PlantCare
Multi-signal vision engine that validates whether an uploaded image contains an authentic
plant leaf or agricultural crop specimen before running pathology classification.

Signals:
1. Deep Semantic Categorization (ImageNet-1K MobileNetV3-Small)
   Detects vehicles, animals, consumer electronics, apparel, furniture, buildings, tools, food.
2. Botanical & Bio-Pigment Spectral Analysis
   Computes Excess Green Index (ExG), Visible Atmospherically Resistant Index (VARI),
   and multi-spectrum masks (chlorophyll green, chlorotic yellow, necrotic rust/brown).
3. Geometric Structural Edge & Line Density Analysis
   Detects rigid straight-line man-made contours (windshields, wheels, grilles, bezels).
4. Multi-Leaf & Subject Focus Analysis
   Detects multiple leaves, excessive background, partial leaves, or occlusions.
5. Gemini Vision Multimodal Fallback / Verification (Configurable)
"""

import re
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field

import torch
import torchvision.models as models
from torchvision import transforms

@dataclass
class PlantValidationResult:
    is_plant: bool
    status: str  # "suitable", "warning", "rejected"
    detected_subject: str
    subject_category: str  # "plant", "vehicle", "animal", "electronics", "person", "furniture", "food", "manmade", "non_plant"
    plant_confidence: float  # 0.0 to 100.0
    reason_code: str  # "SUITABLE_PLANT", "NON_PLANT_OBJECT", "LEAF_TOO_SMALL", "MULTIPLE_LEAVES", "PARTIAL_LEAF", "OBSTRUCTION", etc.
    warnings: List[str] = field(default_factory=list)
    has_multiple_leaves: bool = False
    leaf_count_estimate: int = 1
    leaf_focus_status: str = "optimal"  # "optimal", "leaf_too_small", "excessive_background", "partial_leaf", "obstructed"
    rejection_reason: Optional[str] = None
    foliage_ratio: float = 0.0
    background_ratio: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


class PlantPresenceValidator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[torch.nn.Module] = None
        self._transform = None
        self._categories: List[str] = []
        self._group_indices: Dict[str, List[int]] = {}
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return

        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            model = models.mobilenet_v3_small(weights=weights)
            model.eval()
            model.to(self.device)
            self._model = model
            self._transform = weights.transforms()
            self._categories = weights.meta["categories"]
            self._build_category_groups()
            self._initialized = True
            print("PlantPresenceValidator: Loaded MobileNetV3-Small ImageNet model successfully.")
        except Exception as e:
            print(f"PlantPresenceValidator initialization warning: {e}")
            self._initialized = False

    def _build_category_groups(self):
        """
        Maps all 1,000 ImageNet categories into semantic domain buckets.
        """
        groups = {
            "vehicle": [],
            "animal": list(range(398)),  # 0-397 are fish, amphibians, reptiles, birds, dogs, mammals
            "electronics": [],
            "apparel_person": [],
            "furniture_building": [],
            "prepared_food": [],
            "tools_household": [],
            "botanical": []
        }

        vehicle_pat = re.compile(
            r'\b(car|cars|cab|taxi|racer|racing car|sports car|convertible|minivan|beach wagon|station wagon|'
            r'jeep|limousine|limo|pickup|truck|trailer|tow truck|fire engine|ambulance|police van|garbage truck|'
            r'moving van|bus|minibus|school bus|trolleybus|motorcycle|motorbike|moped|scooter|bicycle|bike|'
            r'mountain bike|tandem|tricycle|unicycle|kart|go-kart|golfcart|snowmobile|snowplow|locomotive|'
            r'train|railway|bullet train|aircraft|airplane|airliner|warplane|helicopter|shuttle|airship|'
            r'submarine|speedboat|lifeboat|canoe|catamaran|yawl|yacht|liner|ship|vessel|grille|car mirror|'
            r'car wheel|steering wheel|odometer|speedometer)\b',
            re.IGNORECASE
        )

        electronics_pat = re.compile(
            r'\b(laptop|notebook|computer|keyboard|mouse|trackball|monitor|screen|television|tv|'
            r'cellular telephone|mobile phone|dial telephone|ipod|cassette|cd player|radio|tape player|'
            r'printer|camera|projector|joystick|loudspeaker|microphone|vacuum|dishwasher|refrigerator|'
            r'microwave|toaster|modem|remote control|hard disc)\b',
            re.IGNORECASE
        )

        apparel_pat = re.compile(
            r'\b(suit|trench coat|lab coat|fur coat|overcoat|cloak|cardigan|jersey|sweater|kimono|'
            r'sarong|gown|jean|sunglasses|sunglass|bonnet|sombrero|cowboy hat|mortarboard|brassiere|'
            r'bikini|maillot|sock|shoe|clog|running shoe|sandal|boot|uniform|ballplayer|groom|scuba diver)\b',
            re.IGNORECASE
        )

        furn_bldg_pat = re.compile(
            r'\b(dining table|desk|table|rocking chair|folding chair|barber chair|throne|park bench|'
            r'studio couch|sofa|couch|wardrobe|bookcase|four-poster bed|pillow|castle|church|mosque|stupa|'
            r'palace|monastery|cinema|lighthouse|bridge|dam|cliff|geyser|valley|volcano)\b',
            re.IGNORECASE
        )

        food_pat = re.compile(
            r'\b(pizza|cheeseburger|hotdog|french loaf|bagel|pretzel|meat loaf|potpie|burrito|'
            r'carbonara|chocolate sauce|dough|plate|frying pan|wok|teapot|coffee mug|goblet|'
            r'wine bottle|beer bottle|beer glass|cocktail shaker|espresso|ice cream|ice pop|trifle)\b',
            re.IGNORECASE
        )

        tools_pat = re.compile(
            r'\b(hammer|screwdriver|wrench|pliers|hatchet|axe|drill|power drill|saw|chainsaw|shovel|'
            r'spade|rake|scissors|knife|cleaver|chisel|tape measure|anvil|bellows|iron|paintbrush|'
            r'clock|wall clock|analog clock|digital clock|watch|stopwatch|sundial|vase|pitcher|'
            r'bucket|pail|can|barrel|crate|umbrella|mask|shield|dumbbell|barbell|whistle)\b',
            re.IGNORECASE
        )

        botanical_pat = re.compile(
            r'\b(cabbage|head cabbage|broccoli|cauliflower|zucchini|squash|spaghetti squash|'
            r'acorn squash|butternut squash|cucumber|artichoke|bell pepper|cardoon|mushroom|'
            r'granny smith|strawberry|orange|lemon|fig|pineapple|banana|jackfruit|custard apple|'
            r'pomegranate|hay|rapeseed|daisy|yellow lady\'s slipper|corn|ear|acorn|hip|buckeye|'
            r'coral fungus|agaric|gyromitra|stinkhorn|earthstar|hen-of-the-woods|bolete|pot|flowerpot)\b',
            re.IGNORECASE
        )

        for i in range(398, len(self._categories)):
            cat = self._categories[i]
            if vehicle_pat.search(cat):
                groups["vehicle"].append(i)
            elif electronics_pat.search(cat):
                groups["electronics"].append(i)
            elif apparel_pat.search(cat):
                groups["apparel_person"].append(i)
            elif furn_bldg_pat.search(cat):
                groups["furniture_building"].append(i)
            elif food_pat.search(cat):
                groups["prepared_food"].append(i)
            elif tools_pat.search(cat):
                groups["tools_household"].append(i)
            elif botanical_pat.search(cat):
                groups["botanical"].append(i)

        self._group_indices = groups

    def _analyze_multi_leaf_and_focus(
        self,
        tissue_mask: np.ndarray,
        width: int,
        height: int,
        foliage_ratio: float
    ) -> Tuple[bool, int, str, List[str]]:
        """
        Analyzes connected components and contours to detect multiple leaves,
        excessive background, framing, or obstructions.
        """
        warnings = []
        has_multiple_leaves = False
        leaf_count_est = 1
        leaf_focus_status = "optimal"

        # 1. Morphological filtering to isolate distinct leaf bodies
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        # 2. Find contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = width * height

        # Filter contours by minimum significant size (> 3.5% of image area)
        min_leaf_area = total_area * 0.035
        significant_leaf_contours = [c for c in contours if cv2.contourArea(c) > min_leaf_area]

        leaf_count_est = max(1, len(significant_leaf_contours))
        if leaf_count_est >= 3:
            has_multiple_leaves = True
            warnings.append("MULTIPLE_LEAVES")

        # 3. Leaf / Subject Focus Analysis
        if foliage_ratio < 0.15:
            leaf_focus_status = "leaf_too_small"
            warnings.append("LEAF_TOO_SMALL")
        elif foliage_ratio > 0.88:
            # Overfilling or partial leaf cut-off
            leaf_focus_status = "partial_leaf"
            warnings.append("PARTIAL_LEAF")
        elif (1.0 - foliage_ratio) > 0.85:
            leaf_focus_status = "excessive_background"
            warnings.append("EXCESSIVE_BACKGROUND")

        # 4. Check if leaf touches 3+ image borders (partially outside frame)
        if len(significant_leaf_contours) > 0:
            primary_c = max(significant_leaf_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(primary_c)
            border_touches = 0
            if x <= 2: border_touches += 1
            if y <= 2: border_touches += 1
            if (x + w) >= (width - 3): border_touches += 1
            if (y + h) >= (height - 3): border_touches += 1

            if border_touches >= 3 and foliage_ratio < 0.70:
                if "PARTIAL_LEAF" not in warnings:
                    warnings.append("PARTIAL_LEAF")
                    leaf_focus_status = "partial_leaf"

        return has_multiple_leaves, leaf_count_est, leaf_focus_status, warnings

    def validate_image(self, image_bytes: bytes) -> PlantValidationResult:
        """
        Runs comprehensive multi-signal validation to verify if the image is a genuine plant/leaf.
        """
        self._lazy_init()

        # Load image
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = pil_image.size
        cv_img = np.array(pil_image)

        # ---------------------------------------------------------
        # Signal 1: Bio-Pigment & Botanical Spectral Analysis
        # ---------------------------------------------------------
        r = cv_img[:, :, 0].astype(float)
        g = cv_img[:, :, 1].astype(float)
        b = cv_img[:, :, 2].astype(float)

        # Excess Green Index: 2G - R - B
        exg = 2.0 * g - r - b
        exg_positive = exg > 10.0
        exg_ratio = float(np.sum(exg_positive) / (width * height)) if (width * height) > 0 else 0.0

        # Multi-Spectrum Leaf Tissue Masks (HSV)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV)

        # Green leaf foliage (Chlorophyll)
        mask_green = cv2.inRange(hsv, np.array([18, 20, 20]), np.array([95, 255, 255]))
        green_pixels = cv2.countNonZero(mask_green)
        green_ratio = float(green_pixels / (width * height)) if (width * height) > 0 else 0.0

        # Yellow / Chlorotic / Senescent leaf tissue (Requires active saturation S >= 50 and V >= 40)
        mask_yellow = cv2.inRange(hsv, np.array([12, 50, 40]), np.array([25, 255, 255]))
        yellow_pixels = cv2.countNonZero(mask_yellow)
        yellow_ratio = float(yellow_pixels / (width * height)) if (width * height) > 0 else 0.0

        # Brown / Rust / Necrotic blight lesion tissue
        mask_brown_1 = cv2.inRange(hsv, np.array([0, 18, 18]), np.array([16, 255, 200]))
        mask_brown_2 = cv2.inRange(hsv, np.array([165, 18, 18]), np.array([180, 255, 200]))
        mask_brown = cv2.bitwise_or(mask_brown_1, mask_brown_2)

        # In authentic plants, brown lesion tissue is only counted as plant foliage
        # if there is also active chlorophyll (green/yellow) OR positive Excess Green
        if (green_ratio + yellow_ratio) > 0.02 or exg_ratio > 0.03:
            combined_tissue_mask = cv2.bitwise_or(cv2.bitwise_or(mask_green, mask_yellow), mask_brown)
        else:
            combined_tissue_mask = cv2.bitwise_or(mask_green, mask_yellow)

        foliage_pixels = cv2.countNonZero(combined_tissue_mask)
        foliage_ratio = float(foliage_pixels / (width * height)) if (width * height) > 0 else 0.0
        background_ratio = round(max(0.0, 1.0 - foliage_ratio), 3)

        # ---------------------------------------------------------
        # Signal 2: Geometric Straight-Line & Contour Density
        # ---------------------------------------------------------
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, minLineLength=35, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0
        img_diag = np.sqrt(width**2 + height**2)
        total_line_length = 0.0
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                total_line_length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        straight_line_density = float(total_line_length / (img_diag * 10.0)) if img_diag > 0 else 0.0

        # ---------------------------------------------------------
        # Signal 3: ImageNet Deep Semantic Classification
        # ---------------------------------------------------------
        top_name = "Unknown Object"
        top_prob = 0.0
        top_idx = -1
        p_veh = 0.0
        p_anim = 0.0
        p_elec = 0.0
        p_app = 0.0
        p_furn = 0.0
        p_food = 0.0
        p_tool = 0.0
        p_bot = 0.0

        if self._model is not None and self._transform is not None:
            try:
                input_tensor = self._transform(pil_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self._model(input_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    top_p, top_c = torch.topk(probs, 3)

                top_idx = top_c[0].item()
                top_name = self._categories[top_idx]
                top_prob = top_p[0].item()

                p_veh = sum(probs[i].item() for i in self._group_indices.get("vehicle", []))
                p_anim = sum(probs[i].item() for i in self._group_indices.get("animal", []))
                p_elec = sum(probs[i].item() for i in self._group_indices.get("electronics", []))
                p_app = sum(probs[i].item() for i in self._group_indices.get("apparel_person", []))
                p_furn = sum(probs[i].item() for i in self._group_indices.get("furniture_building", []))
                p_food = sum(probs[i].item() for i in self._group_indices.get("prepared_food", []))
                p_tool = sum(probs[i].item() for i in self._group_indices.get("tools_household", []))
                p_bot = sum(probs[i].item() for i in self._group_indices.get("botanical", []))
            except Exception as e:
                print(f"Deep semantic inference error: {e}")

        # Multi-Leaf & Focus Analysis
        has_multi, leaf_est, focus_status, focus_warnings = self._analyze_multi_leaf_and_focus(
            combined_tissue_mask, width, height, foliage_ratio
        )

        metrics_dict = {
            "foliage_ratio_percent": round(foliage_ratio * 100.0, 2),
            "green_ratio_percent": round(green_ratio * 100.0, 2),
            "background_ratio_percent": round(background_ratio * 100.0, 2),
            "exg_ratio_percent": round(exg_ratio * 100.0, 2),
            "straight_lines_detected": line_count,
            "straight_line_density": round(straight_line_density, 3),
            "estimated_leaf_count": leaf_est,
            "top_semantic_class": top_name,
            "top_semantic_prob": round(top_prob * 100.0, 1),
            "vehicle_prob": round(p_veh * 100.0, 1),
            "animal_prob": round(p_anim * 100.0, 1),
            "electronics_prob": round(p_elec * 100.0, 1),
            "furniture_prob": round(p_furn * 100.0, 1),
            "tool_prob": round(p_tool * 100.0, 1)
        }

        # ---------------------------------------------------------
        # Decision Logic: Multi-Tiered Verification & Vetoes
        # ---------------------------------------------------------

        # 1. Definite Vehicle Veto (Car, Truck, Motorcycle, etc.)
        if (p_veh > 0.35) or (p_veh > 0.15 and green_ratio < 0.05) or (top_idx in self._group_indices.get("vehicle", []) and top_prob > 0.20 and green_ratio < 0.05):
            clean_name = top_name.replace("_", " ").title()
            return PlantValidationResult(
                is_plant=False,
                status="rejected",
                detected_subject=f"Vehicle / Automobile ({clean_name})",
                subject_category="vehicle",
                plant_confidence=round(max(0.0, 1.0 - p_veh) * 10.0, 1),
                reason_code="NON_PLANT_OBJECT",
                warnings=["NON_PLANT_VEHICLE"],
                has_multiple_leaves=False,
                leaf_count_estimate=0,
                leaf_focus_status="non_plant",
                rejection_reason=f"The uploaded image was identified as a vehicle ({clean_name}, {top_prob*100:.1f}% confidence), not a plant or leaf.",
                foliage_ratio=round(foliage_ratio * 100.0, 1),
                background_ratio=background_ratio,
                metrics=metrics_dict
            )

        # 2. Definite Animal / Pet Veto (Dog, Cat, Bird, etc.)
        if (p_anim > 0.35) or (p_anim > 0.15 and green_ratio < 0.05) or (top_idx in self._group_indices.get("animal", []) and top_prob > 0.20 and green_ratio < 0.05):
            clean_name = top_name.replace("_", " ").title()
            return PlantValidationResult(
                is_plant=False,
                status="rejected",
                detected_subject=f"Animal / Pet ({clean_name})",
                subject_category="animal",
                plant_confidence=round(max(0.0, 1.0 - p_anim) * 10.0, 1),
                reason_code="NON_PLANT_OBJECT",
                warnings=["NON_PLANT_ANIMAL"],
                has_multiple_leaves=False,
                leaf_count_estimate=0,
                leaf_focus_status="non_plant",
                rejection_reason=f"The uploaded image appears to be an animal or pet ({clean_name}, {top_prob*100:.1f}% confidence).",
                foliage_ratio=round(foliage_ratio * 100.0, 1),
                background_ratio=background_ratio,
                metrics=metrics_dict
            )

        # 3. Definite Electronic Device Veto (Laptop, Phone, Monitor, etc.)
        if (p_elec > 0.35) or (p_elec > 0.15 and green_ratio < 0.05) or (top_idx in self._group_indices.get("electronics", []) and top_prob > 0.20 and green_ratio < 0.05):
            clean_name = top_name.replace("_", " ").title()
            return PlantValidationResult(
                is_plant=False,
                status="rejected",
                detected_subject=f"Electronic Device ({clean_name})",
                subject_category="electronics",
                plant_confidence=round(max(0.0, 1.0 - p_elec) * 10.0, 1),
                reason_code="NON_PLANT_OBJECT",
                warnings=["NON_PLANT_ELECTRONICS"],
                has_multiple_leaves=False,
                leaf_count_estimate=0,
                leaf_focus_status="non_plant",
                rejection_reason=f"The uploaded image appears to be an electronic device ({clean_name}).",
                foliage_ratio=round(foliage_ratio * 100.0, 1),
                background_ratio=background_ratio,
                metrics=metrics_dict
            )

        # 4. Human / Apparel Veto
        if (p_app > 0.40) or (p_app > 0.20 and green_ratio < 0.05) or (top_idx in self._group_indices.get("apparel_person", []) and top_prob > 0.25 and green_ratio < 0.05):
            clean_name = top_name.replace("_", " ").title()
            return PlantValidationResult(
                is_plant=False,
                status="rejected",
                detected_subject=f"Clothing / Person ({clean_name})",
                subject_category="person",
                plant_confidence=round(max(0.0, 1.0 - p_app) * 10.0, 1),
                reason_code="NON_PLANT_OBJECT",
                warnings=["NON_PLANT_PERSON"],
                has_multiple_leaves=False,
                leaf_count_estimate=0,
                leaf_focus_status="non_plant",
                rejection_reason=f"The uploaded image contains human attire or portrait features ({clean_name}).",
                foliage_ratio=round(foliage_ratio * 100.0, 1),
                background_ratio=background_ratio,
                metrics=metrics_dict
            )

        # 5. Furniture / Architecture Veto
        if (p_furn > 0.40) or (p_furn > 0.20 and green_ratio < 0.05) or (top_idx in self._group_indices.get("furniture_building", []) and top_prob > 0.25 and green_ratio < 0.05):
            clean_name = top_name.replace("_", " ").title()
            return PlantValidationResult(
                is_plant=False,
                status="rejected",
                detected_subject=f"Building / Furniture ({clean_name})",
                subject_category="furniture",
                plant_confidence=round(max(0.0, 1.0 - max(p_furn, top_prob)) * 10.0, 1),
                reason_code="NON_PLANT_OBJECT",
                warnings=["NON_PLANT_FURNITURE"],
                has_multiple_leaves=False,
                leaf_count_estimate=0,
                leaf_focus_status="non_plant",
                rejection_reason=f"The image shows architectural or furniture elements ({clean_name}, {top_prob*100:.1f}% confidence).",
                foliage_ratio=round(foliage_ratio * 100.0, 1),
                background_ratio=background_ratio,
                metrics=metrics_dict
            )

        # 6. Prepared Food Veto
        if (p_food > 0.40) or (p_food > 0.20 and green_ratio < 0.05) or (top_idx in self._group_indices.get("prepared_food", []) and top_prob > 0.25 and green_ratio < 0.05):
            clean_name = top_name.replace("_", " ").title()
            return PlantValidationResult(
                is_plant=False,
                status="rejected",
                detected_subject=f"Prepared Food ({clean_name})",
                subject_category="food",
                plant_confidence=round(max(0.0, 1.0 - p_food) * 10.0, 1),
                reason_code="NON_PLANT_OBJECT",
                warnings=["NON_PLANT_FOOD"],
                has_multiple_leaves=False,
                leaf_count_estimate=0,
                leaf_focus_status="non_plant",
                rejection_reason=f"The uploaded image contains prepared food or kitchenware ({clean_name}).",
                foliage_ratio=round(foliage_ratio * 100.0, 1),
                background_ratio=background_ratio,
                metrics=metrics_dict
            )

        # 7. Hardware / Tools / Utensils Veto
        if (p_tool > 0.40) or (p_tool > 0.20 and green_ratio < 0.05) or (top_idx in self._group_indices.get("tools_household", []) and top_prob > 0.25 and green_ratio < 0.05):
            clean_name = top_name.replace("_", " ").title()
            return PlantValidationResult(
                is_plant=False,
                status="rejected",
                detected_subject=f"Hardware / Tool ({clean_name})",
                subject_category="manmade",
                plant_confidence=round(max(0.0, 1.0 - p_tool) * 10.0, 1),
                reason_code="NON_PLANT_OBJECT",
                warnings=["NON_PLANT_TOOL"],
                has_multiple_leaves=False,
                leaf_count_estimate=0,
                leaf_focus_status="non_plant",
                rejection_reason=f"The uploaded image contains mechanical tools or hardware ({clean_name}).",
                foliage_ratio=round(foliage_ratio * 100.0, 1),
                background_ratio=background_ratio,
                metrics=metrics_dict
            )

        # 8. Man-Made Rigid Object / High Straight Line Density with Low Vegetation
        if straight_line_density > 0.04 and foliage_ratio < 0.15:
            return PlantValidationResult(
                is_plant=False,
                status="rejected",
                detected_subject="Man-Made Geometric Object",
                subject_category="manmade",
                plant_confidence=5.0,
                reason_code="NON_PLANT_OBJECT",
                warnings=["RIGID_EDGES"],
                has_multiple_leaves=False,
                leaf_count_estimate=0,
                leaf_focus_status="non_plant",
                rejection_reason="Excessive linear and geometric edges detected with insufficient plant tissue.",
                foliage_ratio=round(foliage_ratio * 100.0, 1),
                background_ratio=background_ratio,
                metrics=metrics_dict
            )

        # 9. Insufficient Organic Plant Foliage Check (< 5% green/yellow foliage tissue and no botanical classification)
        if (green_ratio + yellow_ratio) < 0.05 and p_bot < 0.15:
            return PlantValidationResult(
                is_plant=False,
                status="rejected",
                detected_subject="Non-Plant Subject (No Plant Foliage)",
                subject_category="non_plant",
                plant_confidence=round(foliage_ratio * 100.0, 1),
                reason_code="NON_PLANT_OBJECT",
                warnings=["NO_FOLIAGE"],
                has_multiple_leaves=False,
                leaf_count_estimate=0,
                leaf_focus_status="non_plant",
                rejection_reason=f"No recognizable plant chlorophyll tissue detected ({(green_ratio+yellow_ratio)*100:.1f}% vegetative tissue).",
                foliage_ratio=round(foliage_ratio * 100.0, 1),
                background_ratio=background_ratio,
                metrics=metrics_dict
            )

        # ---------------------------------------------------------
        # PASS: Valid Plant / Leaf Specimen Confirmed
        # ---------------------------------------------------------
        plant_score = min(99.5, max(75.0, 60.0 + (foliage_ratio * 50.0) + (p_bot * 30.0)))
        
        # Determine status and primary reason code
        val_status = "suitable"
        primary_reason_code = "SUITABLE_PLANT"
        if len(focus_warnings) > 0:
            val_status = "warning"
            primary_reason_code = focus_warnings[0]

        return PlantValidationResult(
            is_plant=True,
            status=val_status,
            detected_subject="Plant Leaf / Crop Specimen",
            subject_category="plant",
            plant_confidence=round(plant_score, 1),
            reason_code=primary_reason_code,
            warnings=focus_warnings,
            has_multiple_leaves=has_multi,
            leaf_count_estimate=leaf_est,
            leaf_focus_status=focus_status,
            rejection_reason=None,
            foliage_ratio=round(foliage_ratio * 100.0, 1),
            background_ratio=background_ratio,
            metrics=metrics_dict
        )

plant_validator = PlantPresenceValidator()
