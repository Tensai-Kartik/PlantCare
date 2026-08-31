import re
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional
import torch
import torchvision.models as models
from pydantic import BaseModel

class PlantValidationResult(BaseModel):
    is_plant: bool
    status: str                         # "suitable" | "warning" | "rejected"
    detected_subject: str
    subject_category: str               # "plant" | "vehicle" | "animal" | "electronics" | "person" | "furniture" | "food" | "manmade" | "non_plant"
    plant_confidence: float
    reason_code: str                    # "SUITABLE_PLANT" | "NON_PLANT_OBJECT" | "MULTIPLE_LEAVES" | "LEAF_TOO_SMALL" | "PARTIAL_LEAF" | "BLURRY" | "TOO_DARK" | "TOO_BRIGHT"
    warnings: List[str]
    has_multiple_leaves: bool
    leaf_count_estimate: int
    leaf_focus_status: str              # "centered_single" | "multiple_leaves" | "leaf_too_small" | "partial_leaf" | "non_plant"
    rejection_reason: Optional[str] = None
    foliage_ratio: float
    background_ratio: float
    metrics: Dict[str, Any]

class PlantPresenceValidator:
    """
    Production-grade multi-signal botanical presence & specimen focus validator.
    
    Combines:
    1. Vegetative Bio-Pigment & Spectral Indexing (Excess Green ExG = 2G - R - B, Chlorophyll Green & Carotenoid/Necrosis HSV masks)
    2. Geometric Line & Contour Analysis (Edge curvature vs rigid straight-line density)
    3. Deep Semantic Classification (MobileNetV3-Small on 1,000 ImageNet categories)
    4. Multi-Leaf & Spatial Subject Focus Clustering (Connected-component morphology)
    """

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
            "animal": list(range(398)),  # 0-397: fish, amphibians, reptiles, birds, mammals, arthropods
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
        leaf_count_estimate = 1
        focus_status = "centered_single"

        total_area = float(width * height)
        if total_area <= 0:
            return False, 0, "non_plant", []

        # Find connected components of foliage
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tissue_mask, connectivity=8)

        # Filter components by size (> 3% of image area)
        min_comp_area = total_area * 0.03
        significant_components = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_comp_area:
                significant_components.append({
                    "label": i,
                    "area": area,
                    "area_ratio": area / total_area,
                    "x": stats[i, cv2.CC_STAT_LEFT],
                    "y": stats[i, cv2.CC_STAT_TOP],
                    "w": stats[i, cv2.CC_STAT_WIDTH],
                    "h": stats[i, cv2.CC_STAT_HEIGHT],
                    "cx": centroids[i][0],
                    "cy": centroids[i][1]
                })

        # Estimate leaf count
        num_sig = len(significant_components)
        if num_sig >= 2:
            major_comps = [c for c in significant_components if c["area_ratio"] >= 0.04]
            if len(major_comps) >= 2:
                has_multiple_leaves = True
                leaf_count_estimate = len(major_comps)
                focus_status = "multiple_leaves"
                warnings.append("MULTIPLE_LEAVES")
            else:
                leaf_count_estimate = max(1, num_sig)
        else:
            leaf_count_estimate = 1

        # Check leaf size / zoom (framing ratio)
        if foliage_ratio < 0.15:
            focus_status = "leaf_too_small"
            warnings.append("LEAF_TOO_SMALL")
        elif foliage_ratio > 0.88:
            focus_status = "partial_leaf"
            if num_sig == 1 and significant_components:
                c = significant_components[0]
                touches_all_borders = (c["x"] == 0 and c["y"] == 0 and (c["x"] + c["w"] >= width - 1) and (c["y"] + c["h"] >= height - 1))
                if touches_all_borders and foliage_ratio > 0.95:
                    warnings.append("PARTIAL_LEAF")

        return has_multiple_leaves, leaf_count_estimate, focus_status, warnings

    def validate_image(self, image_input: Any) -> PlantValidationResult:
        """
        Runs comprehensive multi-signal botanical validation.
        Accepts raw image bytes, PIL.Image.Image, or numpy array.
        """
        if isinstance(image_input, bytes):
            pil_image = Image.open(BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        else:
            pil_image = Image.fromarray(image_input).convert("RGB")

        self._lazy_init()
        rgb_img = np.array(pil_image)
        height, width, _ = rgb_img.shape
        total_pixels = float(width * height)

        cv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        # ---------------------------------------------------------
        # Signal 1: Bio-Pigment & Spectral Reflectance Analysis
        # ---------------------------------------------------------
        R = rgb_img[:, :, 0].astype(np.float32) / 255.0
        G = rgb_img[:, :, 1].astype(np.float32) / 255.0
        B = rgb_img[:, :, 2].astype(np.float32) / 255.0

        # Excess Green Index: ExG = 2G - R - B (Vegetative index)
        exg = (2.0 * G) - R - B
        exg_positive_pixels = np.sum(exg > 0.05)
        exg_ratio = float(exg_positive_pixels / total_pixels) if total_pixels > 0 else 0.0

        # Chlorophyll Green Hue Mask: HSV H in [25, 95], S in [20, 255], V in [20, 255]
        lower_green = np.array([25, 20, 20])
        upper_green = np.array([95, 255, 255])
        mask_green = cv2.inRange(hsv_img, lower_green, upper_green)
        green_ratio = float(cv2.countNonZero(mask_green) / total_pixels) if total_pixels > 0 else 0.0

        # Carotenoid / Yellow Chlorosis Hue Mask: HSV H in [15, 28]
        lower_yellow = np.array([15, 30, 40])
        upper_yellow = np.array([28, 255, 255])
        mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
        yellow_ratio = float(cv2.countNonZero(mask_yellow) / total_pixels) if total_pixels > 0 else 0.0

        # Necrosis / Brown Lesion Hue Mask: HSV H in [0, 18] and [165, 180]
        lower_brown_1 = np.array([0, 25, 20])
        upper_brown_1 = np.array([18, 255, 200])
        lower_brown_2 = np.array([165, 25, 20])
        upper_brown_2 = np.array([180, 255, 200])
        mask_brown_1 = cv2.inRange(hsv_img, lower_brown_1, upper_brown_1)
        mask_brown_2 = cv2.inRange(hsv_img, lower_brown_2, upper_brown_2)
        mask_brown = cv2.bitwise_or(mask_brown_1, mask_brown_2)
        brown_ratio = float(cv2.countNonZero(mask_brown) / total_pixels) if total_pixels > 0 else 0.0

        # Combine plant tissue mask
        if (green_ratio + yellow_ratio) > 0.02 or exg_ratio > 0.02:
            combined_tissue_mask = cv2.bitwise_or(cv2.bitwise_or(mask_green, mask_yellow), mask_brown)
        else:
            combined_tissue_mask = cv2.bitwise_or(mask_green, mask_yellow)

        foliage_pixels = cv2.countNonZero(combined_tissue_mask)
        foliage_ratio = float(foliage_pixels / total_pixels) if total_pixels > 0 else 0.0
        background_ratio = round(max(0.0, 1.0 - foliage_ratio), 3)

        # ---------------------------------------------------------
        # Signal 2: Geometric Line & Edge Analysis (Canny + Hough)
        # ---------------------------------------------------------
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=10)
        
        line_count = len(lines) if lines is not None else 0
        img_diag = float(np.sqrt(width**2 + height**2))
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
            "yellow_ratio_percent": round(yellow_ratio * 100.0, 2),
            "brown_ratio_percent": round(brown_ratio * 100.0, 2),
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
            "food_prob": round(p_food * 100.0, 1),
            "tool_prob": round(p_tool * 100.0, 1)
        }

        # ---------------------------------------------------------
        # Decision Logic: Robust Multi-Signal Guardrail Hierarchy
        # ---------------------------------------------------------
        clean_name = top_name.replace("_", " ").title()

        # Step 1: Definite Vehicle & Electronics Vetoes (Vehicles, Gadgets, Computers)
        # 1. Vehicle / Automobile Veto (Cars, trucks, station wagons, bikes, boats, planes)
        if p_veh > 0.30 or (top_idx in self._group_indices.get("vehicle", []) and top_prob > 0.20):
            return PlantValidationResult(
                is_plant=False,
                status="rejected",
                detected_subject=f"Vehicle / Automobile ({clean_name})",
                subject_category="vehicle",
                plant_confidence=round(max(0.0, 1.0 - max(p_veh, top_prob)) * 10.0, 1),
                reason_code="NON_PLANT_OBJECT",
                warnings=["NON_PLANT_VEHICLE"],
                has_multiple_leaves=False,
                leaf_count_estimate=0,
                leaf_focus_status="non_plant",
                rejection_reason=f"The uploaded image was verified as a vehicle ({clean_name}, {max(top_prob, p_veh)*100:.1f}% confidence), not a plant specimen.",
                foliage_ratio=round(foliage_ratio * 100.0, 1),
                background_ratio=background_ratio,
                metrics=metrics_dict
            )

        # 2. Electronic Device Veto (Laptops, smartphones, screens, TVs, cameras, clocks)
        if p_elec > 0.30 or (top_idx in self._group_indices.get("electronics", []) and top_prob > 0.20):
            return PlantValidationResult(
                is_plant=False,
                status="rejected",
                detected_subject=f"Electronic Device ({clean_name})",
                subject_category="electronics",
                plant_confidence=round(max(0.0, 1.0 - max(p_elec, top_prob)) * 10.0, 1),
                reason_code="NON_PLANT_OBJECT",
                warnings=["NON_PLANT_ELECTRONICS"],
                has_multiple_leaves=False,
                leaf_count_estimate=0,
                leaf_focus_status="non_plant",
                rejection_reason=f"The uploaded image was verified as an electronic device ({clean_name}).",
                foliage_ratio=round(foliage_ratio * 100.0, 1),
                background_ratio=background_ratio,
                metrics=metrics_dict
            )

        # Step 2: Strong Botanical Foliage Dominance (Guaranteed Plant Leaf)
        # Real agricultural leaves have high chlorophyll green and vegetative tissue.
        # Overrides generic ImageNet insect/ant false positives on foliar necrosis spots.
        is_strongly_botanical = (
            (green_ratio >= 0.15) or
            (foliage_ratio >= 0.35 and green_ratio >= 0.05) or
            (exg_ratio >= 0.15 and foliage_ratio >= 0.30)
        )

        if is_strongly_botanical:
            plant_score = min(99.5, max(88.0, 78.0 + (foliage_ratio * 22.0)))
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

        # Step 3: Definite Non-Plant Semantic Vetoes (Pets, Apparel, Furniture, Food, Hardware)
        # 1. Definite Mammal / Pet / Bird Veto (Dogs, cats, domestic pets, animals)
        # Note: ImageNet classes < 300 are mammals, birds, reptiles, fish (excluding small insects).
        if (top_idx < 300 and top_idx in self._group_indices.get("animal", []) and top_prob > 0.35 and green_ratio < 0.15 and foliage_ratio < 0.35):
            return PlantValidationResult(
                is_plant=False,
                status="rejected",
                detected_subject=f"Animal / Pet ({clean_name})",
                subject_category="animal",
                plant_confidence=round(max(0.0, 1.0 - top_prob) * 10.0, 1),
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

        # 2. Human / Apparel Veto
        if (p_app > 0.25 or (top_idx in self._group_indices.get("apparel_person", []) and top_prob > 0.20)) and green_ratio < 0.25:
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

        # 3. Furniture / Architecture Veto
        if ((p_furn > 0.25 or top_idx in self._group_indices.get("furniture_building", [])) and green_ratio < 0.15 and foliage_ratio < 0.30):
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

        # 4. Prepared Food Veto
        if ((p_food > 0.25 or top_idx in self._group_indices.get("prepared_food", [])) and green_ratio < 0.15 and foliage_ratio < 0.30):
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

        # 5. Hardware / Tools Veto
        if ((p_tool > 0.25 or top_idx in self._group_indices.get("tools_household", [])) and green_ratio < 0.15 and foliage_ratio < 0.30):
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

        # 6. Man-Made Geometric Object / High Straight Line Density with Low Vegetation
        if straight_line_density > 0.035 and foliage_ratio < 0.15 and green_ratio < 0.08:
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

        # Step 4: Botanical Tissue Confirmation (Includes Small / Distant Leaves)
        has_organic_plant_tissue = (
            (green_ratio >= 0.03 and foliage_ratio >= 0.06) or
            (exg_ratio >= 0.03 and foliage_ratio >= 0.06) or
            (p_bot >= 0.15)
        )

        if has_organic_plant_tissue:
            plant_score = min(99.5, max(85.0, 75.0 + (foliage_ratio * 25.0)))
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

        # Step 5: Generic Non-Plant Rejection
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

plant_validator = PlantPresenceValidator()
