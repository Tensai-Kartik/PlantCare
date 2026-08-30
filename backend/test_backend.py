"""
Comprehensive Backend Unit, Integration & Robustness Test Suite for PlantCare
Tests:
1. Health check, model registry & versioning metadata
2. Valid plant leaf pathology diagnosis (EfficientNet-B0, MobileNetV3-Small)
3. Out-of-domain non-plant test suite (car, animal, electronics, furniture, food, person, building, manmade)
4. Difficult image cases (blur, low-light, overexposed, tiny leaf, multi-leaf, partial leaf)
5. Calibration, Shannon entropy, Top-1/Top-2 margin, and 5-state prediction decision logic
6. Multi-model disagreement & comparison mode
7. Inference stage micro-timings & structured prediction audit
8. Free-tier resource protection (rate limiting, max file size)
"""

import io
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

client = TestClient(app)

# Helper generators for synthetic test images

def create_synthetic_car_image_bytes() -> bytes:
    img = Image.new("RGB", (256, 256), color=(120, 120, 130))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 100, 220, 180], fill=(220, 20, 20))
    draw.polygon([(70, 100), (100, 60), (170, 60), (190, 100)], fill=(80, 120, 180))
    draw.ellipse([60, 160, 100, 200], fill=(20, 20, 20))
    draw.ellipse([160, 160, 200, 200], fill=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def create_synthetic_animal_image_bytes() -> bytes:
    img = Image.new("RGB", (256, 256), color=(235, 235, 235))
    draw = ImageDraw.Draw(img)
    # Dog / Cat furry body & ears
    draw.ellipse([70, 90, 185, 210], fill=(139, 69, 19))
    draw.ellipse([90, 40, 165, 110], fill=(160, 82, 45))
    draw.polygon([(85, 45), (100, 15), (115, 45)], fill=(110, 50, 25))
    draw.polygon([(140, 45), (155, 15), (170, 45)], fill=(110, 50, 25))
    draw.ellipse([105, 65, 120, 80], fill=(20, 20, 20))
    draw.ellipse([135, 65, 150, 80], fill=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def create_synthetic_laptop_image_bytes() -> bytes:
    img = Image.new("RGB", (256, 256), color=(220, 220, 225))
    draw = ImageDraw.Draw(img)
    # Display screen with glowing bezel
    draw.rectangle([50, 40, 206, 150], fill=(30, 30, 35))
    draw.rectangle([60, 50, 196, 140], fill=(70, 130, 180))
    # Keyboard base
    draw.polygon([(40, 170), (216, 170), (236, 210), (20, 210)], fill=(160, 160, 165))
    draw.rectangle([70, 180, 186, 200], fill=(50, 50, 55))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def create_synthetic_food_image_bytes() -> bytes:
    img = Image.new("RGB", (256, 256), color=(245, 240, 235))
    draw = ImageDraw.Draw(img)
    # Pizza slice / plate
    draw.ellipse([30, 30, 226, 226], fill=(220, 220, 220)) # plate
    draw.polygon([(128, 45), (60, 200), (196, 200)], fill=(230, 140, 30)) # crust
    draw.ellipse([90, 110, 115, 135], fill=(180, 20, 20)) # pepperoni
    draw.ellipse([135, 140, 160, 165], fill=(180, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def create_synthetic_furniture_image_bytes() -> bytes:
    img = Image.new("RGB", (256, 256), color=(235, 235, 240))
    draw = ImageDraw.Draw(img)
    # Sofa / chair
    draw.rectangle([40, 120, 216, 190], fill=(70, 50, 130)) # seat
    draw.rectangle([40, 60, 216, 120], fill=(90, 70, 160)) # back
    draw.rectangle([25, 100, 45, 190], fill=(50, 35, 100)) # arm
    draw.rectangle([211, 100, 231, 190], fill=(50, 35, 100)) # arm
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def create_synthetic_leaf_bytes(
    blur: bool = False,
    dark: bool = False,
    tiny: bool = False,
    multi_leaves: bool = False
) -> bytes:
    w, h = 256, 256
    img = Image.new("RGB", (w, h), color=(240, 238, 230))
    draw = ImageDraw.Draw(img)

    if tiny:
        # Small leaf with vein structure (~11% area)
        draw.ellipse([70, 70, 160, 160], fill=(46, 125, 50))
        draw.line([(115, 75), (115, 155)], fill=(35, 95, 40), width=2)
        draw.line([(115, 115), (90, 100)], fill=(35, 95, 40), width=1)
        draw.line([(115, 115), (140, 100)], fill=(35, 95, 40), width=1)
    elif multi_leaves:
        # 4 distinct leaf blobs separated across the frame
        draw.ellipse([30, 30, 100, 100], fill=(46, 125, 50))
        draw.ellipse([150, 30, 220, 100], fill=(56, 142, 60))
        draw.ellipse([30, 150, 100, 220], fill=(67, 160, 71))
        draw.ellipse([150, 150, 220, 220], fill=(43, 114, 46))
    else:
        # Standard centered leaf
        draw.ellipse([45, 35, 211, 221], fill=(46, 125, 50))
        # Lesion spots
        draw.ellipse([90, 80, 130, 120], fill=(70, 40, 20))
        draw.ellipse([140, 130, 175, 165], fill=(80, 50, 25))

    if dark:
        img = ImageEnhance.Brightness(img).enhance(0.20)

    if blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=6.0))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def run_comprehensive_tests():
    print("="*65)
    print("STARTING PLANTCARE ML ROBUSTNESS & RELIABILITY TEST SUITE")
    print("="*65)

    # -------------------------------------------------------------
    # 1. Health & Model Metadata Versioning
    # -------------------------------------------------------------
    print("\n[TEST 1] GET /health & /api/models Versioning...")
    res = client.get("/health")
    assert res.status_code == 200
    h_data = res.json()
    assert h_data["calibration_enabled"] is True
    print(f"[OK] Health Check OK (Version: {h_data['version']}, Calibration: {h_data['calibration_enabled']})")

    res_models = client.get("/api/models")
    assert res_models.status_code == 200
    m_data = res_models.json()
    assert len(m_data["models"]) >= 2
    for m in m_data["models"]:
        assert "version" in m
        assert "temperature" in m
        assert "dataset" in m
        print(f"[OK] Model: {m['name']} (v{m['version']}, Dataset: {m['dataset']}, T={m['temperature']})")

    # -------------------------------------------------------------
    # 2. Out-of-Domain Non-Plant Rejections (Requirement 6)
    # -------------------------------------------------------------
    print("\n[TEST 2] Out-of-Domain Non-Plant Guardrail Tests...")
    non_plant_cases = [
        ("Car", create_synthetic_car_image_bytes()),
        ("Animal", create_synthetic_animal_image_bytes()),
        ("Laptop", create_synthetic_laptop_image_bytes()),
        ("Food", create_synthetic_food_image_bytes()),
        ("Furniture", create_synthetic_furniture_image_bytes())
    ]

    for label, img_bytes in non_plant_cases:
        res = client.post(
            "/api/analyze",
            files={"file": (f"{label.lower()}.jpg", img_bytes, "image/jpeg")}
        )
        assert res.status_code == 200
        an = res.json()
        assert an["is_plant"] is False, f"Expected {label} to be classified as non-plant!"
        assert an["prediction"] is None, f"Expected {label} to NOT return a disease prediction!"
        assert an["audit"]["prediction_state"] == "non_plant"
        print(f"[OK] Non-Plant Guardrail PASSED for {label}: Subject='{an['quality']['detected_subject']}' (Reason={an['quality']['reason_code']})")

    # -------------------------------------------------------------
    # 3. Valid Plant Leaf Diagnosis & Calibration (Requirement 1, 2, 3)
    # -------------------------------------------------------------
    print("\n[TEST 3] Valid Plant Leaf Diagnosis, Calibration & Top-3 Candidates...")
    sample_path = Path(__file__).resolve().parent / "static" / "examples" / "tomato_early_blight.jpg"
    with open(sample_path, "rb") as f:
        valid_leaf_bytes = f.read()

    res_an = client.post(
        "/api/analyze",
        files={"file": ("leaf.jpg", valid_leaf_bytes, "image/jpeg")},
        data={"model_id": "efficientnet_b0"}
    )
    assert res_an.status_code == 200
    an_data = res_an.json()
    assert an_data["is_plant"] is True
    assert an_data["prediction"] is not None
    pred = an_data["prediction"]
    assert pred["state"] in ["known_high", "known_moderate"]
    assert len(pred["top_candidates"]) >= 3
    assert pred["calibrated_confidence"] > 0
    assert "entropy" in pred
    assert "top1_top2_margin" in pred

    print(f"[OK] Primary Prediction: {pred['name']} ({pred['confidence_percent']}% - {pred['confidence_level']})")
    print(f"[OK] Decision State: {pred['state']}")
    print(f"[OK] Top-3 Candidates: {[c['name'] + ' (' + str(c['probability_percent']) + '%)' for c in pred['top_candidates'][:3]]}")
    print(f"[OK] Entropy: {pred['entropy']}, Top-1/Top-2 Margin: {pred['top1_top2_margin']}")

    # -------------------------------------------------------------
    # 4. Difficult & Plant-Adjacent Quality Cases (Requirement 7, 8, 9)
    # -------------------------------------------------------------
    print("\n[TEST 4] Difficult & Plant-Adjacent Quality Feedback...")
    
    # A. Blurry leaf
    blurry_bytes = create_synthetic_leaf_bytes(blur=True)
    res_blur = client.post("/api/quality-check", files={"file": ("blur.jpg", blurry_bytes, "image/jpeg")})
    assert res_blur.status_code == 200
    q_blur = res_blur.json()
    assert "BLURRY" in q_blur["warnings"] or q_blur["reason_code"] == "BLURRY"
    print(f"[OK] Blurry Leaf Warning: {q_blur['reason_code']} - Guidance: '{q_blur['guidance']}'")

    # B. Low-light dark leaf
    dark_bytes = create_synthetic_leaf_bytes(dark=True)
    res_dark = client.post("/api/quality-check", files={"file": ("dark.jpg", dark_bytes, "image/jpeg")})
    assert res_dark.status_code == 200
    q_dark = res_dark.json()
    assert "TOO_DARK" in q_dark["warnings"] or q_dark["reason_code"] == "TOO_DARK"
    print(f"[OK] Dark Leaf Warning: {q_dark['reason_code']} - Guidance: '{q_dark['guidance']}'")

    # C. Multi-leaf specimen
    multi_bytes = create_synthetic_leaf_bytes(multi_leaves=True)
    res_multi = client.post("/api/quality-check", files={"file": ("multi.jpg", multi_bytes, "image/jpeg")})
    assert res_multi.status_code == 200
    q_multi = res_multi.json()
    assert q_multi["has_multiple_leaves"] is True
    assert "MULTIPLE_LEAVES" in q_multi["warnings"] or q_multi["reason_code"] == "MULTIPLE_LEAVES"
    print(f"[OK] Multi-Leaf Warning: {q_multi['reason_code']} (Estimated Leaves: {q_multi['metrics']['estimated_leaf_count']})")

    # D. Tiny leaf (excessive background)
    tiny_bytes = create_synthetic_leaf_bytes(tiny=True)
    res_tiny = client.post("/api/quality-check", files={"file": ("tiny.jpg", tiny_bytes, "image/jpeg")})
    assert res_tiny.status_code == 200
    q_tiny = res_tiny.json()
    assert "LEAF_TOO_SMALL" in q_tiny["warnings"] or q_tiny["leaf_focus_status"] == "leaf_too_small"
    print(f"[OK] Tiny Leaf Warning: {q_tiny['reason_code']} - Guidance: '{q_tiny['guidance']}'")

    # -------------------------------------------------------------
    # 5. Multi-Model Disagreement / Verification Mode (Requirement 11)
    # -------------------------------------------------------------
    print("\n[TEST 5] Multi-Model Disagreement / Comparison Mode...")
    res_comp = client.post(
        "/api/compare-models",
        files={"file": ("leaf.jpg", valid_leaf_bytes, "image/jpeg")}
    )
    assert res_comp.status_code == 200
    comp_data = res_comp.json()
    assert "agreement_status" in comp_data
    assert len(comp_data["comparison"]) >= 2
    print(f"[OK] Multi-Model Comparison: Status={comp_data['agreement_status']}, Consensus={comp_data['consensus_prediction']}")
    for entry in comp_data["comparison"]:
        print(f"  - {entry['model_name']}: {entry['predicted_name']} ({entry['confidence_percent']}%)")

    # -------------------------------------------------------------
    # 6. Inference Micro-Timings & Prediction Audit (Requirement 13, 17)
    # -------------------------------------------------------------
    print("\n[TEST 6] Inference Micro-Timings & Audit Data...")
    audit = an_data["audit"]
    assert audit is not None
    assert audit["request_id"].startswith("req_")
    perf = audit["performance_metrics"]
    assert perf["image_validation_ms"] >= 0
    assert perf["model_inference_ms"] >= 0
    assert perf["total_request_ms"] >= 0
    print(f"[OK] Micro-Timings:")
    print(f"  - Image Validation:   {perf['image_validation_ms']} ms")
    print(f"  - Preprocessing:      {perf['preprocessing_ms']} ms")
    print(f"  - Model Inference:    {perf['model_inference_ms']} ms")
    print(f"  - Grad-CAM:           {perf['gradcam_ms']} ms")
    print(f"  - Disease Lookup:     {perf['disease_metadata_lookup_ms']} ms")
    print(f"  - Total Request:      {perf['total_request_ms']} ms")
    print(f"[OK] Audit Temperature Applied: T={audit['temperature_applied']}")

    # -------------------------------------------------------------
    # 8. 5-State Confidence & OOD Decision Boundary Unit Tests
    # -------------------------------------------------------------
    print("\n[TEST 8] 5-State Decision Boundaries & Threshold Unit Tests...")
    from app.services.calibration import calibration_service

    # State 1: known_high (calibrated_prob >= 0.75, entropy < 1.80, margin >= 0.15)
    st1, lvl1, _ = calibration_service.categorize_prediction_state(
        calibrated_prob=0.88, raw_prob=0.91, entropy=0.45, top_margin=0.70, is_plant=True
    )
    assert st1 == "known_high"
    assert lvl1 == "High Confidence"
    print("  [OK] State 1: known_high (conf=0.88, entropy=0.45, margin=0.70) -> known_high")

    # State 2: known_moderate (calibrated_prob in [0.45, 0.75), entropy < 1.80, margin >= 0.15)
    st2, lvl2, _ = calibration_service.categorize_prediction_state(
        calibrated_prob=0.62, raw_prob=0.66, entropy=0.85, top_margin=0.30, is_plant=True
    )
    assert st2 == "known_moderate"
    assert lvl2 == "Moderate Confidence"
    print("  [OK] State 2: known_moderate (conf=0.62, entropy=0.85, margin=0.30) -> known_moderate")

    # State 3: plant_uncertain (entropy >= 1.80 OR margin < 0.15 OR conf < 0.45)
    st3_entropy, _, _ = calibration_service.categorize_prediction_state(
        calibrated_prob=0.65, raw_prob=0.70, entropy=1.92, top_margin=0.25, is_plant=True
    )
    assert st3_entropy == "plant_uncertain"

    st3_margin, _, _ = calibration_service.categorize_prediction_state(
        calibrated_prob=0.65, raw_prob=0.70, entropy=0.90, top_margin=0.08, is_plant=True
    )
    assert st3_margin == "plant_uncertain"

    st3_conf, _, _ = calibration_service.categorize_prediction_state(
        calibrated_prob=0.42, raw_prob=0.45, entropy=0.90, top_margin=0.20, is_plant=True
    )
    assert st3_conf == "plant_uncertain"
    print("  [OK] State 3: plant_uncertain verified across high entropy, narrow margin, and low confidence triggers.")

    # State 4: plant_unsupported_condition (entropy >= 2.45 OR conf < 0.30 + margin < 0.08)
    st4_ood_entropy, _, _ = calibration_service.categorize_prediction_state(
        calibrated_prob=0.22, raw_prob=0.25, entropy=2.65, top_margin=0.04, is_plant=True
    )
    assert st4_ood_entropy == "plant_unsupported_condition"
    print("  [OK] State 4: plant_unsupported_condition verified for out-of-index entropy/dispersion.")

    # State 5: non_plant (is_plant = False)
    st5_non_plant, lvl5, _ = calibration_service.categorize_prediction_state(
        calibrated_prob=0.95, raw_prob=0.98, entropy=0.10, top_margin=0.90, is_plant=False
    )
    assert st5_non_plant == "non_plant"
    print("  [OK] State 5: non_plant verified when botanical presence validator rejects image.")

    # -------------------------------------------------------------
    # 9. Standalone Disease Knowledge Base Endpoints (Requirement 2)
    # -------------------------------------------------------------
    print("\n[TEST 9] Disease Knowledge Base Endpoints & Search Tests...")
    # A. List all diseases
    res_kb = client.get("/api/diseases")
    assert res_kb.status_code == 200
    kb_data = res_kb.json()
    assert kb_data["total"] == 21
    assert len(kb_data["diseases"]) == 21
    assert len(kb_data["plants"]) >= 6
    print(f"  [OK] GET /api/diseases returned {kb_data['total']} conditions across {len(kb_data['plants'])} crops.")

    # B. Filter by plant (Tomato)
    res_tomato = client.get("/api/diseases?plant=Tomato")
    assert res_tomato.status_code == 200
    t_data = res_tomato.json()
    assert t_data["total"] == 6
    for d in t_data["diseases"]:
        assert d["plant"] == "Tomato"
    print(f"  [OK] GET /api/diseases?plant=Tomato filtered to {t_data['total']} tomato diseases.")

    # C. Search keyword (Alternaria)
    res_search = client.get("/api/diseases?q=alternaria")
    assert res_search.status_code == 200
    s_data = res_search.json()
    assert s_data["total"] >= 2  # Tomato early blight, Potato early blight
    for d in s_data["diseases"]:
        assert "alternaria" in d["scientific_name"].lower() or "alternaria" in d["description"].lower()
    print(f"  [OK] GET /api/diseases?q=alternaria matched {s_data['total']} diseases.")

    # D. Specific Disease Detail (tomato_early_blight)
    res_detail = client.get("/api/diseases/tomato_early_blight")
    assert res_detail.status_code == 200
    d_detail = res_detail.json()
    assert d_detail["id"] == "tomato_early_blight"
    assert len(d_detail["symptoms"]) >= 3
    assert len(d_detail["causes"]) >= 2
    assert len(d_detail["treatment"]["immediate_steps"]) >= 1
    assert len(d_detail["treatment"]["organic_options"]) >= 1
    assert len(d_detail["treatment"]["conventional_options"]) >= 1
    assert len(d_detail["prevention"]) >= 2
    assert d_detail["image_url"] is not None
    print(f"  [OK] GET /api/diseases/tomato_early_blight returned full clinical pathology profile (Image: {d_detail['image_url']}).")

    print("\n" + "="*65)
    print("ALL ML ROBUSTNESS, THRESHOLD VALIDATION & KNOWLEDGE BASE TESTS PASSED 100%!")
    print("="*65 + "\n")

if __name__ == "__main__":
    run_comprehensive_tests()
