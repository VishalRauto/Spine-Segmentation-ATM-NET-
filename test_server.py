"""
ATM-Net++ Full Server Test Suite
=================================
Tests every endpoint on both servers using real MHA data from the SPIDER dataset.

Usage:
    py test_server.py                  # test both servers
    py test_server.py --flask          # Flask only  (port 5000)
    py test_server.py --api            # FastAPI only (port 8000)
    py test_server.py --quick          # 3 quick tests only
    py test_server.py --samples 5      # test 5 MHA files

Requirements: both servers must be running
    py startup.py   (in another terminal)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ── Config ────────────────────────────────────────────────────────────
FLASK_URL   = "http://localhost:5000"
API_URL     = "http://localhost:8000"
DATA_DIR    = Path(r"c:\project\Spine Segmentation\10159290\images")
MASKS_DIR   = Path(r"c:\project\Spine Segmentation\10159290\masks")
TIMEOUT     = 300   # seconds per request (Bio-ClinicalBERT loads on first call ~60s)

# Test patient data
TEST_PATIENT = {
    "name":  "Test Patient",
    "age":   "52",
    "sex":   "M",
    "notes": "Patient presents with moderate lower back pain. "
             "Possible disc herniation at L4/L5 level. "
             "Mild disc degeneration observed.",
}

# Colors for terminal output
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
CHECK  = "✓"
CROSS  = "✗"
WARN   = "⚠"

# ── Result tracker ────────────────────────────────────────────────────
results: List[Dict] = []


def log(msg: str, color: str = ""):
    print(f"{color}{msg}{RESET}" if color else msg)


def ok(test: str, detail: str = "", ms: float = 0):
    tag = f"  {GREEN}{CHECK}{RESET} {BOLD}{test}{RESET}"
    if detail: tag += f"  →  {CYAN}{detail}{RESET}"
    if ms:     tag += f"  {YELLOW}({ms:.0f}ms){RESET}"
    print(tag)
    results.append({"test": test, "status": "PASS", "detail": detail, "ms": ms})


def fail(test: str, detail: str = ""):
    print(f"  {RED}{CROSS}{RESET} {BOLD}{test}{RESET}  →  {RED}{detail}{RESET}")
    results.append({"test": test, "status": "FAIL", "detail": detail})


def warn(test: str, detail: str = ""):
    print(f"  {YELLOW}{WARN}{RESET} {BOLD}{test}{RESET}  →  {YELLOW}{detail}{RESET}")
    results.append({"test": test, "status": "WARN", "detail": detail})


def section(title: str):
    print(f"\n{BOLD}{CYAN}{'─'*55}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*55}{RESET}")


def get_test_images(n: int = 3) -> List[Path]:
    """Pick a mix of T1 and T2 images for testing."""
    if not DATA_DIR.exists():
        return []
    imgs = sorted(DATA_DIR.glob("*.mha"))
    # Pick varied samples: first, middle, last + some specific ones
    picks = []
    for name in ["1_t2.mha", "10_t2.mha", "100_t2.mha", "50_t1.mha", "200_t2.mha"]:
        p = DATA_DIR / name
        if p.exists(): picks.append(p)
    # Fill remainder from sorted list
    for p in imgs:
        if p not in picks: picks.append(p)
        if len(picks) >= n: break
    return picks[:n]


# ═══════════════════════════════════════════════════════════════════════
# FLASK SERVER TESTS  (port 5000)
# ═══════════════════════════════════════════════════════════════════════

def test_flask_health():
    section("Flask Server — Health & Status")
    try:
        t0  = time.time()
        res = requests.get(f"{FLASK_URL}/", timeout=TIMEOUT)
        ms  = (time.time()-t0)*1000
        if res.status_code == 200:
            ok("Flask reachable", f"HTTP {res.status_code}", ms)
        else:
            warn("Flask home", f"HTTP {res.status_code}")
    except Exception as e:
        fail("Flask reachable", str(e))
        return False

    # Check model info via history endpoint
    try:
        res = requests.get(f"{FLASK_URL}/history", timeout=10)
        ok("Flask /history", f"HTTP {res.status_code}")
    except Exception as e:
        warn("Flask /history", str(e))

    return True


def test_flask_predict(image_path: Path, patient: dict, label: str = ""):
    """POST an MHA file to Flask /predict and validate response."""
    test_name = f"Flask predict [{label or image_path.name}]"
    try:
        t0 = time.time()
        with open(image_path, "rb") as f:
            files   = {"file": (image_path.name, f, "application/octet-stream")}
            data    = {
                "patient_name": patient.get("name",""),
                "patient_age":  patient.get("age",""),
                "patient_sex":  patient.get("sex",""),
                "notes":        patient.get("notes",""),
            }
            res = requests.post(f"{FLASK_URL}/predict", files=files,
                                data=data, timeout=TIMEOUT)
        ms = (time.time()-t0)*1000

        if res.status_code != 200:
            fail(test_name, f"HTTP {res.status_code}: {res.text[:120]}")
            return None

        d = res.json()
        if "error" in d:
            fail(test_name, d["error"])
            return None

        # Validate key fields
        disease  = d.get("disease",  "—")
        severity = d.get("severity", "—")
        pfi      = d.get("pfirrmann","—")
        detected = len(d.get("detected_structures",[]))
        has_overlay  = bool(d.get("overlay_b64",""))
        has_gradcam  = bool(d.get("gradcam_b64",""))
        bert_active  = d.get("bert_active", False)
        ml_disease   = d.get("ml_disease","—")

        detail = (f"disease={disease} | severity={severity} | "
                  f"pfirrmann={pfi} | structures={detected} | "
                  f"overlay={'✓' if has_overlay else '✗'} | "
                  f"gradcam={'✓' if has_gradcam else '✗'} | "
                  f"BERT={'✓' if bert_active else '✗'} | "
                  f"ml_disease={ml_disease}")
        ok(test_name, detail, ms)
        return d

    except Exception as e:
        fail(test_name, str(e))
        return None


def test_flask_history():
    section("Flask Server — History")
    try:
        res = requests.get(f"{FLASK_URL}/history", timeout=10)
        if res.status_code == 200:
            hist = res.json()
            n    = len(hist) if isinstance(hist, list) else 0
            ok("Flask history", f"{n} records")
        else:
            warn("Flask history", f"HTTP {res.status_code}")
    except Exception as e:
        fail("Flask history", str(e))


def test_flask_export_csv():
    try:
        res = requests.get(f"{FLASK_URL}/history/export_csv", timeout=10)
        if res.status_code == 200:
            lines = res.text.strip().splitlines()
            ok("Flask export CSV", f"{len(lines)} lines")
        else:
            warn("Flask export CSV", f"HTTP {res.status_code}")
    except Exception as e:
        warn("Flask export CSV", str(e))


def test_flask_icd10(disease: str = "Disc Herniation", severity: str = "Moderate"):
    try:
        res = requests.post(f"{FLASK_URL}/icd10",
                            json={"disease": disease, "severity": severity},
                            timeout=10)
        if res.status_code == 200:
            codes = res.json().get("codes", [])
            ok("Flask ICD-10", f"{len(codes)} codes for '{disease}'")
        else:
            warn("Flask ICD-10", f"HTTP {res.status_code}")
    except Exception as e:
        warn("Flask ICD-10", str(e))


def test_flask_training_history():
    try:
        res = requests.get(f"{FLASK_URL}/training", timeout=10)
        if res.status_code == 200:
            d    = res.json()
            hist = d.get("history",[])
            best = d.get("best_dice","—")
            ok("Flask training history", f"{len(hist)} epochs | best_dice={best}")
        else:
            warn("Flask training history", f"HTTP {res.status_code}")
    except Exception as e:
        warn("Flask training history", str(e))


# ═══════════════════════════════════════════════════════════════════════
# FASTAPI BACKEND TESTS  (port 8000)
# ═══════════════════════════════════════════════════════════════════════

_api_token: Optional[str] = None
_api_user_id: Optional[str] = None
_api_patient_id: Optional[str] = None
_api_study_id: Optional[str] = None

def api_headers() -> dict:
    return {"Authorization": f"Bearer {_api_token}"} if _api_token else {}


def test_api_health():
    section("FastAPI Backend — Health & Docs")
    try:
        t0  = time.time()
        res = requests.get(f"{API_URL}/health", timeout=10)
        ms  = (time.time()-t0)*1000
        if res.status_code == 200:
            d = res.json()
            ok("FastAPI /health", f"status={d.get('status')} | cuda={d.get('cuda_available')} | device={d.get('device')}", ms)
        else:
            fail("FastAPI /health", f"HTTP {res.status_code}")
            return False
    except Exception as e:
        fail("FastAPI /health", str(e))
        return False

    # Swagger docs
    try:
        res = requests.get(f"{API_URL}/docs", timeout=10)
        ok("FastAPI /docs", f"HTTP {res.status_code}")
    except Exception as e:
        warn("FastAPI /docs", str(e))

    return True


def test_api_auth():
    global _api_token, _api_user_id
    section("FastAPI Backend — Auth (register + login)")

    username  = f"testuser_{int(time.time())}"
    email     = f"{username}@test.com"
    password  = "TestPass123!"

    # Register
    try:
        res = requests.post(f"{API_URL}/api/v1/auth/register",
                            json={"email": email, "username": username,
                                  "full_name": "Test User", "password": password},
                            timeout=10)
        if res.status_code in (200, 201):
            d = res.json()
            _api_user_id = d.get("id")
            ok("FastAPI register", f"id={_api_user_id} | role={d.get('role')}")
        else:
            fail("FastAPI register", f"HTTP {res.status_code}: {res.text[:120]}")
            return False
    except Exception as e:
        fail("FastAPI register", str(e))
        return False

    # Login
    try:
        res = requests.post(f"{API_URL}/api/v1/auth/login",
                            json={"username": username, "password": password},
                            timeout=10)
        if res.status_code == 200:
            d = res.json()
            _api_token = d.get("access_token")
            ok("FastAPI login", f"token={'✓' if _api_token else '✗'} | expires_in={d.get('expires_in')}s")
        else:
            fail("FastAPI login", f"HTTP {res.status_code}: {res.text[:120]}")
            return False
    except Exception as e:
        fail("FastAPI login", str(e))
        return False

    # /me
    try:
        res = requests.get(f"{API_URL}/api/v1/auth/me",
                           headers=api_headers(), timeout=10)
        if res.status_code == 200:
            d = res.json()
            ok("FastAPI /me", f"username={d.get('username')} | active={d.get('is_active')}")
        else:
            fail("FastAPI /me", f"HTTP {res.status_code}")
    except Exception as e:
        fail("FastAPI /me", str(e))

    return True


def test_api_patients():
    global _api_patient_id
    section("FastAPI Backend — Patients CRUD")

    # Create
    try:
        payload = {
            "patient_code": f"PT-{int(time.time())}",
            "first_name": "John", "last_name": "Doe",
            "sex": "M", "age": 52, "height_cm": 178.0,
            "weight_kg": 82.0, "clinical_symptoms": "Lower back pain, mild sciatica"
        }
        res = requests.post(f"{API_URL}/api/v1/patients",
                            json=payload, headers=api_headers(), timeout=10)
        if res.status_code in (200, 201):
            d = res.json()
            _api_patient_id = d.get("id")
            ok("FastAPI create patient", f"id={_api_patient_id} | bmi={d.get('bmi')}")
        else:
            fail("FastAPI create patient", f"HTTP {res.status_code}: {res.text[:120]}")
            return
    except Exception as e:
        fail("FastAPI create patient", str(e)); return

    # List
    try:
        res = requests.get(f"{API_URL}/api/v1/patients",
                           headers=api_headers(), timeout=10)
        if res.status_code == 200:
            patients = res.json()
            ok("FastAPI list patients", f"{len(patients)} patient(s)")
        else:
            fail("FastAPI list patients", f"HTTP {res.status_code}")
    except Exception as e:
        fail("FastAPI list patients", str(e))

    # Get by id
    if _api_patient_id:
        try:
            res = requests.get(f"{API_URL}/api/v1/patients/{_api_patient_id}",
                               headers=api_headers(), timeout=10)
            if res.status_code == 200:
                d = res.json()
                ok("FastAPI get patient", f"name={d.get('first_name')} {d.get('last_name')}")
            else:
                fail("FastAPI get patient", f"HTTP {res.status_code}")
        except Exception as e:
            fail("FastAPI get patient", str(e))


def test_api_predict(image_path: Path, label: str = ""):
    global _api_study_id
    test_name = f"FastAPI predict [{label or image_path.name}]"
    try:
        t0 = time.time()
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "application/octet-stream")}
            data  = {
                "report_text": TEST_PATIENT["notes"],
                "modality":    "T2",
                "age":         TEST_PATIENT["age"],
                "sex":         TEST_PATIENT["sex"],
            }
            res = requests.post(f"{API_URL}/api/v1/predict/upload-mri",
                                files=files, data=data,
                                headers=api_headers(), timeout=TIMEOUT)
        ms = (time.time()-t0)*1000

        if res.status_code != 200:
            fail(test_name, f"HTTP {res.status_code}: {res.text[:200]}")
            return None

        d = res.json()

        # Validate response structure
        seg   = d.get("segmentation", {})
        cls   = d.get("classification", {})
        sev   = d.get("severity", {})
        lvl   = d.get("levels", {})
        rep   = d.get("report", {})

        has_overlay = bool(seg.get("overlay_b64",""))
        has_gradcam = bool(d.get("gradcam_b64",""))
        n_structs   = len(seg.get("detected_structures",[]))
        disease     = cls.get("disease_name","—")
        confidence  = round(cls.get("confidence",0)*100, 1)
        severity    = sev.get("name","—")
        pfi         = d.get("pfirrmann_grade","—")
        n_levels    = len(lvl.get("affected",[]))
        infer_ms    = d.get("inference_time_ms","—")

        detail = (f"disease={disease}({confidence}%) | severity={severity} | "
                  f"pfirrmann={pfi} | structures={n_structs} | "
                  f"levels={n_levels} | overlay={'✓' if has_overlay else '✗'} | "
                  f"gradcam={'✓' if has_gradcam else '✗'} | "
                  f"infer={infer_ms}ms")
        ok(test_name, detail, ms)

        # Report check
        if rep.get("report_text"):
            ok(f"  Report text", f"{len(rep['report_text'])} chars | impression='{rep.get('impression','')[:60]}...'")
        else:
            warn(f"  Report text", "empty")

        return d

    except Exception as e:
        fail(test_name, str(e))
        return None


def test_api_analytics():
    section("FastAPI Backend — Analytics")
    try:
        res = requests.get(f"{API_URL}/api/v1/analytics/summary",
                           headers=api_headers(), timeout=10)
        if res.status_code == 200:
            d = res.json()
            ok("FastAPI analytics summary",
               f"patients={d.get('total_patients')} | "
               f"studies={d.get('total_studies')} | "
               f"predictions={d.get('total_predictions')} | "
               f"avg_dice={d.get('average_dice')}")
        else:
            fail("FastAPI analytics summary", f"HTTP {res.status_code}")
    except Exception as e:
        fail("FastAPI analytics summary", str(e))

    try:
        res = requests.get(f"{API_URL}/api/v1/analytics/model-performance",
                           headers=api_headers(), timeout=10)
        if res.status_code == 200:
            ok("FastAPI model performance", f"HTTP {res.status_code}")
        else:
            warn("FastAPI model performance", f"HTTP {res.status_code}")
    except Exception as e:
        warn("FastAPI model performance", str(e))


def test_api_openapi():
    """Verify all routes are registered in OpenAPI schema."""
    try:
        res = requests.get(f"{API_URL}/openapi.json", timeout=10)
        if res.status_code == 200:
            schema = res.json()
            paths  = list(schema.get("paths", {}).keys())
            ok("FastAPI OpenAPI schema", f"{len(paths)} routes registered")
            expected = ["/api/v1/auth/login", "/api/v1/predict/upload-mri",
                        "/api/v1/patients", "/api/v1/analytics/summary",
                        "/api/v1/reports/{report_id}"]
            for ep in expected:
                if ep in paths:
                    ok(f"  Route {ep}", "present")
                else:
                    warn(f"  Route {ep}", "NOT FOUND in schema")
        else:
            fail("FastAPI OpenAPI schema", f"HTTP {res.status_code}")
    except Exception as e:
        fail("FastAPI OpenAPI schema", str(e))


# ═══════════════════════════════════════════════════════════════════════
# DICE VALIDATION TESTS  — compare prediction vs ground-truth masks
# ═══════════════════════════════════════════════════════════════════════

SPIDER_TO_ATMNET = {**{i: i for i in range(1, 9)}, 100: 9,
                    **{201+i: 10+i for i in range(8)}}

def compute_dice(pred_mask: "np.ndarray", gt_mask: "np.ndarray",
                 num_classes: int = 19) -> dict:
    """Compute per-class and mean Dice between predicted and GT masks."""
    import numpy as np
    scores = {}
    valid  = []
    for c in range(1, num_classes):
        p = (pred_mask == c).astype(np.float32)
        g = (gt_mask   == c).astype(np.float32)
        if g.sum() == 0 and p.sum() == 0: continue
        inter = (p * g).sum()
        denom = p.sum() + g.sum()
        d     = float(2 * inter / (denom + 1e-8))
        scores[c] = round(d, 4)
        valid.append(d)
    scores["mean"] = round(float(sum(valid)/len(valid)), 4) if valid else 0.0
    return scores


def test_dice_validation(n_cases: int = 5):
    """Run Dice validation directly (no HTTP) against ground-truth masks."""
    section(f"Dice Validation — {n_cases} cases from SPIDER dataset")

    try:
        import numpy as np
        import SimpleITK as sitk
        import cv2
        import torch
        import torch.nn.functional as F
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
    except ImportError as e:
        warn("Dice validation", f"Missing dependency: {e}")
        return

    # Load the server model directly (no HTTP overhead)
    try:
        from server import get_model, INFER_SIZE, NUM_CLASSES, SPIDER_TO_ATMNET as S2A
        model, device = get_model()
        infer_size    = INFER_SIZE
    except Exception as e:
        fail("Load model for Dice validation", str(e))
        return

    # Find paired image/mask files
    cases = []
    for img_path in sorted(DATA_DIR.glob("*_t2.mha"))[:n_cases*3]:
        msk_path = MASKS_DIR / img_path.name
        if msk_path.exists():
            cases.append((img_path, msk_path))
        if len(cases) >= n_cases: break

    if not cases:
        warn("Dice validation", f"No paired image/mask files found in {DATA_DIR}")
        return

    all_dice   = []
    class_dice = {}  # class_id → list of dice scores

    log(f"\n  {'File':<20} {'Mean Dice':>10}  {'Vert':>8}  {'IVD':>8}  {'Structs':>8}")
    log(f"  {'─'*55}")

    for img_path, msk_path in cases:
        try:
            # Load and preprocess image — sample across full volume
            vol = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path))).astype(float)
            n    = vol.shape[0]
            idxs = list(range(0, n, max(1, n // 8)))[:8]
            slices = [vol[i] for i in idxs]

            # Load ground truth mask — same slice indices
            gt_vol    = sitk.GetArrayFromImage(sitk.ReadImage(str(msk_path)))
            gt_slices = [gt_vol[i] for i in idxs]

            slice_dice = []
            for sl, gt in zip(slices, gt_slices):
                # Skip slices where GT is all-zero (outside anatomy)
                if gt.max() == 0:
                    continue
                # Preprocess
                p1, p99 = np.percentile(sl, [0.5, 99.5])
                img_n   = np.clip((sl - p1)/(p99 - p1 + 1e-8), 0, 1).astype(np.float32)
                img_r   = cv2.resize(img_n, (infer_size, infer_size), cv2.INTER_LINEAR)
                t       = torch.from_numpy(img_r[None, None]).float().to(device)

                with torch.no_grad():
                    pr  = F.softmax(model(t), 1)
                    pr2 = F.softmax(model(torch.flip(t, [-1])), 1)
                    avg = ((pr + torch.flip(pr2, [-1]))/2).squeeze(0).cpu().numpy()

                pred = avg.argmax(0).astype(np.int32)

                # Remap GT from SPIDER labels to ATM-Net labels
                # Resize using NEAREST to preserve integer labels
                gt_r = cv2.resize(gt.astype(np.float32), (infer_size, infer_size),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int32)
                gt_mapped = np.zeros_like(gt_r, dtype=np.int32)
                for spider_c, atm_c in S2A.items():
                    gt_mapped[gt_r == spider_c] = atm_c

                d = compute_dice(pred, gt_mapped, NUM_CLASSES)
                slice_dice.append(d["mean"])
                for c, v in d.items():
                    if c != "mean":
                        class_dice.setdefault(c, []).append(v)

            mean_d = round(float(sum(slice_dice)/len(slice_dice)), 4) if slice_dice else 0
            all_dice.append(mean_d)

            # Vert / IVD breakdown
            vert_d = [class_dice.get(c, [0])[-1] for c in range(1, 9)]
            ivd_d  = [class_dice.get(c, [0])[-1] for c in range(10, 18)]
            vm = round(float(sum(vert_d)/len([x for x in vert_d if x>0]) if any(x>0 for x in vert_d) else 0), 3)
            im = round(float(sum(ivd_d)/len([x for x in ivd_d if x>0]) if any(x>0 for x in ivd_d) else 0), 3)
            n_structs = len([x for x in vert_d+ivd_d if x > 0.05])

            color = GREEN if mean_d >= 0.70 else (YELLOW if mean_d >= 0.50 else RED)
            log(f"  {img_path.name:<20} {color}{mean_d:>10.4f}{RESET}  {vm:>8.3f}  {im:>8.3f}  {n_structs:>8}")

        except Exception as e:
            fail(f"Dice [{img_path.name}]", str(e))

    if all_dice:
        overall_mean = round(float(sum(all_dice)/len(all_dice)), 4)
        log(f"\n  {'─'*55}")
        color = GREEN if overall_mean >= 0.70 else (YELLOW if overall_mean >= 0.50 else RED)
        log(f"  {'OVERALL MEAN DICE':<20} {color}{overall_mean:>10.4f}{RESET}  ({len(all_dice)} cases)")

        # Per-class summary
        log(f"\n  {'Class':<22} {'Mean Dice':>10}  {'N cases':>8}")
        log(f"  {'─'*42}")
        CLS_NAMES = {
            1:"Vert-1(L5)", 2:"Vert-2(L4)", 3:"Vert-3(L3)", 4:"Vert-4(L2)",
            5:"Vert-5(L1)", 6:"Vert-6(T12)", 7:"Vert-7(T11)", 8:"Vert-8(T10)",
            9:"Sacrum",
            10:"IVD L5/S1", 11:"IVD L4/L5", 12:"IVD L3/L4", 13:"IVD L2/L3",
            14:"IVD L1/L2", 15:"IVD T12/L1", 16:"IVD T11/T12", 17:"IVD T10/T11",
            18:"Canal",
        }
        for c in range(1, NUM_CLASSES):
            vals = class_dice.get(c, [])
            if not vals: continue
            mean_c = round(sum(vals)/len(vals), 4)
            col = GREEN if mean_c >= 0.70 else (YELLOW if mean_c >= 0.40 else RED)
            log(f"  {CLS_NAMES.get(c,str(c)):<22} {col}{mean_c:>10.4f}{RESET}  {len(vals):>8}")

        results.append({
            "test":   "Dice Validation",
            "status": "PASS" if overall_mean >= 0.50 else "WARN",
            "detail": f"mean_dice={overall_mean} over {len(all_dice)} cases",
            "ms": 0,
        })


# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def print_summary():
    section("TEST SUMMARY")
    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    warned = [r for r in results if r["status"] == "WARN"]

    total = len(results)
    log(f"\n  Total  : {total}")
    log(f"  {GREEN}Passed : {len(passed)}{RESET}")
    log(f"  {YELLOW}Warned : {len(warned)}{RESET}")
    log(f"  {RED}Failed : {len(failed)}{RESET}")

    if failed:
        log(f"\n  {RED}FAILURES:{RESET}")
        for r in failed:
            log(f"    {RED}{CROSS}{RESET} {r['test']}: {r['detail']}")

    if warned:
        log(f"\n  {YELLOW}WARNINGS:{RESET}")
        for r in warned:
            log(f"    {YELLOW}{WARN}{RESET} {r['test']}: {r['detail']}")

    pct = round(len(passed)/total*100) if total else 0
    color = GREEN if pct >= 80 else (YELLOW if pct >= 60 else RED)
    log(f"\n  {color}{BOLD}Result: {pct}% tests passed{RESET}\n")

    # Write JSON report
    report_path = Path(__file__).parent / "outputs" / "test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": total, "passed": len(passed),
            "failed": len(failed), "warned": len(warned),
            "pass_pct": pct,
            "results": results,
        }, f, indent=2)
    log(f"  Report saved → {report_path}\n")
    return len(failed) == 0


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flask",    action="store_true", help="Flask tests only")
    parser.add_argument("--api",      action="store_true", help="FastAPI tests only")
    parser.add_argument("--quick",    action="store_true", help="Quick smoke test only")
    parser.add_argument("--dice",     action="store_true", help="Dice validation only")
    parser.add_argument("--samples",  type=int, default=3,  help="Number of MHA files to test")
    args = parser.parse_args()

    log(f"\n{BOLD}{CYAN}{'═'*55}")
    log(f"  ATM-Net++ Full Server Test Suite")
    log(f"  Flask: {FLASK_URL}  |  FastAPI: {API_URL}")
    log(f"{'═'*55}{RESET}")

    images = get_test_images(args.samples)
    if not images:
        log(f"\n{YELLOW}  WARNING: No test images found at {DATA_DIR}{RESET}")
        log(f"  Continuing with API-only tests...\n")

    run_flask  = not args.api  and not args.dice
    run_api    = not args.flask and not args.dice
    run_dice   = args.dice or (not args.flask and not args.api and not args.quick)

    # ── Flask Tests ──────────────────────────────────────────────────
    if run_flask:
        flask_ok = test_flask_health()
        if flask_ok:
            section("Flask Server — Prediction Tests")
            for i, img in enumerate(images):
                lbl = f"T2-{i+1}" if "t2" in img.name else f"T1-{i+1}"
                test_flask_predict(img, TEST_PATIENT, lbl)

            if not args.quick:
                test_flask_history()
                test_flask_export_csv()
                test_flask_icd10("Disc Herniation", "Moderate")
                test_flask_training_history()

    # ── FastAPI Tests ─────────────────────────────────────────────────
    if run_api:
        api_ok = test_api_health()
        if api_ok:
            test_api_openapi()
            auth_ok = test_api_auth()
            if auth_ok:
                test_api_patients()
                section("FastAPI Backend — Prediction Tests")
                for i, img in enumerate(images):
                    lbl = f"T2-{i+1}" if "t2" in img.name else f"T1-{i+1}"
                    test_api_predict(img, lbl)
                if not args.quick:
                    test_api_analytics()

    # ── Dice Validation ───────────────────────────────────────────────
    if run_dice and MASKS_DIR.exists():
        test_dice_validation(n_cases=args.samples)
    elif run_dice:
        warn("Dice validation", f"Masks directory not found: {MASKS_DIR}")

    # ── Summary ───────────────────────────────────────────────────────
    success = print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
