"""
Full feature smoke test — hits every new endpoint and checks responses.
"""
import sys, os
# Fix Windows cp1252 encoding for console output
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
import requests, json, base64
from pathlib import Path

BASE = "http://localhost:5000"
MHA  = Path(r"c:\project\Spine Segmentation\10159290\images\100_t2.mha")

PASS = "[PASS]"
FAIL = "[FAIL]"

results = {}

def check(name, cond, detail=""):
    ok = bool(cond)
    results[name] = ok
    tag = PASS if ok else FAIL
    msg = f"  {tag}  {name}"
    if detail: msg += f"  ->  {detail}"
    print(msg)
    return ok

print("=" * 55)
print("  ATM-Net++ Feature Test Suite")
print("=" * 55)

# ── 1. Health check ───────────────────────────────────────────────────
print("\n[1] Health check")
try:
    r = requests.get(f"{BASE}/health", timeout=5).json()
    check("Status running",   r.get("status") == "running")
    check("GPU detected",     r.get("gpu") and "CPU" not in r.get("gpu",""))
    check("Checkpoint loaded",r.get("checkpoint"))
    check("Infer size set",   r.get("infer_size") in [192, 256], str(r.get("infer_size")))
except Exception as e:
    print(f"  {FAIL}  Health check FAILED: {e}"); sys.exit(1)

# ── 2. Predict (main endpoint) ────────────────────────────────────────
print("\n[2] Predict endpoint (uploads real MHA file)")
with open(MHA, "rb") as f:
    resp = requests.post(f"{BASE}/predict",
                         files={"file": ("100_t2.mha", f)},
                         data={"name":"Test Patient","age":"55","sex":"M"},
                         timeout=120)
d = resp.json()

if "error" in d:
    print(f"  {FAIL}  Predict failed: {d['error'][:200]}")
    print(d.get("trace","")[:500])
    sys.exit(1)

# Core fields
check("num_slices >= 4",      d.get("num_slices",0) >= 4,       str(d.get("num_slices")))
check("disease returned",     bool(d.get("disease")))
check("severity returned",    bool(d.get("severity")))
check("pfirrmann returned",   d.get("pfirrmann_grade") is not None)
check("inference_ms",         d.get("inference_ms",0) > 0,      f"{d.get('inference_ms')}ms")
check("record_id saved",      bool(d.get("record_id")))

# Images
for img_key in ["image_b64","overlay_b64","mask_b64","scoliosis_b64","legend_b64"]:
    check(f"image: {img_key}", bool(d.get(img_key)))

# New feature images
check("gradcam_b64 present",      bool(d.get("gradcam_b64")),     "Grad-CAM")
check("uncertainty_b64 present",  bool(d.get("uncertainty_b64")), "MC Dropout")

# Multi-slice
check("slice_thumbs >= 4",    len(d.get("slice_thumbs",[])) >= 4,
      f"{len(d.get('slice_thumbs',[]))} thumbs")

# Analytics dicts
check("ivd_grades populated",  isinstance(d.get("ivd_grades"), dict))
check("curvature returned",    isinstance(d.get("curvature"), dict))
check("disc_heights returned", isinstance(d.get("disc_heights"), dict))
check("stenosis returned",     isinstance(d.get("stenosis"), dict),
      d.get("stenosis",{}).get("risk",""))
check("lordosis returned",     isinstance(d.get("lordosis"), dict),
      d.get("lordosis",{}).get("type",""))
check("t2_signal returned",    isinstance(d.get("t2_signal"), dict))
check("fracture_risk returned",isinstance(d.get("fracture_risk"), dict))
check("uncertainty_mean",      d.get("uncertainty_mean") is not None,
      str(d.get("uncertainty_mean")))

# Deep-check IVD grades
grades = d.get("ivd_grades", {})
visible = {k: v for k,v in grades.items() if v.get("grade") is not None}
check("at least 1 IVD graded", len(visible) >= 1, f"{len(visible)} graded")
if visible:
    lvl, g = next(iter(visible.items()))
    check("grade in 1-5",  1 <= g["grade"] <= 5, f"{lvl}=Grade{g['grade']}")
    check("status string", bool(g.get("status")))

# Report text
rep = d.get("report","")
check("report non-empty",     len(rep) > 100)
check("report has stenosis",  "CANAL STENOSIS" in rep)
check("report has T2",        "T2 SIGNAL" in rep)
check("report has fracture",  "FRACTURE" in rep)
check("report has uncertainty","UNCERTAINTY" in rep)

# ── 3. PDF export ─────────────────────────────────────────────────────
print("\n[3] PDF export")
try:
    pdf_resp = requests.post(f"{BASE}/export_pdf",
                              json=d, timeout=30)
    check("PDF status 200",   pdf_resp.status_code == 200,
          f"status={pdf_resp.status_code}")
    check("PDF content-type", "pdf" in pdf_resp.headers.get("Content-Type","").lower())
    check("PDF non-empty",    len(pdf_resp.content) > 1000,
          f"{len(pdf_resp.content)//1024} KB")
    # Save to disk for manual inspection
    out = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\test_report.pdf")
    out.write_bytes(pdf_resp.content)
    check("PDF saved to disk", out.exists(), str(out))
except Exception as e:
    check("PDF export", False, str(e))

# ── 4. History ────────────────────────────────────────────────────────
print("\n[4] Patient history")
try:
    hist = requests.get(f"{BASE}/history", timeout=5).json()
    check("history returns list",  isinstance(hist, list))
    check("record saved in hist",  any(r.get("id")==d.get("record_id") for r in hist))
    if hist:
        r0 = hist[0]
        check("hist has timestamp",  bool(r0.get("timestamp")))
        check("hist has disease",    bool(r0.get("disease")))
        check("hist has lordosis",   bool(r0.get("lordosis_type")))
        check("hist has stenosis",   bool(r0.get("stenosis_risk")))
        check("hist has uncertainty",r0.get("uncertainty") is not None)
except Exception as e:
    check("History endpoint", False, str(e))

# ── 5. Training endpoint with new features ───────────────────────────
print("\n[5] Training monitor")
try:
    train = requests.get(f"{BASE}/training", timeout=5).json()
    check("training returns history",   isinstance(train.get("history"), list))
    check("checkpoints returned",       isinstance(train.get("checkpoints"), dict))
    check("training history non-empty", len(train.get("history",[])) > 0,
          f"{len(train.get('history',[]))} epochs")
    if train.get("history"):
        h0 = train["history"][0]
        check("epoch field present",    "ep" in h0 or "epoch" in h0)
        check("val dice field present", "vd" in h0)
    check("per_class returned",         isinstance(train.get("per_class"), dict))
except Exception as e:
    check("Training endpoint", False, str(e))

# ── 6. ICD-10 suggestion ─────────────────────────────────────────────
print("\n[6] ICD-10 codes")
try:
    icd_resp = requests.post(f"{BASE}/icd10",
                              json={"disease":"Disc Herniation","severity":"Severe",
                                    "curvature":{"risk":"Mild Scoliosis"},
                                    "stenosis":{"risk":"Stenosis suspected"},
                                    "fracture_risk":{}},
                              timeout=5).json()
    check("icd10 returns codes",   isinstance(icd_resp.get("codes"), list))
    check("at least 1 code",       len(icd_resp.get("codes",[])) >= 1)
    if icd_resp.get("codes"):
        c0 = icd_resp["codes"][0]
        check("code has code field", bool(c0.get("code")))
        check("code has desc field", bool(c0.get("desc")))
        check("ICD format M##",      c0["code"].startswith("M") or c0["code"].startswith("Z"),
              c0["code"])
except Exception as e:
    check("ICD-10 endpoint", False, str(e))

# ── 7. UI feature checks ──────────────────────────────────────────────
print("\n[7] UI feature presence")
try:
    html = requests.get(f"{BASE}/", timeout=5).text
    ui_checks = {
        "toast system"      : "toastContainer" in html,
        "keyboard shortcuts": "kbdModal" in html,
        "chart mode toggle" : "setChartMode" in html,
        "per-class chart"   : "classChart" in html,
        "timeline chart"    : "timelineChart" in html,
        "history search"    : "filterHistory" in html,
        "history stats"     : "histStats" in html,
        "health score ring" : "healthScoreRing" in html,
        "ICD panel"         : "icdDiv" in html,
        "export CSV"        : "exportHistCSV" in html,
        "training CSV"      : "exportTrainCSV" in html,
        "split compare"     : "toggleSplit" in html,
        "heatmap toggle"    : "setOverlay" in html,
        "settings panel"    : "settingsPanel" in html,
        "theme toggle"      : "toggleTheme" in html,
        "keyboard shortcut?": "kbdModal" in html,
    }
    for k, v in ui_checks.items():
        check(k, v)
except Exception as e:
    check("UI check", False, str(e))

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
passed = sum(results.values())
total  = len(results)
pct    = round(passed/total*100)
print(f"  Results: {passed}/{total} passed ({pct}%)")
if passed == total:
    print("  [ALL FEATURES WORKING]")
else:
    failed = [k for k,v in results.items() if not v]
    print(f"  Failed: {failed}")
print("=" * 55)
