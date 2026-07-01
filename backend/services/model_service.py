"""
Model service: bridges FastAPI backend to the trained ResUNet checkpoint.
Uses the same inference logic as server.py but as a proper service class.
ResUNet (Dice 0.77) + Bio-ClinicalBERT + ATPG/HASF/CCAE fusion.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Project root on sys.path
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

_predictor_instance: Optional["ResUNetPredictor"] = None
_lock = asyncio.Lock()


async def get_predictor() -> "ResUNetPredictor":
    global _predictor_instance
    if _predictor_instance is not None:
        return _predictor_instance
    async with _lock:
        if _predictor_instance is not None:
            return _predictor_instance
        _predictor_instance = ResUNetPredictor()
        await asyncio.get_event_loop().run_in_executor(None, _predictor_instance.load)
        return _predictor_instance


# ── Class / label maps (identical to server.py) ───────────────────────
NUM_CLASSES = 19
CLASS_NAMES = {
    0: "Background",       1: "Vertebra-1(L5)",  2: "Vertebra-2(L4)",
    3: "Vertebra-3(L3)",   4: "Vertebra-4(L2)",  5: "Vertebra-5(L1)",
    6: "Vertebra-6(T12)",  7: "Vertebra-7(T11)", 8: "Vertebra-8(T10)",
    9: "Sacrum",
    10: "IVD L5/S1",  11: "IVD L4/L5",  12: "IVD L3/L4",  13: "IVD L2/L3",
    14: "IVD L1/L2",  15: "IVD T12/L1", 16: "IVD T11/T12", 17: "IVD T10/T11",
    18: "Spinal Canal",
}
DISEASE_NAMES = [
    "Normal", "Disc Herniation", "Disc Bulge",
    "Spinal Stenosis", "Disc Degeneration", "Spondylolisthesis", "Compression Fracture",
]
IVD_CLASSES  = list(range(10, 18))
VERT_CLASSES = list(range(1, 9))
IVD_LABELS   = ["L5/S1","L4/L5","L3/L4","L2/L3","L1/L2","T12/L1","T11/T12","T10/T11"]
LEVEL_NAMES  = ["T10/T11","T11/T12","T12/L1","L1/L2","L2/L3","L3/L4","L4/L5","L5/S1"]

COLORS = {
    0:(0,0,0),        1:(220,50,50),    2:(255,100,50),   3:(255,180,50),
    4:(200,230,50),   5:(80,220,80),    6:(50,210,150),   7:(50,180,230),
    8:(50,120,255),   9:(80,80,255),    10:(160,60,255),  11:(240,60,200),
    12:(255,60,120),  13:(200,120,255), 14:(100,200,255), 15:(255,160,60),
    16:(60,240,160),  17:(140,240,80),  18:(220,220,220),
}


class ResUNetPredictor:
    """Wraps the ResUNet checkpoint for use in the FastAPI backend."""

    def __init__(self):
        self.model = None
        self.device = None
        self.infer_size = 384
        self.base_ch = 40
        self._loaded = False

    def load(self):
        """Load model + checkpoint. Called once at startup."""
        import torch
        import torch.nn as nn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Model definition (identical to server.py) ──────────────
        class CA(nn.Module):
            def __init__(self, ch, r=8):
                super().__init__(); r = max(1, ch // r)
                self.avg = nn.AdaptiveAvgPool2d(1); self.max = nn.AdaptiveMaxPool2d(1)
                self.fc  = nn.Sequential(nn.Flatten(), nn.Linear(ch, r), nn.ReLU(True),
                                         nn.Linear(r, ch), nn.Sigmoid())
            def forward(self, x):
                a = self.fc(self.avg(x)) + self.fc(self.max(x))
                return x * a.clamp(0,1).view(x.shape[0],-1,1,1)

        class SA(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(nn.Conv2d(2,1,7,padding=3,bias=False),
                                          nn.BatchNorm2d(1), nn.Sigmoid())
            def forward(self, x):
                return x * self.conv(torch.cat([x.mean(1,keepdim=True),
                                                x.max(1,keepdim=True)[0]],1))

        class RB(nn.Module):
            def __init__(self, ch):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(ch,ch,3,1,1,bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
                    nn.Conv2d(ch,ch,3,1,1,bias=False), nn.BatchNorm2d(ch))
                self.ca = CA(ch); self.sa = SA(); self.act = nn.ReLU(True)
            def forward(self, x): return self.act(self.sa(self.ca(self.net(x)))+x)

        class Enc(nn.Module):
            def __init__(self, ci, co, drop=0.0):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(ci,co,3,1,1,bias=False), nn.BatchNorm2d(co), nn.ReLU(True),
                    nn.Conv2d(co,co,3,1,1,bias=False), nn.BatchNorm2d(co), nn.ReLU(True))
                self.res  = RB(co)
                self.drop = nn.Dropout2d(drop) if drop>0 else nn.Identity()
            def forward(self, x): return self.drop(self.res(self.conv(x)))

        class ResUNet(nn.Module):
            def __init__(self, b=32, nc=NUM_CLASSES, drop=0.25):
                super().__init__()
                self.e1=Enc(1,b); self.e2=Enc(b,b*2,drop*.3)
                self.e3=Enc(b*2,b*4,drop*.6); self.e4=Enc(b*4,b*8,drop*.8)
                self.bn=nn.Sequential(Enc(b*8,b*16,drop),nn.Dropout2d(drop))
                self.pool=nn.MaxPool2d(2)
                self.u4=nn.ConvTranspose2d(b*16,b*8,2,2); self.d4=Enc(b*16,b*8,drop*.4)
                self.u3=nn.ConvTranspose2d(b*8,b*4,2,2);  self.d3=Enc(b*8,b*4,drop*.2)
                self.u2=nn.ConvTranspose2d(b*4,b*2,2,2);  self.d2=Enc(b*4,b*2)
                self.u1=nn.ConvTranspose2d(b*2,b,2,2);    self.d1=Enc(b*2,b)
                self.out=nn.Conv2d(b,nc,1)
            def forward(self, x):
                e1=self.e1(x); e2=self.e2(self.pool(e1))
                e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
                d=self.bn(self.pool(e4))
                d=self.d4(torch.cat([self.u4(d),e4],1))
                d=self.d3(torch.cat([self.u3(d),e3],1))
                d=self.d2(torch.cat([self.u2(d),e2],1))
                d=self.d1(torch.cat([self.u1(d),e1],1))
                return self.out(d)

        # ── Load checkpoint ────────────────────────────────────────
        ckpt_path = _ROOT / "outputs/gpu_run/kaggle_v2.pth"
        self.model = ResUNet(b=self.base_ch, nc=NUM_CLASSES, drop=0.20).to(self.device)
        if ckpt_path.exists():
            ck = torch.load(str(ckpt_path), map_location=self.device)
            missing, _ = self.model.load_state_dict(ck["model_state_dict"], strict=False)
            ep   = ck.get("epoch", "?")
            dice = ck.get("best_dice", 0.0)
            self.infer_size = ck.get("cfg", {}).get("img_size", 384)
            self.base_ch    = ck.get("cfg", {}).get("base_ch", 40)
            logger.info(f"[Model] Loaded kaggle_v2.pth: epoch={ep} dice={dice:.4f}")
        else:
            logger.warning("[Model] Checkpoint not found — using random weights")
        self.model.eval()
        self._loaded = True

    def predict(self, file_bytes: bytes, filename: str,
                report_text: Optional[str] = None,
                demographics: Optional[Dict] = None) -> Dict[str, Any]:
        """Run full prediction pipeline and return structured result dict."""
        import base64
        t0 = time.time()
        pat = demographics or {}

        # ── NLP pipeline (Bio-ClinicalBERT + ATPG/HASF/CCAE) ────────
        # Import the functions registered globally in server.py
        # Only use if BERT is already loaded (don't block on download during tests)
        try:
            from server import (bert_encode, atpg_prompts_from_image,
                                hasf_fuse, ccae_enhance,
                                get_zero_shot, DISEASE_LABELS, SEVERITY_LABELS,
                                _bert_model)
            _nlp_available = _bert_model is not None  # only if already loaded
        except Exception:
            _nlp_available = False

        notes_text  = pat.get("notes", report_text or "") or ""
        notes_lower = notes_text.lower()

        # Rule-based keyword parse
        _PATHO_KW = {
            "Disc Herniation":      ["herniation","herniated","protrusion","extruded"],
            "Disc Bulge":           ["bulge","bulging","broad-based"],
            "Spinal Stenosis":      ["stenosis","canal narrowing","foraminal narrowing"],
            "Disc Degeneration":    ["degeneration","degenerative","desiccation","osteophyte"],
            "Spondylolisthesis":    ["spondylolisthesis","retrolisthesis"],
            "Compression Fracture": ["fracture","compression fracture","wedge"],
        }
        _SEV_KW = {"Mild":["mild","minimal"],"Moderate":["moderate","significant"],
                   "Severe":["severe","marked","critical"]}
        _LEVEL_KW = ["T10/T11","T11/T12","T12/L1","L1/L2","L2/L3","L3/L4","L4/L5","L5/S1"]
        text_diseases, text_levels, text_severity = [], [], ""
        for d, kws in _PATHO_KW.items():
            if any(k in notes_lower for k in kws): text_diseases.append(d)
        for lv in _LEVEL_KW:
            if lv.lower() in notes_lower: text_levels.append(lv)
        for sev, kws in _SEV_KW.items():
            if any(k in notes_lower for k in kws): text_severity = sev; break

        # Demographics
        try:   age = float(pat.get("age", 50))
        except: age = 50.0
        sex           = str(pat.get("sex","")).upper()
        age_risk      = min(max((age - 20) / 60.0, 0.0), 1.0)
        sex_frac_risk = 0.15 if sex == "F" else 0.05

        # Bio-ClinicalBERT encoding
        bert_emb = None
        if _nlp_available and notes_text.strip():
            bert_emb = bert_encode(notes_text)

        # Zero-shot disease classification
        zs_disease, zs_conf, zs_severity = None, 0.0, ""
        if _nlp_available and notes_text.strip():
            try:
                zs = get_zero_shot()
                if zs:
                    res = zs(notes_text, DISEASE_LABELS, multi_label=False)
                    _map = {
                        "normal healthy spine":"Normal","disc herniation":"Disc Herniation",
                        "disc bulge":"Disc Bulge","spinal canal stenosis":"Spinal Stenosis",
                        "degenerative disc disease":"Disc Degeneration",
                        "spondylolisthesis":"Spondylolisthesis",
                        "vertebral compression fracture":"Compression Fracture",
                    }
                    zs_disease = _map.get(res["labels"][0], res["labels"][0])
                    zs_conf    = float(res["scores"][0])
                    sev_res    = zs(notes_text, SEVERITY_LABELS, multi_label=False)
                    zs_severity= sev_res["labels"][0].split()[0].capitalize()
            except Exception as _ze:
                logger.debug(f"ZS error: {_ze}")

        # ── Image loading ────────────────────────────────────────────
        ext = "".join(Path(filename).suffixes).lower()
        slices = self._load_slices(file_bytes, ext)

        # ── Inference ────────────────────────────────────────────────
        all_preds = []
        size = self.infer_size
        for sl in slices:
            p1, p99 = np.percentile(sl, [0.5, 99.5])
            img_n = np.clip((sl - p1) / (p99 - p1 + 1e-8), 0, 1).astype(np.float32)
            img_r = cv2.resize(img_n, (size, size), interpolation=cv2.INTER_LINEAR)
            t     = torch.from_numpy(img_r[None, None]).float().to(self.device)
            with torch.no_grad():
                pr  = F.softmax(self.model(t), 1)
                pr2 = F.softmax(self.model(torch.flip(t, [-1])), 1)
                avg = ((pr + torch.flip(pr2, [-1])) / 2).squeeze(0).cpu().numpy()
            bg_prob = avg[0]; fg_max = avg[1:].max(0)
            bg_bias = np.clip(bg_prob - fg_max - 0.02, 0, None)
            avg[0]  = bg_prob - bg_bias
            avg     = avg / (avg.sum(0, keepdims=True) + 1e-8)
            pred    = avg.argmax(0).astype(np.int32)
            all_preds.append((img_r, pred, avg))

        mid = len(all_preds) // 2
        img_pre, pred, prob_arr = all_preds[mid]
        avg_probs = np.mean([p[2] for p in all_preds], axis=0)

        # ── Disease prediction from image ─────────────────────────────
        ivd_conf  = [float(avg_probs[c].max()) for c in IVD_CLASSES]
        mean_ivd  = float(np.mean(ivd_conf)) if ivd_conf else 0
        if   mean_ivd > 0.75: disease, severity, conf = "Normal",            "None",     round(mean_ivd*100,1)
        elif mean_ivd > 0.50: disease, severity, conf = "Disc Bulge",        "Mild",     round(mean_ivd*100,1)
        elif mean_ivd > 0.30: disease, severity, conf = "Disc Degeneration", "Moderate", round(mean_ivd*100,1)
        else:                 disease, severity, conf = "Disc Herniation",   "Severe",   round((1-mean_ivd)*100,1)
        pfirrmann_overall = round(5 - mean_ivd * 4, 1)

        # ── ATPG + HASF + CCAE fusion ─────────────────────────────────
        if _nlp_available:
            anatomy_prompt = atpg_prompts_from_image(avg_probs, NUM_CLASSES)
            img_feat       = np.array([float(avg_probs[c].max()) for c in range(NUM_CLASSES)])
            fusion_scores  = hasf_fuse(img_feat, bert_emb, age_risk, sex_frac_risk)
            fusion_scores, ccae_disease, ccae_severity = ccae_enhance(
                fusion_scores, text_diseases, text_severity)
        else:
            fusion_scores  = {}
            ccae_disease   = None
            ccae_severity  = None

        # Priority: zero-shot > CCAE > image-only
        ml_disease  = disease
        ml_severity = severity
        if zs_disease and zs_conf > 0.45:
            ml_disease  = zs_disease
            ml_severity = zs_severity or severity
        elif ccae_disease:
            ml_disease  = ccae_disease
            ml_severity = ccae_severity or severity

        # ── Class distribution ────────────────────────────────────────
        cls_dist   = {}
        detected   = []
        for c in range(1, NUM_CLASSES):
            px = int((pred == c).sum())
            if px > 30:
                nm  = CLASS_NAMES[c]
                cls_dist[nm] = round(px / pred.size * 100, 2)
                detected.append(nm)

        # ── Overlay image ─────────────────────────────────────────────
        overlay_b64 = self._make_overlay(img_pre, pred)

        # ── Grad-CAM ─────────────────────────────────────────────────
        gradcam_b64 = self._make_gradcam(img_pre, pred)

        # ── Pfirrmann per IVD ─────────────────────────────────────────
        ivd_grades: Dict[str, Dict] = {}
        for i, ivd_c in enumerate(IVD_CLASSES):
            c_conf = float(avg_probs[ivd_c].max())
            px     = int((avg_probs[ivd_c] > 0.1).sum())
            if px < 20: continue
            if   c_conf > 0.80: grade, status = 1, "Normal"
            elif c_conf > 0.60: grade, status = 2, "Normal with changes"
            elif c_conf > 0.40: grade, status = 3, "Early degeneration"
            elif c_conf > 0.20: grade, status = 4, "Moderate degeneration"
            else:               grade, status = 5, "Severe degeneration"
            ivd_grades[IVD_LABELS[i]] = {"grade": grade, "status": status,
                                          "confidence": round(c_conf, 3)}

        # ── Affected levels ───────────────────────────────────────────
        if text_levels:
            affected_levels = text_levels
        else:
            affected_levels = [IVD_LABELS[i] for i,c in enumerate(IVD_CLASSES)
                               if float(avg_probs[c].max()) > 0.25]

        # ── Report text ───────────────────────────────────────────────
        report_text_out = self._build_report(
            ml_disease, ml_severity, conf, pfirrmann_overall,
            affected_levels, ivd_grades, pat, notes_text
        )

        inference_ms = round((time.time() - t0) * 1000, 1)
        disease_id   = DISEASE_NAMES.index(ml_disease) if ml_disease in DISEASE_NAMES else 0
        severity_id  = {"None":0,"Mild":0,"Moderate":1,"Severe":2}.get(ml_severity, 0)

        return {
            "segmentation": {
                "overlay_b64":        overlay_b64,
                "class_distribution": cls_dist,
                "detected_structures": detected,
            },
            "classification": {
                "disease_id":           disease_id,
                "disease_name":         ml_disease,
                "confidence":           round(conf / 100, 4),
                "disease_probabilities":{d: round(v, 4) for d, v in fusion_scores.items()}
                                         if fusion_scores else {},
            },
            "severity": {"id": severity_id, "name": ml_severity if ml_severity != "None" else "Mild"},
            "levels":   {"affected": affected_levels,
                         "all_probs": {lv: round(float(avg_probs[IVD_CLASSES[i]].max()),4)
                                       for i,lv in enumerate(IVD_LABELS)}},
            "pfirrmann_grade":  pfirrmann_overall,
            "report": {
                "report_text":   report_text_out,
                "findings":      f"Segmentation identified: {', '.join(detected[:6])}.",
                "impression":    f"{ml_disease} — Severity: {ml_severity}. Pfirrmann: {pfirrmann_overall}/5.",
                "recommendation": self._recommendation(ml_disease, ml_severity),
                "disease_name":  ml_disease,
                "severity":      ml_severity,
                "affected_levels": affected_levels,
                "confidence":    round(conf / 100, 4),
                "pfirrmann_grade": pfirrmann_overall,
            },
            "gradcam_b64":          gradcam_b64,
            "inference_time_ms":    inference_ms,
            "num_slices_processed": len(all_preds),
            "ivd_grades":           ivd_grades,
            "bert_active":          bert_emb is not None,
            "zs_disease":           zs_disease,
            "fusion_active":        _nlp_available,
        }

    # ── Helpers ───────────────────────────────────────────────────────

    def _load_slices(self, file_bytes: bytes, ext: str):
        import uuid as _uuid
        if ext in {".png", ".jpg", ".jpeg"}:
            npa = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(npa, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            return [img]
        tmp = _ROOT / "uploads" / f"{_uuid.uuid4()}{ext}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(file_bytes)
        try:
            import SimpleITK as sitk
            vol  = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp))).astype(np.float32)
            n    = vol.shape[0]; lo, hi = int(n*0.10), int(n*0.90)
            step = max(1, (hi - lo) // 8)
            slices = [vol[i] for i in range(lo, hi, step)][:8]
            if len(slices) < 2:
                slices = [vol[i] for i in np.linspace(lo, hi-1, 8, dtype=int)]
            return slices
        except Exception as e:
            logger.warning(f"SimpleITK failed ({e}), trying CV2")
            npa = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(npa, cv2.IMREAD_GRAYSCALE)
            return [img.astype(np.float32)] if img is not None else [np.zeros((512,512), np.float32)]
        finally:
            tmp.unlink(missing_ok=True)

    def _make_overlay(self, img_r: np.ndarray, pred: np.ndarray) -> str:
        import base64
        img_u8  = (np.clip(img_r, 0, 1) * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
        mask_rgb = np.zeros_like(img_rgb)
        for c, col in COLORS.items():
            mask_rgb[pred == c] = col
        overlay = cv2.addWeighted(img_rgb, 0.55, mask_rgb, 0.45, 0)
        _, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf.tobytes()).decode()

    def _make_gradcam(self, img_r: np.ndarray, pred: np.ndarray) -> str:
        """Simple activation-based attention map (no grad needed for inference-only model)."""
        import base64
        try:
            img_u8  = (np.clip(img_r, 0, 1) * 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
            # Use IVD class presence as attention proxy
            attn = np.zeros_like(img_r, dtype=np.float32)
            for c in IVD_CLASSES + VERT_CLASSES:
                attn += (pred == c).astype(np.float32)
            if attn.max() > 0:
                attn = (attn / attn.max())
            attn_blur = cv2.GaussianBlur(attn, (21, 21), 0)
            heat_u8   = (attn_blur * 255).astype(np.uint8)
            heat_bgr  = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            heat_rgb  = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
            alpha     = attn_blur[..., np.newaxis] * 0.65
            result    = (img_rgb * (1 - alpha) + heat_rgb * alpha).astype(np.uint8)
            _, buf = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buf.tobytes()).decode()
        except Exception:
            return ""

    def _recommendation(self, disease: str, severity: str) -> str:
        recs = {
            "Normal":            "Continue routine monitoring.",
            "Disc Herniation":   "Conservative management. Physical therapy. Consider epidural if severe.",
            "Disc Bulge":        "Physical therapy, NSAIDs. Follow-up in 3 months.",
            "Spinal Stenosis":   "Decompression exercises. Surgical consult if neurological symptoms.",
            "Disc Degeneration": "Core strengthening, weight management, pain management.",
            "Spondylolisthesis": "Physiotherapy, bracing. Surgical evaluation if grade ≥ II.",
            "Compression Fracture": "Orthopaedic evaluation. Bone density scan. Vertebroplasty if indicated.",
        }
        return recs.get(disease, "Clinical correlation recommended.")

    def _build_report(self, disease, severity, conf, pfirrmann,
                      levels, ivd_grades, pat, notes_text) -> str:
        from datetime import datetime
        name = pat.get("name", "—"); age = pat.get("age", "—"); sex = pat.get("sex","—")
        lines = [
            "ATM-Net++ SPINE MRI ANALYSIS REPORT",
            "=" * 42,
            f"Date    : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Patient : {name}  |  Age: {age}  |  Sex: {sex}",
            "",
            "PRIMARY DIAGNOSIS",
            "-" * 20,
            f"  Diagnosis   : {disease}",
            f"  Severity    : {severity}",
            f"  Confidence  : {conf:.1f}%",
            f"  Pfirrmann   : {pfirrmann}/5",
        ]
        if levels:
            lines += ["", "AFFECTED LEVELS", "-" * 20]
            for lv in levels: lines.append(f"  • {lv}")
        if ivd_grades:
            lines += ["", "PER-DISC GRADING", "-" * 20]
            for lv, g in ivd_grades.items():
                lines.append(f"  {lv:<10} Grade {g['grade']}/5  {g['status']}")
        if notes_text.strip():
            lines += ["", "CLINICAL NOTES", "-" * 20, f"  {notes_text[:300]}"]
        lines += [
            "", "DISCLAIMER",
            "-" * 20,
            "  AI-assisted analysis. Not for standalone clinical use.",
            "  Requires radiologist review before clinical decisions.",
        ]
        return "\n".join(lines)
