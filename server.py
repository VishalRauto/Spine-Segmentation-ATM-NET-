import sys, os, io, json, base64, time, uuid, threading, warnings, math
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
import numpy as np
import cv2

BASE       = Path(__file__).parent
CKPT_BEST  = BASE / "outputs/gpu_run/best_model.pth"
CKPT_LAST  = BASE / "outputs/gpu_run/last_model.pth"
CKPT_CPU   = BASE / "outputs/high_perf_run/best_model.pth"
HIST_GPU   = BASE / "outputs/gpu_run/history.json"
HIST_CPU   = BASE / "outputs/high_perf_run/history.json"
UPLOAD_DIR = BASE / "outputs/uploads"
HISTORY_DB = BASE / "outputs/patient_history.json"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Checkpoint selection ───────────────────────────────────────────────
def _pick_ckpt():
    import torch
    candidates = [
        (BASE / "outputs/gpu_run/kaggle_converted.pth", 256),
        (BASE / "outputs/gpu_run/last_model.pth",       192),
        (BASE / "outputs/gpu_run/best_model.pth",       192),
        (CKPT_CPU,                                      192),
    ]
    best_path, best_dice, best_size = None, -1.0, 192
    for p, sz in candidates:
        if not p.exists(): continue
        try:
            c    = torch.load(str(p), map_location="cpu")
            keys = list(c.get("model_state_dict", {}).keys())
            if not (any(k == "e1.conv.0.weight" for k in keys) and
                    any("sa.conv.0.weight" in k  for k in keys)):
                print(f"  [ckpt] Skipping {p.name} — incompatible"); continue
            ep, dice = c.get("epoch", 0), c.get("best_dice", 0.0)
            print(f"  [ckpt] {p.name}: ep={ep} dice={dice:.4f} sz={sz} ✓")
            if dice > best_dice:
                best_dice, best_path, best_size = dice, p, sz
        except Exception as e:
            print(f"  [ckpt] {p.name}: error — {e}")
    if best_path:
        print(f"  [ckpt] Using: {best_path.name} (dice={best_dice:.4f}, size={best_size})")
    else:
        print("  [ckpt] WARNING: No compatible checkpoint found.")
    return best_path, best_size

CKPT, INFER_SIZE = _pick_ckpt()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

_model = None
_model_lock = threading.Lock()
_device = None

SPIDER_TO_ATMNET = {**{i: i for i in range(1, 9)}, 100: 9,
                    **{201+i: 10+i for i in range(8)}}
CLASS_NAMES = {
    0:"Background",  1:"Vertebra-1(L5)", 2:"Vertebra-2(L4)", 3:"Vertebra-3(L3)",
    4:"Vertebra-4(L2)", 5:"Vertebra-5(L1)", 6:"Vertebra-6(T12)", 7:"Vertebra-7(T11)",
    8:"Vertebra-8(T10)", 9:"Sacrum",
    10:"IVD L5/S1", 11:"IVD L4/L5", 12:"IVD L3/L4", 13:"IVD L2/L3",
    14:"IVD L1/L2", 15:"IVD T12/L1", 16:"IVD T11/T12", 17:"IVD T10/T11",
    18:"Spinal Canal",
}
CLASS_SHORT = {
    0:"BG", 1:"L5", 2:"L4", 3:"L3", 4:"L2", 5:"L1",
    6:"T12", 7:"T11", 8:"T10", 9:"Sac",
    10:"L5/S1", 11:"L4/L5", 12:"L3/L4", 13:"L2/L3",
    14:"L1/L2", 15:"T12/L1", 16:"T11/T12", 17:"T10/T11", 18:"Canal",
}
COLORS = {
    0:(0,0,0),       1:(220,50,50),   2:(255,100,50),  3:(255,180,50),
    4:(200,230,50),  5:(80,220,80),   6:(50,210,150),  7:(50,180,230),
    8:(50,120,255),  9:(80,80,255),   10:(160,60,255), 11:(240,60,200),
    12:(255,60,120), 13:(200,120,255),14:(100,200,255), 15:(255,160,60),
    16:(60,240,160), 17:(140,240,80), 18:(220,220,220),
}
NUM_CLASSES  = 19
VERT_CLASSES = list(range(1, 9))
IVD_CLASSES  = list(range(10, 18))
IVD_LABELS   = ["L5/S1","L4/L5","L3/L4","L2/L3","L1/L2","T12/L1","T11/T12","T10/T11"]

# ── Model definition ──────────────────────────────────────────────────
def remap(m):
    out = np.zeros_like(m, dtype=np.int64)
    for s, d in SPIDER_TO_ATMNET.items(): out[m == s] = d
    return out

def get_model():
    global _model, _device
    with _model_lock:
        if _model is not None: return _model, _device
        import torch, torch.nn as nn
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class CA(nn.Module):
            def __init__(self, ch, r=8):
                super().__init__(); r = max(1, ch // r)
                self.avg = nn.AdaptiveAvgPool2d(1); self.max = nn.AdaptiveMaxPool2d(1)
                self.fc  = nn.Sequential(nn.Flatten(), nn.Linear(ch, r), nn.ReLU(True),
                                         nn.Linear(r, ch), nn.Sigmoid())
            def forward(self, x):
                a = self.fc(self.avg(x)) + self.fc(self.max(x))
                return x * a.clamp(0, 1).view(x.shape[0], -1, 1, 1)

        class SA(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False),
                                          nn.BatchNorm2d(1), nn.Sigmoid())
            def forward(self, x):
                return x * self.conv(torch.cat([x.mean(1, keepdim=True),
                                                x.max(1, keepdim=True)[0]], 1))

        class RB(nn.Module):
            def __init__(self, ch):
                super().__init__()
                self.net = nn.Sequential(nn.Conv2d(ch,ch,3,1,1,bias=False),nn.BatchNorm2d(ch),nn.ReLU(True),
                                         nn.Conv2d(ch,ch,3,1,1,bias=False),nn.BatchNorm2d(ch))
                self.ca = CA(ch); self.sa = SA(); self.act = nn.ReLU(True)
            def forward(self, x): return self.act(self.sa(self.ca(self.net(x))) + x)

        class Enc(nn.Module):
            def __init__(self, ci, co, drop=0.0):
                super().__init__()
                self.conv = nn.Sequential(nn.Conv2d(ci,co,3,1,1,bias=False),nn.BatchNorm2d(co),nn.ReLU(True),
                                          nn.Conv2d(co,co,3,1,1,bias=False),nn.BatchNorm2d(co),nn.ReLU(True))
                self.res  = RB(co)
                self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
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
                self.ds3=nn.Conv2d(b*4,nc,1); self.ds2=nn.Conv2d(b*2,nc,1); self.out=nn.Conv2d(b,nc,1)
                self.aux=nn.Sequential(nn.Conv2d(b,b,3,1,1,bias=False),nn.BatchNorm2d(b),
                                       nn.ReLU(True),nn.Conv2d(b,nc,1))
            def forward(self, x):
                e1=self.e1(x); e2=self.e2(self.pool(e1))
                e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
                d=self.bn(self.pool(e4))
                d=self.d4(torch.cat([self.u4(d),e4],1))
                d=self.d3(torch.cat([self.u3(d),e3],1))
                d=self.d2(torch.cat([self.u2(d),e2],1))
                d=self.d1(torch.cat([self.u1(d),e1],1))
                return self.out(d)

        _model = ResUNet(b=32, nc=NUM_CLASSES, drop=0.25).to(_device)
        if CKPT and CKPT.exists():
            import torch as _t
            ckpt = _t.load(str(CKPT), map_location=_device)
            missing, _ = _model.load_state_dict(ckpt["model_state_dict"], strict=False)
            ep, dice   = ckpt.get("epoch","?"), ckpt.get("best_dice", 0.0)
            print(f"Loaded: epoch={ep} dice={dice:.4f} missing={len(missing)}")
        else:
            print("WARNING: No checkpoint — random weights")
        _model.eval()
        return _model, _device

# ── Image helpers ─────────────────────────────────────────────────────
def arr_to_b64(arr):
    if arr.ndim == 2:
        if arr.max() <= 1: arr = (arr * 255).astype(np.uint8)
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    _, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()

def colorize(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in COLORS.items(): rgb[mask == c] = col
    return rgb

def make_legend():
    """Render a color legend image (200×400 px)."""
    h, w = 20, 200
    canvas = np.zeros((NUM_CLASSES * h, w, 3), dtype=np.uint8)
    canvas[:] = (18, 20, 35)
    for c in range(NUM_CLASSES):
        y = c * h
        col = COLORS[c]
        canvas[y:y+h, 0:28] = col
        label = CLASS_SHORT.get(c, str(c))
        cv2.putText(canvas, label, (32, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 210, 220), 1, cv2.LINE_AA)
    return canvas

def add_annotations(img_rgb, pred, prob_arr):
    """Draw centroid dots and labels on overlay image — spaced to avoid overlap."""
    out   = img_rgb.copy()
    H, W  = pred.shape
    used_y = []  # track y positions to avoid label overlap

    for c in range(1, NUM_CLASSES):
        mask = (pred == c).astype(np.uint8)
        if mask.sum() < 80: continue
        ys, xs  = np.where(mask)
        cy, cx  = int(ys.mean()), int(xs.mean())
        col     = tuple(int(v) for v in COLORS[c])
        # Dot with white border
        cv2.circle(out, (cx, cy), 5, (255,255,255), -1)
        cv2.circle(out, (cx, cy), 4, col, -1)
        # Shift label if too close to a previous one
        ly = cy
        for prev_y in used_y:
            if abs(ly - prev_y) < 13:
                ly = prev_y + 13
        used_y.append(ly)
        label = CLASS_SHORT.get(c, "")
        lx    = min(cx + 7, W - 40)
        # Shadow + text for readability
        cv2.putText(out, label, (lx+1, ly+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(out, label, (lx, ly+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 1, cv2.LINE_AA)
    return out

# ── Feature: Scoliosis / curvature analysis ───────────────────────────
def compute_curvature(pred, prob_arr):
    """
    Estimate Cobb-like angle from vertebra centroid positions.
    Returns angle in degrees and scoliosis risk level.
    """
    centroids = {}
    for c in VERT_CLASSES:
        mask = (pred == c).astype(np.uint8)
        if mask.sum() < 30: continue
        ys, xs = np.where(mask)
        centroids[c] = (float(xs.mean()), float(ys.mean()))  # (x, y)

    if len(centroids) < 3:
        return {"angle": None, "risk": "Insufficient data", "centroids": {}}

    # Fit a line through centroids and compute deviation
    pts = sorted(centroids.items(), key=lambda x: x[1][1])  # sort by y (top→bottom)
    xs_c = np.array([p[1][0] for p in pts])
    ys_c = np.array([p[1][1] for p in pts])

    # Linear regression
    if len(xs_c) >= 2:
        coeffs  = np.polyfit(ys_c, xs_c, 1)
        xs_fit  = np.polyval(coeffs, ys_c)
        residuals = xs_c - xs_fit
        deviation = float(np.std(residuals))

        # Approximate Cobb angle from slope
        slope_deg = float(math.degrees(math.atan(abs(coeffs[0]))))
    else:
        deviation, slope_deg = 0.0, 0.0

    # Risk classification
    if slope_deg < 5:   risk = "Normal"
    elif slope_deg < 10: risk = "Mild Scoliosis"
    elif slope_deg < 20: risk = "Moderate Scoliosis"
    else:               risk = "Severe Scoliosis"

    return {
        "angle"    : round(slope_deg, 1),
        "deviation": round(deviation, 1),
        "risk"     : risk,
        "centroids": {str(c): [round(v[0], 1), round(v[1], 1)]
                      for c, v in centroids.items()},
    }

# ── Feature: Per-IVD Pfirrmann grading ───────────────────────────────
def compute_pfirrmann(prob_arr):
    """
    Grade each IVD individually on Pfirrmann 1–5 scale.
    Uses IVD channel confidence as proxy for disc health.
    Higher confidence = brighter disc = healthier (lower grade).
    """
    grades = {}
    for i, ivd_cls in enumerate(IVD_CLASSES):
        conf = float(prob_arr[ivd_cls].max())
        px   = int((prob_arr[ivd_cls] > 0.1).sum())
        if px < 20:
            grades[IVD_LABELS[i]] = {"grade": None, "confidence": 0.0, "status": "Not visible"}
            continue
        # Map confidence → Pfirrmann grade (inverse: low conf = degenerated)
        if   conf > 0.80: grade, status = 1, "Normal"
        elif conf > 0.60: grade, status = 2, "Normal with changes"
        elif conf > 0.40: grade, status = 3, "Early degeneration"
        elif conf > 0.20: grade, status = 4, "Moderate degeneration"
        else:             grade, status = 5, "Severe degeneration"
        grades[IVD_LABELS[i]] = {
            "grade"     : grade,
            "confidence": round(conf, 3),
            "status"    : status,
        }
    return grades

# ── Feature: Spine height ratio ───────────────────────────────────────
def compute_disc_heights(pred, size):
    """Estimate relative IVD height vs adjacent vertebra height (compression indicator)."""
    heights = {}
    for i, ivd_cls in enumerate(IVD_CLASSES):
        ivd_mask = (pred == ivd_cls)
        if ivd_mask.sum() < 20: continue
        ys = np.where(ivd_mask)[0]
        ivd_h = int(ys.max() - ys.min() + 1)
        heights[IVD_LABELS[i]] = {
            "height_px"  : ivd_h,
            "height_pct" : round(ivd_h / size * 100, 1),
            "compressed" : ivd_h < (size * 0.025),  # <2.5% of image = compressed
        }
    return heights

# ── Feature: Patient history ──────────────────────────────────────────
def load_history():
    if HISTORY_DB.exists():
        try: return json.loads(HISTORY_DB.read_text())
        except: pass
    return []

def save_history(records):
    HISTORY_DB.write_text(json.dumps(records, indent=2))

def add_to_history(record):
    hist = load_history()
    record["id"] = str(uuid.uuid4())[:8]
    hist.insert(0, record)
    hist = hist[:50]  # keep last 50
    save_history(hist)
    return record["id"]

# ── Feature: Grad-CAM explainability ─────────────────────────────────
def compute_gradcam(model, device, img_r, target_class=None):
    """
    Generate Grad-CAM heatmap for the prediction.
    Hooks into the last decoder block (d1) of ResUNet.
    Returns (H,W) float32 heatmap in [0,1].
    """
    import torch, torch.nn.functional as F
    activations, gradients = {}, {}

    def fwd_hook(m, inp, out):  activations['d1'] = out.detach()
    def bwd_hook(m, gi, go):    gradients['d1']   = go[0].detach()

    handle_f = model.d1.register_forward_hook(fwd_hook)
    handle_b = model.d1.register_full_backward_hook(bwd_hook)

    t = torch.from_numpy(img_r[None, None]).float().to(device).requires_grad_(True)
    model.eval()
    out = model(t)                          # (1, NC, H, W)

    if target_class is None:
        # Use the highest-confidence foreground class
        fg_scores = out[0, 1:].mean(dim=(1, 2))  # mean per fg class
        target_class = int(fg_scores.argmax().item()) + 1

    score = out[0, target_class].mean()
    model.zero_grad()
    score.backward()

    handle_f.remove(); handle_b.remove()

    if 'd1' not in activations or 'd1' not in gradients:
        return np.zeros((img_r.shape[0], img_r.shape[1]), dtype=np.float32)

    act = activations['d1'].squeeze(0).cpu().numpy()    # (C, H, W)
    grd = gradients['d1'].squeeze(0).cpu().numpy()      # (C, H, W)
    weights = grd.mean(axis=(1, 2))                     # (C,)
    cam = (weights[:, None, None] * act).sum(0)         # (H, W)
    cam = np.maximum(cam, 0)

    # Upsample to input size
    H, W = img_r.shape
    cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    return cam.astype(np.float32)

def render_gradcam_overlay(img_r, cam):
    """Apply JET colormap heatmap over grayscale MRI."""
    img_u8    = (np.clip(img_r, 0, 1) * 255).astype(np.uint8)
    img_rgb   = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    heat_u8   = (cam * 255).astype(np.uint8)
    heat_bgr  = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_rgb  = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 0.55, heat_rgb, 0.45, 0)

# ── Feature: MC Dropout uncertainty ──────────────────────────────────
def compute_uncertainty(model, device, img_r, n_samples=10):
    """
    Monte Carlo Dropout — run inference N times with dropout ON.
    Returns per-pixel entropy as uncertainty map (H, W) float32 [0,1].
    """
    import torch, torch.nn.functional as F
    t = torch.from_numpy(img_r[None, None]).float().to(device)

    # Enable dropout at inference by setting train mode
    model.train()
    probs_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            p = F.softmax(model(t), 1).squeeze(0).cpu().numpy()
            probs_list.append(p)
    model.eval()

    probs_stack = np.stack(probs_list, axis=0)          # (N, C, H, W)
    mean_probs  = probs_stack.mean(0)                    # (C, H, W)
    # Entropy: −Σ p·log(p)
    entropy = -(mean_probs * np.log(mean_probs + 1e-8)).sum(0)  # (H, W)
    # Normalize to [0,1]
    e_min, e_max = entropy.min(), entropy.max()
    if e_max > e_min:
        entropy = (entropy - e_min) / (e_max - e_min)
    return entropy.astype(np.float32), mean_probs

def render_uncertainty_overlay(img_r, uncertainty):
    """Blue=certain, Red=uncertain overlay."""
    img_u8   = (np.clip(img_r, 0, 1) * 255).astype(np.uint8)
    img_rgb  = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    unc_u8   = (uncertainty * 255).astype(np.uint8)
    unc_bgr  = cv2.applyColorMap(unc_u8, cv2.COLORMAP_HOT)
    unc_rgb  = cv2.cvtColor(unc_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 0.6, unc_rgb, 0.4, 0)

# ── Feature: Canal stenosis detection ────────────────────────────────
def compute_stenosis(pred, size):
    """
    Measure spinal canal width at each vertebra level.
    Narrow canal at a level with IVD → stenosis suspected.
    """
    CANAL_CLASS = 18
    canal_mask  = (pred == CANAL_CLASS)
    if canal_mask.sum() < 20:
        return {"detected": False, "width_pct": None, "levels": {}, "risk": "Canal not visible"}

    results = {}
    for i, ivd_cls in enumerate(IVD_CLASSES):
        ivd_mask = (pred == ivd_cls)
        if ivd_mask.sum() < 10: continue
        # Get y-range of IVD
        ys_ivd = np.where(ivd_mask)[0]
        y_mid  = int(ys_ivd.mean())
        # Measure canal width at that y level ±3px
        y_lo, y_hi = max(0, y_mid-3), min(size, y_mid+3)
        canal_row   = canal_mask[y_lo:y_hi, :].any(axis=0)
        xs_canal    = np.where(canal_row)[0]
        if len(xs_canal) < 2:
            continue
        width_px  = int(xs_canal.max() - xs_canal.min() + 1)
        width_pct = round(width_px / size * 100, 1)
        stenosis  = width_pct < 5.0   # <5% of image width = stenotic
        results[IVD_LABELS[i]] = {
            "width_px" : width_px,
            "width_pct": width_pct,
            "stenosis" : stenosis,
        }

    any_stenosis = any(v["stenosis"] for v in results.values())
    overall_risk = "Stenosis suspected" if any_stenosis else "Normal"
    return {"detected": True, "levels": results, "risk": overall_risk}

# ── Feature: Lordosis/Kyphosis (spinal curve type) ───────────────────
def compute_lordosis(pred):
    """
    Fit degree-2 polynomial to vertebra centroids.
    Concavity direction tells lordosis vs kyphosis.
    Returns curve type and curvature magnitude.
    """
    centroids_y, centroids_x = [], []
    for c in VERT_CLASSES:
        m = (pred == c)
        if m.sum() < 30: continue
        ys, xs = np.where(m)
        centroids_y.append(float(ys.mean()))
        centroids_x.append(float(xs.mean()))

    if len(centroids_y) < 4:
        return {"type": "Insufficient data", "curvature": None}

    # Fit quadratic: x = a*y^2 + b*y + c
    coeffs = np.polyfit(centroids_y, centroids_x, 2)
    a = coeffs[0]  # concavity

    magnitude = round(abs(a) * 1000, 2)
    if   abs(a) < 0.0005: curve_type = "Straight (loss of lordosis)"
    elif a > 0:           curve_type = "Lordosis (normal lumbar curve)"
    else:                 curve_type = "Kyphosis (reversed curve)"

    return {"type": curve_type, "curvature": magnitude, "coeff_a": round(a, 6)}

# ── Feature: T2 signal intensity (IVD hydration proxy) ───────────────
def compute_t2_signal(img_r, pred):
    """
    Measure mean T2 signal inside each IVD mask.
    Bright IVD = well-hydrated = healthy.
    Dark IVD   = dehydrated    = degenerated.
    Returns normalized signal 0–100 per disc.
    """
    signals = {}
    for i, ivd_cls in enumerate(IVD_CLASSES):
        mask = (pred == ivd_cls)
        if mask.sum() < 20: continue
        mean_signal = float(img_r[mask].mean())    # 0–1 normalized
        pct         = round(mean_signal * 100, 1)
        if   pct > 60: hydration = "Well-hydrated (healthy)"
        elif pct > 40: hydration = "Moderate hydration"
        elif pct > 25: hydration = "Dehydrated (early degen.)"
        else:          hydration = "Severely dehydrated"
        signals[IVD_LABELS[i]] = {
            "signal"   : pct,
            "hydration": hydration,
            "dark"     : pct < 30,
        }
    return signals

# ── Feature: Vertebral compression / fracture risk ───────────────────
def compute_fracture_risk(pred, size):
    """
    Compare vertebra height at each level.
    Anterior:Posterior height ratio < 0.8 = compression fracture risk.
    """
    risks = {}
    for c in VERT_CLASSES:
        mask = (pred == c)
        if mask.sum() < 50: continue
        ys, xs = np.where(mask)
        total_h = int(ys.max() - ys.min() + 1)
        # Anterior = left 30% of mask columns, Posterior = right 30%
        x_min, x_max = xs.min(), xs.max()
        x_range = max(x_max - x_min, 1)
        ant_mask = mask[:, :int(x_min + x_range*0.3)]
        pos_mask = mask[:, int(x_min + x_range*0.7):]
        ant_h = int(np.where(ant_mask)[0].ptp() + 1) if ant_mask.sum() > 5 else total_h
        pos_h = int(np.where(pos_mask)[0].ptp() + 1) if pos_mask.sum() > 5 else total_h
        ratio = round(ant_h / max(pos_h, 1), 2)
        fractured = ratio < 0.80
        name = CLASS_SHORT.get(c, str(c))
        risks[name] = {
            "height_px": total_h,
            "ap_ratio" : ratio,
            "risk"     : "Compression risk" if fractured else "Normal",
        }
    return risks

# ── Feature: PDF report generation ────────────────────────────────────
def generate_pdf(result_data, patient_info):
    """
    Generate a PDF clinical report with images and findings.
    Returns bytes of the PDF.
    """
    try:
        from fpdf import FPDF
        import tempfile, os

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # ── Header ──
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(30, 100, 180)
        pdf.cell(0, 10, "ATM-Net++ Spine MRI Analysis Report", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, f"Generated: {time.strftime('%Y-%m-%d %H:%M')}  |  AI-assisted — Not for clinical use", ln=True)
        pdf.ln(3)

        # ── Patient info ──
        pdf.set_draw_color(50, 130, 200)
        pdf.set_fill_color(240, 247, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(20, 60, 120)
        pdf.cell(0, 7, "Patient Information", ln=True, fill=True, border="B")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pat = patient_info or {}
        info_pairs = [
            ("Name", pat.get("name") or "—"),
            ("Age",  pat.get("age")  or "—"),
            ("Sex",  pat.get("sex")  or "—"),
        ]
        for label, val in info_pairs:
            pdf.cell(40, 6, f"{label}:", border=0)
            pdf.cell(0, 6, str(val), ln=True)
        pdf.ln(3)

        # ── Primary diagnosis ──
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(20, 60, 120)
        pdf.cell(0, 7, "Primary Diagnosis", ln=True, fill=True, border="B")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        diag_items = [
            ("Diagnosis",   result_data.get("disease",  "—")),
            ("Severity",    result_data.get("severity", "—")),
            ("Confidence",  f"{result_data.get('confidence', 0)}%"),
            ("Pfirrmann",   f"{result_data.get('pfirrmann_grade', '—')}/5"),
        ]
        for label, val in diag_items:
            pdf.cell(50, 6, f"{label}:", border=0)
            pdf.cell(0, 6, str(val), ln=True)
        pdf.ln(3)

        # ── Scoliosis ──
        cv = result_data.get("curvature", {})
        if cv.get("angle") is not None:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(20, 60, 120)
            pdf.cell(0, 7, "Scoliosis Assessment", ln=True, fill=True, border="B")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
            pdf.cell(50, 6, "Cobb Angle:"); pdf.cell(0, 6, f"{cv['angle']}°", ln=True)
            pdf.cell(50, 6, "Risk Level:"); pdf.cell(0, 6, cv.get("risk","—"), ln=True)
            lk = result_data.get("lordosis", {})
            if lk.get("type"):
                pdf.cell(50, 6, "Curve Type:"); pdf.cell(0, 6, lk["type"], ln=True)
            pdf.ln(3)

        # ── Per-disc Pfirrmann ──
        grades = result_data.get("ivd_grades", {})
        if grades:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(20, 60, 120)
            pdf.cell(0, 7, "Per-Disc Pfirrmann Grading", ln=True, fill=True, border="B")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
            for disc, g in grades.items():
                if g.get("grade") is None: continue
                pdf.cell(35, 6, disc)
                pdf.cell(25, 6, f"Grade {g['grade']}/5")
                pdf.cell(0,  6, g.get("status",""), ln=True)
            pdf.ln(3)

        # ── T2 signal ──
        t2 = result_data.get("t2_signal", {})
        if t2:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(20, 60, 120)
            pdf.cell(0, 7, "IVD T2 Signal Intensity", ln=True, fill=True, border="B")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
            for disc, s in t2.items():
                pdf.cell(35, 6, disc)
                pdf.cell(20, 6, f"{s['signal']}%")
                pdf.cell(0,  6, s.get("hydration",""), ln=True)
            pdf.ln(3)

        # ── Stenosis ──
        sten = result_data.get("stenosis", {})
        if sten.get("detected"):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(20, 60, 120)
            pdf.cell(0, 7, "Canal Stenosis Assessment", ln=True, fill=True, border="B")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
            pdf.cell(0, 6, f"Overall: {sten.get('risk','—')}", ln=True)
            for level, sv in sten.get("levels", {}).items():
                flag = " *** STENOSIS ***" if sv.get("stenosis") else ""
                pdf.cell(35, 6, level)
                pdf.cell(30, 6, f"{sv.get('width_pct','—')}% width")
                pdf.cell(0,  6, flag, ln=True)
            pdf.ln(3)

        # ── Images ──
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(20, 60, 120)
        pdf.cell(0, 7, "Imaging Results", ln=True, fill=True, border="B")
        pdf.ln(2)

        tmp_files = []
        img_pairs = [
            ("MRI Input",           result_data.get("image_b64")),
            ("Segmentation Overlay",result_data.get("overlay_b64")),
            ("Segmentation Mask",   result_data.get("mask_b64")),
            ("Scoliosis Analysis",  result_data.get("scoliosis_b64")),
            ("Grad-CAM Heatmap",    result_data.get("gradcam_b64")),
            ("Uncertainty Map",     result_data.get("uncertainty_b64")),
        ]
        col, x_positions = 0, [10, 110]
        for label, b64 in img_pairs:
            if not b64: continue
            try:
                img_bytes = base64.b64decode(b64)
                tmp_f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp_f.write(img_bytes); tmp_f.close()
                tmp_files.append(tmp_f.name)
                x = x_positions[col]
                pdf.set_xy(x, pdf.get_y())
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(80, 80, 80)
                pdf.cell(90, 5, label, ln=(col==1))
                pdf.set_xy(x, pdf.get_y())
                pdf.image(tmp_f.name, x=x, w=90)
                col = 1 - col
                if col == 0:
                    pdf.ln(3)
            except Exception as e:
                pass

        # ── Footer disclaimer ──
        pdf.ln(5)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 50, 50)
        pdf.multi_cell(0, 4,
            "DISCLAIMER: This report is AI-generated for research purposes only. "
            "It must be reviewed and validated by a qualified radiologist before any clinical decision-making.")

        # Output
        pdf_bytes = pdf.output(dest='S').encode('latin-1') if isinstance(pdf.output(dest='S'), str) else pdf.output(dest='S')

        # Cleanup temp files
        for f in tmp_files:
            try: os.unlink(f)
            except: pass

        return pdf_bytes

    except Exception as e:
        # Fallback: return plain text as bytes
        return result_data.get("report", f"PDF generation failed: {e}").encode()


# ── Core prediction pipeline ──────────────────────────────────────────
def run_pred(file_bytes, filename, patient_info=None):
    import torch, torch.nn.functional as F
    model, device = get_model()
    size = INFER_SIZE
    ext  = "".join(Path(filename).suffixes).lower()

    if ext in {".png", ".jpg", ".jpeg"}:
        npa  = np.frombuffer(file_bytes, np.uint8)
        img  = cv2.imdecode(npa, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        slices = [img]
    else:
        tmp = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
        tmp.write_bytes(file_bytes)
        try:
            import SimpleITK as sitk
            vol = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp))).astype(np.float32)
            n   = vol.shape[0]; lo, hi = int(n * 0.10), int(n * 0.90)
            step = max(1, (hi - lo) // 8)
            slices = [vol[i] for i in range(lo, hi, step)][:8]
            if len(slices) < 2:   # fallback: evenly spaced
                slices = [vol[i] for i in np.linspace(lo, hi-1, 8, dtype=int)]
        finally:
            tmp.unlink(missing_ok=True)

    # Run inference on all slices
    all_preds = []
    for sl in slices:
        p1, p99 = np.percentile(sl, [0.5, 99.5])
        img_n = np.clip((sl - p1) / (p99 - p1 + 1e-8), 0, 1).astype(np.float32)
        img_r = cv2.resize(img_n, (size, size), interpolation=cv2.INTER_LINEAR)
        t     = torch.from_numpy(img_r[None, None]).float().to(device)
        with torch.no_grad():
            pr  = F.softmax(model(t), 1)
            pr2 = F.softmax(model(torch.flip(t, [-1])), 1)
            avg = ((pr + torch.flip(pr2, [-1])) / 2).squeeze(0).cpu().numpy()

        # Background-bias correction
        bg_prob = avg[0]; fg_max = avg[1:].max(0)
        bg_bias = np.clip(bg_prob - fg_max - 0.02, 0, None)
        avg[0]  = bg_prob - bg_bias
        avg     = avg / (avg.sum(0, keepdims=True) + 1e-8)
        pred    = avg.argmax(0).astype(np.int32)
        all_preds.append((img_r, pred, avg))

    # Use mid slice for display, aggregate class votes across all slices
    mid = len(all_preds) // 2
    img_pre, pred, prob_arr = all_preds[mid]

    # ── Class distribution ────────────────────────────────────────────
    cls_dist = {}; detected = []
    for c in range(1, NUM_CLASSES):
        px = int((pred == c).sum())
        if px > 30:
            nm  = CLASS_NAMES[c]
            pct = round(px / pred.size * 100, 2)
            cls_dist[nm] = {"pixels": px, "percent": pct}
            detected.append(nm)

    # ── NEW: aggregate confidence across all slices for better IVD grading ─
    avg_probs = np.mean([p[2] for p in all_preds], axis=0)

    # ── Disease prediction ────────────────────────────────────────────
    ivd_conf = [float(avg_probs[c].max()) for c in IVD_CLASSES]
    mean_ivd  = float(np.mean(ivd_conf)) if ivd_conf else 0
    if   mean_ivd > 0.75: disease, severity, conf = "Normal",             "None",     round(mean_ivd * 100, 1)
    elif mean_ivd > 0.50: disease, severity, conf = "Disc Bulge",         "Mild",     round(mean_ivd * 100, 1)
    elif mean_ivd > 0.30: disease, severity, conf = "Disc Degeneration",  "Moderate", round(mean_ivd * 100, 1)
    else:                 disease, severity, conf = "Disc Herniation",    "Severe",   round((1 - mean_ivd) * 100, 1)
    pfirrmann_overall = round(5 - mean_ivd * 4, 1)

    # ── NEW FEATURES ──────────────────────────────────────────────────
    # 1. Per-IVD Pfirrmann grading
    ivd_grades = compute_pfirrmann(avg_probs)

    # 2. Scoliosis / curvature analysis
    curvature = compute_curvature(pred, prob_arr)

    # 3. Disc height analysis
    disc_heights = compute_disc_heights(pred, size)

    # 4. NEW: Canal stenosis
    stenosis = compute_stenosis(pred, size)

    # 5. NEW: Lordosis/Kyphosis
    lordosis = compute_lordosis(pred)

    # 6. NEW: T2 signal intensity
    t2_signal = compute_t2_signal(img_pre, pred)

    # 7. NEW: Vertebral fracture risk
    fracture_risk = compute_fracture_risk(pred, size)

    # 8. NEW: Grad-CAM (on mid slice, most confident fg class)
    try:
        cam = compute_gradcam(model, device, img_pre)
        gradcam_img = render_gradcam_overlay(img_pre, cam)
        gradcam_b64 = arr_to_b64(gradcam_img)
    except Exception as _e:
        gradcam_b64 = arr_to_b64((np.clip(img_pre,0,1)*255).astype(np.uint8))

    # 9. NEW: MC Dropout uncertainty (fast: 8 samples)
    try:
        uncertainty_map, _ = compute_uncertainty(model, device, img_pre, n_samples=8)
        uncertainty_img    = render_uncertainty_overlay(img_pre, uncertainty_map)
        uncertainty_b64    = arr_to_b64(uncertainty_img)
        uncertainty_mean   = round(float(uncertainty_map.mean()), 4)
    except Exception as _e:
        uncertainty_b64  = gradcam_b64
        uncertainty_mean = 0.0

    # 4. Multi-slice thumbnails (base64 of all slices, each independently rendered)
    slice_thumbs = []
    for img_sl, pred_sl, _ in all_preds:
        img_u8_   = (np.clip(img_sl, 0, 1) * 255).astype(np.uint8)
        img_rgb_  = cv2.cvtColor(img_u8_, cv2.COLOR_GRAY2RGB)
        msk_rgb_  = colorize(pred_sl)
        fg_       = (pred_sl > 0).astype(np.float32)[..., np.newaxis]
        blend_    = (img_rgb_ * (1 - 0.55 * fg_) + msk_rgb_ * 0.55 * fg_).astype(np.uint8)
        # Resize to 96×96 thumb
        thumb     = cv2.resize(blend_, (96, 96), interpolation=cv2.INTER_LINEAR)
        slice_thumbs.append(arr_to_b64(thumb))

    # 5. Main display images (with annotations)
    img_u8   = (np.clip(img_pre, 0, 1) * 255).astype(np.uint8)
    img_rgb  = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    mask_rgb = colorize(pred)
    fg       = (pred > 0).astype(np.float32)[..., np.newaxis]
    blend    = (img_rgb * (1 - 0.60 * fg) + mask_rgb * 0.60 * fg).astype(np.uint8)
    annotated = add_annotations(blend, pred, prob_arr)

    # 6. Legend image
    legend_img = make_legend()

    # 7. Scoliosis overlay — draw centroid spine line on MRI
    scoliosis_img = img_rgb.copy()
    if curvature.get("centroids"):
        cpts = sorted(curvature["centroids"].items(), key=lambda x: x[1][1])
        # Draw connecting line first (thicker, semi-transparent effect)
        for i in range(len(cpts) - 1):
            p1c = (int(cpts[i][1][0]),   int(cpts[i][1][1]))
            p2c = (int(cpts[i+1][1][0]), int(cpts[i+1][1][1]))
            cv2.line(scoliosis_img, p1c, p2c, (255, 220, 0), 2)
        # Draw dots on top
        for idx, (c_key, cval) in enumerate(cpts):
            pt    = (int(cval[0]), int(cval[1]))
            c_idx = int(c_key)
            color = tuple(int(v) for v in COLORS.get(c_idx, (255, 255, 0)))
            cv2.circle(scoliosis_img, pt, 7, (255,255,255), -1)
            cv2.circle(scoliosis_img, pt, 6, color, -1)
            # Tiny level label beside each dot
            label = CLASS_SHORT.get(c_idx, "")
            cv2.putText(scoliosis_img, label, (pt[0]+9, pt[1]+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(scoliosis_img, label, (pt[0]+8, pt[1]+3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255,255,255), 1, cv2.LINE_AA)
        # Angle badge top-left
        angle_txt = f"~{curvature['angle']}deg"
        risk_col  = (104,211,145) if "Normal" in curvature["risk"] else \
                    (246,173,85)  if "Mild"   in curvature["risk"] else (252,129,129)
        cv2.putText(scoliosis_img, angle_txt, (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(scoliosis_img, angle_txt, (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, risk_col, 1, cv2.LINE_AA)

    # ── Build clinical report ─────────────────────────────────────────
    now  = time.strftime("%Y-%m-%d %H:%M")
    pat  = patient_info or {}
    report  = "LUMBAR SPINE MRI ANALYSIS REPORT\n"
    report += f"Generated by ATM-Net++ v2  |  {now}\n"
    report += "=" * 50 + "\n"
    if pat.get("name"): report += f"Patient     : {pat['name']}\n"
    if pat.get("age"):  report += f"Age / Sex   : {pat.get('age','?')} yrs / {pat.get('sex','?')}\n"
    report += "\nPRIMARY DIAGNOSIS\n" + "-" * 30 + "\n"
    report += f"  Diagnosis   : {disease}\n  Severity    : {severity}\n"
    report += f"  Confidence  : {conf}%\n  Pfirrmann   : {pfirrmann_overall}/5\n"
    report += "\nSCOLIOSIS / CURVATURE\n" + "-" * 30 + "\n"
    report += f"  Cobb Angle  : {curvature.get('angle','N/A')} degrees\n"
    report += f"  Scoliosis   : {curvature.get('risk','N/A')}\n"
    report += f"  Curve Type  : {lordosis.get('type','N/A')}\n"
    report += "\nCANAL STENOSIS\n" + "-" * 30 + "\n"
    report += f"  Overall     : {stenosis.get('risk','N/A')}\n"
    for lvl, sv in stenosis.get("levels",{}).items():
        flag = " *** STENOSIS ***" if sv.get("stenosis") else ""
        report += f"  {lvl:12s}: {sv.get('width_pct','?')}% width{flag}\n"
    report += "\nPER-DISC PFIRRMANN\n" + "-" * 30 + "\n"
    for disc, g in ivd_grades.items():
        if g["grade"] is None: continue
        report += f"  {disc:12s}: Grade {g['grade']}/5 — {g['status']}\n"
    report += "\nT2 SIGNAL INTENSITY\n" + "-" * 30 + "\n"
    for disc, s in t2_signal.items():
        flag = " ⚠ DARK" if s.get("dark") else ""
        report += f"  {disc:12s}: {s['signal']}% — {s['hydration']}{flag}\n"
    report += "\nDISC HEIGHT / COMPRESSION\n" + "-" * 30 + "\n"
    for disc, h in disc_heights.items():
        flag = " ⚠ COMPRESSED" if h["compressed"] else ""
        report += f"  {disc:12s}: {h['height_pct']}% height{flag}\n"
    report += "\nFRACTURE RISK\n" + "-" * 30 + "\n"
    for vert, fr in fracture_risk.items():
        flag = " ⚠ RISK" if "risk" in fr.get("risk","").lower() else ""
        report += f"  {vert:8s}: A/P ratio {fr['ap_ratio']}{flag}\n"
    report += "\nMODEL UNCERTAINTY\n" + "-" * 30 + "\n"
    report += f"  Mean entropy: {uncertainty_mean} (0=certain, 1=uncertain)\n"
    report += "\nDETECTED STRUCTURES\n" + "-" * 30 + "\n"
    for s in detected: report += f"  • {s}\n"
    report += "\n" + "=" * 50 + "\n"
    report += "⚠  AI-GENERATED — For research purposes only.\n"
    report += "    Must be reviewed by a qualified radiologist.\n"

    # ── Save to patient history ───────────────────────────────────────
    history_record = {
        "filename"      : filename,
        "timestamp"     : now,
        "disease"       : disease,
        "severity"      : severity,
        "pfirrmann"     : pfirrmann_overall,
        "cobb_angle"    : curvature.get("angle"),
        "scoliosis_risk": curvature.get("risk"),
        "lordosis_type" : lordosis.get("type"),
        "stenosis_risk" : stenosis.get("risk"),
        "uncertainty"   : uncertainty_mean,
        "detected"      : detected,
        "patient"       : pat,
    }
    rec_id = add_to_history(history_record)

    return {
        # Core
        "num_slices"         : len(slices),
        "detected_structures": detected,
        "class_distribution" : cls_dist,
        "disease"            : disease,
        "severity"           : severity,
        "confidence"         : conf,
        "pfirrmann_grade"    : pfirrmann_overall,
        "report"             : report,
        "record_id"          : rec_id,
        # Images
        "image_b64"          : arr_to_b64((img_pre * 255).astype(np.uint8)),
        "overlay_b64"        : arr_to_b64(annotated),
        "mask_b64"           : arr_to_b64(mask_rgb),
        "scoliosis_b64"      : arr_to_b64(scoliosis_img),
        "legend_b64"         : arr_to_b64(legend_img),
        "gradcam_b64"        : gradcam_b64,
        "uncertainty_b64"    : uncertainty_b64,
        "slice_thumbs"       : slice_thumbs,
        # Analytics
        "ivd_grades"         : ivd_grades,
        "curvature"          : curvature,
        "disc_heights"       : disc_heights,
        "stenosis"           : stenosis,
        "lordosis"           : lordosis,
        "t2_signal"          : t2_signal,
        "fracture_risk"      : fracture_risk,
        "uncertainty_mean"   : uncertainty_mean,
        # Keep patient_info for PDF generation
        "_patient_info"      : pat,
    }

# ── Flask routes ──────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/training")
def training():
    hist = []
    for hf in [HIST_GPU, HIST_CPU]:
        if hf.exists():
            try: hist = json.load(open(hf)); break
            except: pass
    ckpts = {}
    for lbl, ck in [("Best", CKPT_BEST), ("Last", CKPT_LAST), ("CPU", CKPT_CPU)]:
        if ck.exists():
            try:
                import torch; c = torch.load(str(ck), map_location="cpu")
                ckpts[lbl] = {"epoch": c.get("epoch","?"),
                              "best_dice": round(c.get("best_dice", 0), 4)}
            except: pass
    return jsonify({"history": hist[-60:], "checkpoints": ckpts})

@app.route("/history")
def history():
    return jsonify(load_history())

@app.route("/history/<rec_id>", methods=["DELETE"])
def delete_history(rec_id):
    hist = [r for r in load_history() if r.get("id") != rec_id]
    save_history(hist)
    return jsonify({"ok": True})

@app.route("/health")
def health():
    import torch
    info = {
        "status": "running",
        "gpu"   : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "infer_size": INFER_SIZE,
    }
    if CKPT and CKPT.exists():
        try:
            c = torch.load(str(CKPT), map_location="cpu")
            info["checkpoint"] = f"epoch {c.get('epoch','?')} | dice={c.get('best_dice',0):.4f}"
            info["checkpoint_file"] = CKPT.name
        except: pass
    return jsonify(info)

# ── NEW: PDF export ───────────────────────────────────────────────────
_last_result  = {}   # cache latest result for PDF endpoint
_last_patient = {}

@app.route("/predict", methods=["POST"])
def predict():
    global _last_result, _last_patient
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    patient_info = {k: request.form.get(k, "")
                    for k in ["name","age","sex","ht","wt","notes"]}
    try:
        t0  = time.time()
        res = run_pred(f.read(), f.filename, patient_info)
        res["inference_ms"] = round((time.time() - t0) * 1000)
        _last_result  = res
        _last_patient = patient_info
        return jsonify(res)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    """Generate and return a PDF of the latest result."""
    from flask import Response
    data = request.get_json(silent=True) or _last_result
    pat  = data.get("_patient_info") or _last_patient
    if not data:
        return jsonify({"error": "No result available — run /predict first"}), 400
    try:
        pdf_bytes = generate_pdf(data, pat)
        fname = f"spine_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
        return Response(pdf_bytes, mimetype="application/pdf",
                        headers={"Content-Disposition": f"attachment; filename={fname}"})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ATM-Net++ | Spine MRI Analysis</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#e2e8f0;min-height:100vh}
:root{--blue:#63b3ed;--green:#68d391;--orange:#f6ad55;--red:#fc8181;--purple:#b794f4;
      --card:#161b27;--border:#1e2a3a;--muted:#718096}
.nav{background:#111827;border-bottom:1px solid var(--border);padding:12px 24px;
     display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:100}
.logo{font-size:19px;font-weight:800;color:var(--blue);letter-spacing:-0.5px}
.logo span{color:var(--green)}
.badge{background:#1e3a5f;color:var(--blue);font-size:10px;padding:2px 7px;border-radius:20px;font-weight:600}
.nav-tabs{display:flex;gap:3px;margin-left:auto}
.tab-btn{padding:6px 16px;border-radius:8px;border:none;cursor:pointer;font-size:12px;
         font-weight:600;background:transparent;color:#718096;transition:.15s;letter-spacing:.3px}
.tab-btn.active,.tab-btn:hover{background:#1e2a3a;color:#e2e8f0}
.container{max-width:1200px;margin:0 auto;padding:24px}
.section{display:none}.section.active{display:block}

/* Upload zone */
.drop-zone{border:2px dashed #2d3748;border-radius:16px;padding:40px;text-align:center;
           cursor:pointer;transition:.2s;background:var(--card)}
.drop-zone:hover,.drop-zone.drag{border-color:var(--blue);background:#0f1e2e}
.drop-icon{font-size:44px;margin-bottom:10px}
.file-info{background:#1a2535;border-radius:8px;padding:9px 14px;margin-top:10px;
           font-size:13px;color:var(--green);display:none}

/* Forms */
.form-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}
.form-group{display:flex;flex-direction:column;gap:3px}
.form-group label{font-size:11px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.4px}
.form-group input,.form-group select,.form-group textarea{
  background:#1a2535;border:1px solid #2d3748;border-radius:8px;
  padding:7px 11px;color:#e2e8f0;font-size:13px;outline:none;transition:.2s}
.form-group input:focus,.form-group select:focus{border-color:var(--blue)}
.form-group textarea{resize:vertical;min-height:60px;grid-column:1/-1}
.full-width{grid-column:1/-1}

/* Buttons */
.btn{padding:10px 22px;border-radius:10px;border:none;cursor:pointer;font-size:13px;
     font-weight:700;transition:.2s;display:inline-flex;align-items:center;gap:7px;letter-spacing:.3px}
.btn-primary{background:linear-gradient(135deg,#2b6cb0,#3182ce);color:#fff}
.btn-primary:hover{background:linear-gradient(135deg,#2c5282,#2b6cb0);transform:translateY(-1px)}
.btn-primary:disabled{background:#2d3748;cursor:not-allowed;transform:none;color:#718096}
.btn-outline{background:transparent;border:1px solid #2d3748;color:var(--muted);font-size:12px}
.btn-outline:hover{border-color:var(--blue);color:var(--blue)}
.btn-danger{background:transparent;border:1px solid #742a2a;color:var(--red);font-size:12px}
.btn-danger:hover{background:#2d1515}

/* Cards */
.card{background:var(--card);border-radius:14px;padding:16px;border:1px solid var(--border)}
.card h3{font-size:11px;color:var(--muted);font-weight:700;margin-bottom:12px;
         text-transform:uppercase;letter-spacing:.6px}
.card img{width:100%;border-radius:8px;border:1px solid var(--border)}

/* Diagnosis grid */
.diag-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:18px}
.diag-card{background:var(--card);border-radius:12px;padding:14px;border:1px solid var(--border);text-align:center}
.diag-label{font-size:9px;color:var(--muted);margin-bottom:5px;text-transform:uppercase;letter-spacing:.6px}
.diag-val{font-size:17px;font-weight:800;color:var(--blue)}

/* Results layout */
.results-grid{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin:16px 0}
.metric-row{display:flex;justify-content:space-between;align-items:center;
            padding:6px 0;border-bottom:1px solid var(--border);font-size:13px}
.metric-row:last-child{border:none}

/* Tags */
.struct-tags{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}
.tag{padding:3px 9px;background:#1e2a3a;border-radius:20px;font-size:11px;color:var(--muted)}
.tag.vert{color:#9ae6b4;background:#1a3a2a}.tag.ivd{color:var(--blue);background:#1a2a3a}
.tag.other{color:var(--orange);background:#3a2a1a}

/* Pfirrmann table */
.pf-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
.pf-table th{background:#1a2535;padding:7px 10px;text-align:left;color:var(--muted);font-weight:600;font-size:11px}
.pf-table td{padding:6px 10px;border-bottom:1px solid var(--border)}
.pf-g1{color:#68d391}.pf-g2{color:#9ae6b4}.pf-g3{color:var(--orange)}
.pf-g4{color:#fc8181}.pf-g5{color:#f56565;font-weight:700}

/* Slice viewer */
.slice-strip{display:flex;gap:6px;overflow-x:auto;padding:6px 0;margin-top:8px}
.slice-thumb{flex:0 0 80px;height:80px;border-radius:7px;border:2px solid transparent;
             cursor:pointer;object-fit:cover;transition:.15s;opacity:.7}
.slice-thumb:hover,.slice-thumb.active{border-color:var(--blue);opacity:1;transform:scale(1.05)}

/* Report box */
.report-box{background:#0d1117;border:1px solid var(--border);border-radius:10px;
            padding:14px;font-family:'Courier New',monospace;font-size:11.5px;color:#a0aec0;
            white-space:pre-wrap;max-height:300px;overflow-y:auto;line-height:1.6}

/* Training */
.train-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px}
.stat-card{background:var(--card);border-radius:12px;padding:14px;border:1px solid var(--border);text-align:center}
.stat-val{font-size:26px;font-weight:800;color:var(--blue)}
.stat-label{font-size:10px;color:var(--muted);margin-top:3px;text-transform:uppercase;letter-spacing:.4px}
#chartArea{background:var(--card);border-radius:12px;padding:18px;border:1px solid var(--border);margin-bottom:14px}
.epoch-table{width:100%;border-collapse:collapse;font-size:12px}
.epoch-table th{background:#1a2535;padding:7px 12px;text-align:left;color:var(--muted);font-weight:600}
.epoch-table td{padding:6px 12px;border-bottom:1px solid var(--border)}
.epoch-table tr.best td{color:var(--green)}

/* History */
.hist-item{background:var(--card);border:1px solid var(--border);border-radius:12px;
           padding:14px;display:flex;align-items:center;gap:14px;margin-bottom:8px}
.hist-badge{padding:4px 10px;border-radius:20px;font-size:11px;font-weight:700}
.sev-none{background:#1a3a2a;color:var(--green)}.sev-mild{background:#2d2a1a;color:var(--orange)}
.sev-moderate{background:#2d1a1a;color:#fc8181}.sev-severe{background:#4a1a1a;color:#f56565}

/* Spinner / overlay */
.spinner{display:inline-block;width:16px;height:16px;border:2px solid rgba(255,255,255,.15);
         border-top-color:var(--blue);border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:999;
         align-items:center;justify-content:center;flex-direction:column;gap:14px;font-size:15px}
.overlay.on{display:flex}
.progress-bar{width:220px;height:4px;background:#2d3748;border-radius:4px;overflow:hidden;margin-top:6px}
.progress-fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--green));
               border-radius:4px;animation:progress 3s ease-in-out infinite}
@keyframes progress{0%{width:5%}60%{width:85%}100%{width:95%}}

/* Risk colors */
.risk-normal{color:var(--green)}.risk-mild{color:var(--orange)}.risk-moderate{color:var(--red)}.risk-severe{color:#f56565}

/* Responsive */
@media(max-width:900px){
  .results-grid{grid-template-columns:1fr 1fr}
  .diag-grid{grid-template-columns:repeat(3,1fr)}
  .form-grid{grid-template-columns:1fr 1fr}
  .train-grid{grid-template-columns:1fr 1fr}
}
@media(max-width:600px){
  .results-grid,.diag-grid,.form-grid,.train-grid{grid-template-columns:1fr}
}
</style></head><body>

<div class="overlay" id="ov">
  <div class="spinner" style="width:44px;height:44px;border-width:4px"></div>
  <div id="ovTxt" style="color:#e2e8f0">Analyzing MRI...</div>
  <div class="progress-bar"><div class="progress-fill"></div></div>
</div>

<nav class="nav">
  <div class="logo">ATM-Net<span>++</span></div>
  <span class="badge">v2.0</span>
  <span style="color:#2d3748;font-size:13px">Lumbar Spine MRI Analysis</span>
  <div class="nav-tabs">
    <button class="tab-btn active" onclick="showTab('predict',this)">🔬 Predict</button>
    <button class="tab-btn" onclick="showTab('training',this)">📈 Training</button>
    <button class="tab-btn" onclick="showTab('history',this)">📋 History</button>
    <button class="tab-btn" onclick="showTab('about',this)">ℹ️ About</button>
  </div>
</nav>

<!-- ═══════════════════════════════ PREDICT ═══════════════════════════ -->
<div class="section active" id="s-predict"><div class="container">
  <h2 style="font-size:22px;font-weight:800;margin-bottom:4px">Spine MRI Analysis</h2>
  <p style="color:var(--muted);font-size:13px;margin-bottom:20px">
    Upload MRI → AI segmentation → per-disc Pfirrmann grading → scoliosis detection → clinical report
  </p>

  <div class="drop-zone" id="dz" onclick="document.getElementById('fi').click()">
    <div class="drop-icon">🩻</div>
    <h3 style="font-size:15px;margin-bottom:6px">Drop MRI file here or click to upload</h3>
    <p style="color:var(--muted);font-size:13px">Supports: .mha &nbsp;.mhd &nbsp;.nii &nbsp;.nii.gz &nbsp;.dcm &nbsp;.png &nbsp;.jpg</p>
    <input type="file" id="fi" style="display:none" accept=".mha,.mhd,.nii,.gz,.dcm,.png,.jpg,.jpeg" onchange="setFile(this.files[0])">
  </div>
  <div class="file-info" id="finfo"></div>

  <div style="margin-top:16px">
    <div style="font-size:11px;color:var(--muted);font-weight:700;margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px">Patient Details (optional)</div>
    <div class="form-grid">
      <div class="form-group"><label>Patient Name</label><input id="pname" placeholder="e.g. John Doe"></div>
      <div class="form-group"><label>Age</label><input type="number" id="page" placeholder="55" min="0" max="120"></div>
      <div class="form-group"><label>Sex</label>
        <select id="psex"><option value="">—</option><option value="M">Male</option><option value="F">Female</option></select>
      </div>
      <div class="form-group"><label>Height (cm)</label><input type="number" id="pht" placeholder="170"></div>
      <div class="form-group"><label>Weight (kg)</label><input type="number" id="pwt" placeholder="75"></div>
      <div class="form-group"><label>BMI</label><input id="pbmi" placeholder="Auto" readonly style="color:var(--blue)"></div>
      <div class="form-group full-width"><label>Clinical Notes / Symptoms</label>
        <textarea id="pnotes" placeholder="e.g. Lower back pain radiating to left leg, worsened by sitting..."></textarea>
      </div>
    </div>
  </div>

  <div style="margin-top:16px;display:flex;gap:10px;flex-wrap:wrap">
    <button class="btn btn-primary" id="aBtn" onclick="analyze()" disabled>🔬 Analyze MRI</button>
    <button class="btn btn-outline" onclick="clearAll()">✕ Clear</button>
  </div>

  <!-- Results -->
  <div id="results" style="display:none;margin-top:28px">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;flex-wrap:wrap;gap:8px">
      <h3 style="font-size:18px;font-weight:800">Analysis Results</h3>
      <div style="display:flex;gap:8px;align-items:center">
        <span id="itime" style="font-size:12px;color:var(--muted)"></span>
        <button class="btn btn-outline" onclick="dlReport()">⬇ Report .txt</button>
        <button class="btn btn-outline" onclick="dlPDF()">📄 PDF Report</button>
        <button class="btn btn-outline" onclick="dlImages()">🖼 Images</button>
      </div>
    </div>

    <!-- Primary diagnosis row -->
    <div class="diag-grid">
      <div class="diag-card"><div class="diag-label">Diagnosis</div><div class="diag-val" id="r-dis" style="font-size:13px">—</div></div>
      <div class="diag-card"><div class="diag-label">Severity</div><div class="diag-val" id="r-sev">—</div></div>
      <div class="diag-card"><div class="diag-label">Confidence</div><div class="diag-val" id="r-con" style="color:var(--green)">—</div></div>
      <div class="diag-card"><div class="diag-label">Pfirrmann</div><div class="diag-val" id="r-pfi" style="color:var(--orange)">—</div></div>
      <div class="diag-card"><div class="diag-label">Scoliosis</div><div class="diag-val" id="r-scol" style="font-size:13px">—</div></div>
    </div>

    <!-- Image grid: 6 panels -->
    <div class="results-grid">
      <div class="card"><h3>MRI Input</h3><img id="i-orig" src="" alt="Original"></div>
      <div class="card"><h3>Segmentation Overlay</h3><img id="i-over" src="" alt="Overlay" style="cursor:zoom-in" onclick="zoom(this)"></div>
      <div class="card"><h3>Segmentation Mask</h3><img id="i-mask" src="" alt="Mask"></div>
      <div class="card"><h3>Scoliosis Analysis</h3><img id="i-scol" src="" alt="Scoliosis"></div>
      <div class="card"><h3>🔥 Grad-CAM <span style="color:var(--muted);font-weight:400;font-size:10px">Model attention</span></h3><img id="i-gcam" src="" alt="GradCAM" style="cursor:zoom-in" onclick="zoom(this)"></div>
      <div class="card"><h3>⚡ Uncertainty Map <span style="color:var(--muted);font-weight:400;font-size:10px">Blue=certain Red=unsure</span></h3><img id="i-unc" src="" alt="Uncertainty" style="cursor:zoom-in" onclick="zoom(this)"></div>
    </div>

    <!-- Multi-slice viewer -->
    <div class="card" style="margin-top:12px">
      <h3>Multi-Slice Viewer <span style="color:var(--muted);font-weight:400;font-size:10px">(click to enlarge)</span></h3>
      <div class="slice-strip" id="sliceStrip"></div>
    </div>

    <!-- Per-disc Pfirrmann + Disc heights + Stenosis + T2 signal + Fracture risk -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px">
      <div class="card">
        <h3>Per-Disc Pfirrmann Grading</h3>
        <table class="pf-table" id="pfTable">
          <thead><tr><th>Level</th><th>Grade</th><th>Status</th><th>Conf</th></tr></thead>
          <tbody id="pfBody"></tbody>
        </table>
      </div>
      <div class="card">
        <h3>IVD T2 Signal Intensity</h3>
        <div id="t2Div"></div>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:12px">
      <div class="card">
        <h3>Canal Stenosis</h3>
        <div id="stenosisDiv"></div>
      </div>
      <div class="card">
        <h3>Disc Height & Compression</h3>
        <div id="dhDiv"></div>
      </div>
      <div class="card">
        <h3>Vertebral Fracture Risk</h3>
        <div id="fractureDiv"></div>
        <div style="margin-top:12px">
          <h3 style="margin-bottom:6px">Spine Curvature</h3>
          <div id="scolDiv"></div>
        </div>
      </div>
    </div>

    <!-- Uncertainty score badge -->
    <div id="uncBadge" style="display:none;margin-top:12px;padding:10px 16px;
         background:#1a2535;border-radius:10px;border:1px solid var(--border);
         display:flex;align-items:center;gap:12px;font-size:13px">
      <span>⚡ Model Uncertainty:</span>
      <span id="uncScore" style="color:var(--orange);font-weight:700"></span>
      <span style="color:var(--muted);font-size:11px">(0=certain · 1=uncertain — high values = check manually)</span>
    </div>

    <!-- Detected structures + color legend -->
    <div style="display:grid;grid-template-columns:1fr auto;gap:12px;margin-top:12px;align-items:start">
      <div>
        <div class="card">
          <h3>Detected Structures</h3>
          <div class="struct-tags" id="stags"></div>
        </div>
        <div class="card" style="margin-top:12px">
          <h3>Class Distribution</h3>
          <div id="cdist"></div>
        </div>
      </div>
      <div class="card" style="min-width:160px">
        <h3>Color Legend</h3>
        <img id="i-legend" src="" alt="Legend" style="width:100%">
      </div>
    </div>

    <!-- Clinical report -->
    <div style="margin-top:12px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <div style="font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.5px">Clinical Report</div>
        <button class="btn btn-outline" onclick="dlReport()">⬇ Download .txt</button>
      </div>
      <div class="report-box" id="rbox"></div>
    </div>

    <div style="margin-top:10px;padding:10px 14px;background:#1a0e0e;border-radius:10px;
                border:1px solid #742a2a;font-size:12px;color:var(--red)">
      ⚠ AI-generated analysis. Must be reviewed and validated by a qualified radiologist before any clinical use.
    </div>
  </div>
</div></div>

<!-- ═══════════════════════════════ TRAINING ══════════════════════════ -->
<div class="section" id="s-training"><div class="container">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px">
    <div>
      <h2 style="font-size:22px;font-weight:800">Training Monitor</h2>
      <p style="color:var(--muted);font-size:13px;margin-top:3px">ResUNet+CBAM | SPIDER dataset | RTX 3050</p>
    </div>
    <button class="btn btn-outline" onclick="loadTrain()">🔄 Refresh</button>
  </div>
  <div class="train-grid">
    <div class="stat-card"><div class="stat-val" id="st-ep">—</div><div class="stat-label">Epochs Done</div></div>
    <div class="stat-card"><div class="stat-val" id="st-bd" style="color:var(--green)">—</div><div class="stat-label">Best Val Dice</div></div>
    <div class="stat-card"><div class="stat-val" id="st-td">—</div><div class="stat-label">Latest Train Dice</div></div>
    <div class="stat-card"><div class="stat-val" id="st-vl">—</div><div class="stat-label">Latest Val Loss</div></div>
  </div>
  <div id="chartArea">
    <div style="font-size:11px;font-weight:700;color:var(--muted);margin-bottom:10px;text-transform:uppercase;letter-spacing:.5px">Dice Score Progress</div>
    <canvas id="dc" height="200"></canvas>
  </div>
  <div style="overflow-x:auto">
    <table class="epoch-table">
      <thead><tr><th>Ep</th><th>Train Loss</th><th>Train Dice</th><th>Val Loss</th><th>Val Dice</th><th>Best</th><th>Gap</th></tr></thead>
      <tbody id="etbody"></tbody>
    </table>
  </div>
</div></div>

<!-- ═══════════════════════════════ HISTORY ═══════════════════════════ -->
<div class="section" id="s-history"><div class="container">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px">
    <div>
      <h2 style="font-size:22px;font-weight:800">Patient History</h2>
      <p style="color:var(--muted);font-size:13px;margin-top:3px">Last 50 analyses stored locally</p>
    </div>
    <button class="btn btn-outline" onclick="loadHistory()">🔄 Refresh</button>
  </div>
  <div id="histList"><p style="color:var(--muted);text-align:center;padding:40px">No history yet — run an analysis first</p></div>
</div></div>

<!-- ═══════════════════════════════ ABOUT ════════════════════════════ -->
<div class="section" id="s-about"><div class="container" style="max-width:760px">
  <h2 style="font-size:22px;font-weight:800;margin-bottom:20px">About ATM-Net++</h2>
  <div style="display:grid;gap:14px">
    <div class="card"><h3>Architecture</h3>
      <div class="metric-row"><span>Model</span><strong style="color:var(--blue)">ResUNet + CBAM Attention</strong></div>
      <div class="metric-row"><span>Parameters</span><strong style="color:var(--blue)">15.75M</strong></div>
      <div class="metric-row"><span>Dataset</span><strong style="color:var(--blue)">SPIDER (210 patients, 447 scans)</strong></div>
      <div class="metric-row"><span>Training</span><strong style="color:var(--blue)">RTX 3050 4GB + Kaggle T4 (256×256)</strong></div>
      <div class="metric-row"><span>Best Dice</span><strong style="color:var(--green)">0.6529 (epoch 48, Kaggle)</strong></div>
    </div>
    <div class="card"><h3>Features</h3>
      <div class="metric-row"><span>🔬 Segmentation</span><span style="color:var(--blue)">19 classes — vertebrae, IVDs, sacrum, canal</span></div>
      <div class="metric-row"><span>📊 Pfirrmann</span><span style="color:var(--blue)">Per-disc grading (8 levels, grade 1–5)</span></div>
      <div class="metric-row"><span>📐 Scoliosis</span><span style="color:var(--blue)">Cobb angle estimation from vertebra centroids</span></div>
      <div class="metric-row"><span>📏 Disc Height</span><span style="color:var(--blue)">Compression detection per IVD</span></div>
      <div class="metric-row"><span>🖼 Multi-slice</span><span style="color:var(--blue)">8-slice viewer with TTA</span></div>
      <div class="metric-row"><span>📋 History</span><span style="color:var(--blue)">Patient records with timeline</span></div>
      <div class="metric-row"><span>📄 Reports</span><span style="color:var(--blue)">Downloadable clinical text report</span></div>
    </div>
    <div class="card"><h3>Segmentation Classes (19)</h3>
      <div class="struct-tags">
        <span class="tag vert">L5/L4/L3/L2/L1/T12/T11/T10 (Vertebrae)</span>
        <span class="tag other">Sacrum</span>
        <span class="tag ivd">L5-S1 through T10-T11 (IVDs)</span>
        <span class="tag other">Spinal Canal</span>
        <span class="tag">Background</span>
      </div>
    </div>
    <div class="card"><h3>API</h3>
      <div class="metric-row"><span>POST /predict</span><strong style="color:var(--blue)">Upload MRI → full analysis JSON</strong></div>
      <div class="metric-row"><span>GET /training</span><strong style="color:var(--blue)">Live training history</strong></div>
      <div class="metric-row"><span>GET /history</span><strong style="color:var(--blue)">Patient records</strong></div>
      <div class="metric-row"><span>GET /health</span><strong style="color:var(--blue)">GPU/model status</strong></div>
    </div>
  </div>
</div></div>

<!-- ═══════════════════════════════ IMAGE ZOOM MODAL ═════════════════ -->
<div id="zoomModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.9);z-index:9999;
     align-items:center;justify-content:center;cursor:zoom-out" onclick="this.style.display='none'">
  <img id="zoomImg" src="" style="max-width:90vw;max-height:90vh;border-radius:12px;box-shadow:0 0 60px rgba(0,0,0,.8)">
</div>

<script>
let selFile=null, lastRep="", lastData=null;

// ── BMI auto-calc ─────────────────────────────────────────────────────
document.getElementById('pht').addEventListener('input', calcBMI);
document.getElementById('pwt').addEventListener('input', calcBMI);
function calcBMI(){
  const h=parseFloat(document.getElementById('pht').value)/100;
  const w=parseFloat(document.getElementById('pwt').value);
  if(h>0&&w>0) document.getElementById('pbmi').value=(w/(h*h)).toFixed(1);
  else document.getElementById('pbmi').value='';
}

// ── Tab navigation ────────────────────────────────────────────────────
function showTab(n,btn){
  document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('s-'+n).classList.add('active');
  btn.classList.add('active');
  if(n==='training') loadTrain();
  if(n==='history')  loadHistory();
}

// ── Drag & drop ───────────────────────────────────────────────────────
const dz=document.getElementById('dz');
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('drag')});
dz.addEventListener('dragleave',()=>dz.classList.remove('drag'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('drag');
  const f=e.dataTransfer.files[0];if(f)setFile(f)});
function setFile(f){
  selFile=f;
  const fi=document.getElementById('finfo');
  fi.style.display='block';
  fi.textContent=`✓  ${f.name}   (${(f.size/1024/1024).toFixed(2)} MB)`;
  document.getElementById('aBtn').disabled=false;
  document.getElementById('results').style.display='none';
}

// ── Analyze ───────────────────────────────────────────────────────────
async function analyze(){
  if(!selFile) return;
  document.getElementById('ov').classList.add('on');
  document.getElementById('ovTxt').textContent='Uploading & analyzing MRI...';
  const fd=new FormData();
  fd.append('file', selFile);
  fd.append('name', document.getElementById('pname').value);
  fd.append('age',  document.getElementById('page').value);
  fd.append('sex',  document.getElementById('psex').value);
  fd.append('ht',   document.getElementById('pht').value);
  fd.append('wt',   document.getElementById('pwt').value);
  fd.append('notes',document.getElementById('pnotes').value);
  try{
    const r=await fetch('/predict',{method:'POST',body:fd});
    const d=await r.json();
    document.getElementById('ov').classList.remove('on');
    if(d.error){alert('Error: '+d.error+'\n\n'+d.trace);return;}
    lastData=d; showRes(d);
  }catch(e){
    document.getElementById('ov').classList.remove('on');
    alert('Request failed: '+e.message);
  }
}

// ── Display results ───────────────────────────────────────────────────
function showRes(d){
  document.getElementById('results').style.display='block';
  document.getElementById('itime').textContent=
    `⏱ ${d.inference_ms}ms  |  ${d.num_slices} slices analyzed`;

  // Primary diagnosis
  document.getElementById('r-dis').textContent=d.disease;
  const sev=document.getElementById('r-sev');
  sev.textContent=d.severity;
  sev.style.color=d.severity==='None'?'#68d391':d.severity==='Mild'?'#f6ad55':
                  d.severity==='Moderate'?'#fc8181':'#f56565';
  document.getElementById('r-con').textContent=d.confidence+'%';
  document.getElementById('r-pfi').textContent=d.pfirrmann_grade+'/5';
  const cv=d.curvature||{};
  const scolEl=document.getElementById('r-scol');
  if(cv.angle!=null){
    scolEl.textContent=cv.risk;
    scolEl.className='diag-val risk-'+cv.risk.toLowerCase().split(' ')[0];
  } else { scolEl.textContent='N/A'; }

  // Images
  document.getElementById('i-orig').src='data:image/png;base64,'+d.image_b64;
  document.getElementById('i-over').src='data:image/png;base64,'+d.overlay_b64;
  document.getElementById('i-mask').src='data:image/png;base64,'+d.mask_b64;
  document.getElementById('i-scol').src='data:image/png;base64,'+d.scoliosis_b64;
  document.getElementById('i-legend').src='data:image/png;base64,'+d.legend_b64;
  if(d.gradcam_b64)    document.getElementById('i-gcam').src='data:image/png;base64,'+d.gradcam_b64;
  if(d.uncertainty_b64)document.getElementById('i-unc').src='data:image/png;base64,'+d.uncertainty_b64;

  // Uncertainty badge
  if(d.uncertainty_mean!=null){
    document.getElementById('uncBadge').style.display='flex';
    const u=d.uncertainty_mean;
    const ucol=u<0.2?'var(--green)':u<0.5?'var(--orange)':'var(--red)';
    document.getElementById('uncScore').style.color=ucol;
    document.getElementById('uncScore').textContent=u.toFixed(3)+
      (u<0.2?' — High confidence':u<0.5?' — Moderate uncertainty':' — Low confidence, review carefully');
  }

  // Multi-slice strip
  const strip=document.getElementById('sliceStrip'); strip.innerHTML='';
  (d.slice_thumbs||[]).forEach((b64,i)=>{
    const img=document.createElement('img');
    img.src='data:image/png;base64,'+b64;
    img.className='slice-thumb'+(i===Math.floor((d.slice_thumbs.length)/2)?' active':'');
    img.title=`Slice ${i+1}`;
    img.onclick=()=>{
      document.querySelectorAll('.slice-thumb').forEach(t=>t.classList.remove('active'));
      img.classList.add('active');
      document.getElementById('i-over').src='data:image/png;base64,'+b64;
    };
    strip.appendChild(img);
  });

  // Per-disc Pfirrmann table
  const pfBody=document.getElementById('pfBody'); pfBody.innerHTML='';
  const grades=d.ivd_grades||{};
  Object.entries(grades).forEach(([disc,g])=>{
    if(g.grade==null) return;
    const cls='pf-g'+g.grade;
    pfBody.innerHTML+=`<tr>
      <td><strong>${disc}</strong></td>
      <td class="${cls}"><strong>Grade ${g.grade}/5</strong></td>
      <td class="${cls}">${g.status}</td>
      <td style="color:var(--muted)">${(g.confidence*100).toFixed(0)}%</td>
    </tr>`;
  });
  if(!pfBody.innerHTML) pfBody.innerHTML='<tr><td colspan="4" style="color:var(--muted);padding:10px">No IVDs detected in this slice</td></tr>';

  // Disc heights
  const dh=d.disc_heights||{};
  const dhDiv=document.getElementById('dhDiv'); dhDiv.innerHTML='';
  Object.entries(dh).forEach(([disc,h])=>{
    const warn=h.compressed?'<span style="color:var(--red);font-size:10px"> ⚠ COMPRESSED</span>':'';
    const pct=h.height_pct;
    dhDiv.innerHTML+=`<div class="metric-row">
      <span>${disc}</span>
      <div style="display:flex;align-items:center;gap:8px">
        <div style="width:70px;background:#1a2535;border-radius:4px;height:5px">
          <div style="width:${Math.min(pct*3,100)}%;background:${h.compressed?'var(--red)':'var(--blue)'};height:5px;border-radius:4px"></div>
        </div>
        <span style="font-size:12px;color:var(--blue)">${pct}%</span>${warn}
      </div></div>`;
  });
  if(!Object.keys(dh).length) dhDiv.innerHTML='<p style="color:var(--muted);font-size:12px;padding:8px 0">No disc heights detected</p>';

  // T2 signal intensity
  const t2Div=document.getElementById('t2Div'); t2Div.innerHTML='';
  Object.entries(d.t2_signal||{}).forEach(([disc,s])=>{
    const dark=s.dark;
    const col=s.signal>60?'var(--green)':s.signal>40?'var(--blue)':s.signal>25?'var(--orange)':'var(--red)';
    const warn=dark?'<span style="color:var(--red);font-size:10px"> ⚠ DARK</span>':'';
    t2Div.innerHTML+=`<div class="metric-row">
      <span>${disc}</span>
      <div style="display:flex;align-items:center;gap:8px">
        <div style="width:70px;background:#1a2535;border-radius:4px;height:5px">
          <div style="width:${Math.min(s.signal,100)}%;background:${col};height:5px;border-radius:4px"></div>
        </div>
        <span style="font-size:11px;color:${col}">${s.signal}%</span>${warn}
      </div></div>`;
  });
  if(!Object.keys(d.t2_signal||{}).length) t2Div.innerHTML='<p style="color:var(--muted);font-size:12px;padding:8px 0">No IVDs detected</p>';

  // Canal stenosis
  const stenDiv=document.getElementById('stenosisDiv'); stenDiv.innerHTML='';
  const sten=d.stenosis||{};
  if(sten.detected){
    const rCol=sten.risk.includes('Stenosis')?'var(--red)':'var(--green)';
    stenDiv.innerHTML=`<div class="metric-row"><span>Overall</span><strong style="color:${rCol}">${sten.risk}</strong></div>`;
    Object.entries(sten.levels||{}).forEach(([lvl,sv])=>{
      const sc=sv.stenosis?'var(--red)':'var(--green)';
      const flag=sv.stenosis?'⚠ STENOSIS':'Normal';
      stenDiv.innerHTML+=`<div class="metric-row">
        <span>${lvl}</span>
        <div style="display:flex;align-items:center;gap:6px">
          <span style="font-size:11px;color:var(--muted)">${sv.width_pct}%</span>
          <span style="font-size:11px;color:${sc};font-weight:600">${flag}</span>
        </div></div>`;
    });
  } else {
    stenDiv.innerHTML='<p style="color:var(--muted);font-size:12px;padding:8px 0">Canal not visible in this slice</p>';
  }

  // Fracture risk
  const fracDiv=document.getElementById('fractureDiv'); fracDiv.innerHTML='';
  Object.entries(d.fracture_risk||{}).forEach(([vert,fr])=>{
    const risk=fr.risk.includes('risk');
    const col=risk?'var(--red)':'var(--green)';
    fracDiv.innerHTML+=`<div class="metric-row">
      <span>${vert}</span>
      <div style="display:flex;align-items:center;gap:6px">
        <span style="font-size:11px;color:var(--muted)">A/P ${fr.ap_ratio}</span>
        <span style="font-size:11px;color:${col};font-weight:${risk?700:400}">${risk?'⚠ Risk':'OK'}</span>
      </div></div>`;
  });
  if(!Object.keys(d.fracture_risk||{}).length) fracDiv.innerHTML='<p style="color:var(--muted);font-size:12px;padding:6px 0">No vertebrae detected</p>';

  // Curvature / lordosis
  const scolDiv=document.getElementById('scolDiv'); scolDiv.innerHTML='';
  if(cv.angle!=null){
    const riskColor=cv.risk.includes('Normal')?'var(--green)':cv.risk.includes('Mild')?'var(--orange)':'var(--red)';
    const lk=d.lordosis||{};
    scolDiv.innerHTML=`
      <div class="metric-row"><span>Cobb Angle</span><strong style="color:${riskColor}">${cv.angle}°</strong></div>
      <div class="metric-row"><span>Scoliosis</span><strong style="color:${riskColor}">${cv.risk}</strong></div>
      <div class="metric-row"><span>Curve Type</span><strong style="color:var(--blue);font-size:11px">${lk.type||'—'}</strong></div>`;
  } else {
    scolDiv.innerHTML='<p style="color:var(--muted);font-size:12px">Insufficient vertebrae for analysis</p>';
  }

  // Detected structures
  const st=document.getElementById('stags'); st.innerHTML='';
  (d.detected_structures||[]).forEach(s=>{
    const sp=document.createElement('span');
    sp.className='tag '+(s.startsWith('Vertebra')||s.includes('(L')||s.includes('(T'))?'vert':
                  s.startsWith('IVD')||s.includes('/')?'ivd':'other';
    sp.textContent=s; st.appendChild(sp);
  });
  if(!d.detected_structures?.length) st.innerHTML='<span style="color:var(--muted);font-size:12px">No foreground structures detected</span>';

  // Class distribution bars
  const cd=document.getElementById('cdist'); cd.innerHTML='';
  Object.entries(d.class_distribution||{}).forEach(([nm,v])=>{
    cd.innerHTML+=`<div class="metric-row">
      <span style="font-size:12px">${nm}</span>
      <div style="display:flex;align-items:center;gap:8px">
        <div style="width:80px;background:#1a2535;border-radius:4px;height:5px">
          <div style="width:${Math.min(v.percent*5,100)}%;background:var(--blue);height:5px;border-radius:4px"></div>
        </div>
        <span style="font-size:12px;color:var(--blue);min-width:38px">${v.percent}%</span>
      </div></div>`;
  });
  if(!Object.keys(d.class_distribution||{}).length)
    cd.innerHTML='<p style="color:var(--muted);font-size:12px;padding:8px 0">No foreground classes detected — upload a sagittal spine MRI</p>';

  lastRep=d.report||'';
  document.getElementById('rbox').textContent=lastRep;
  document.getElementById('results').scrollIntoView({behavior:'smooth'});
}

// ── Download report ───────────────────────────────────────────────────
function dlReport(){
  if(!lastRep) return;
  const b=new Blob([lastRep],{type:'text/plain'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(b);
  a.download='spine_report_'+new Date().toISOString().slice(0,10)+'.txt';
  a.click();
}
async function dlPDF(){
  if(!lastData){ alert('Run an analysis first'); return; }
  document.getElementById('ovTxt').textContent='Generating PDF...';
  document.getElementById('ov').classList.add('on');
  try{
    const r=await fetch('/export_pdf',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(lastData)
    });
    document.getElementById('ov').classList.remove('on');
    if(!r.ok){ alert('PDF failed: '+await r.text()); return; }
    const blob=await r.blob();
    const a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download='spine_report_'+new Date().toISOString().slice(0,10)+'.pdf';
    a.click();
  }catch(e){
    document.getElementById('ov').classList.remove('on');
    alert('PDF error: '+e.message);
  }
}
function dlImages(){
  if(!lastData) return;
  ['image_b64','overlay_b64','mask_b64','scoliosis_b64','gradcam_b64','uncertainty_b64'].forEach((k,i)=>{
    if(!lastData[k]) return;
    const names=['original','overlay','mask','scoliosis','gradcam','uncertainty'];
    const a=document.createElement('a');
    a.href='data:image/png;base64,'+lastData[k];
    a.download=`spine_${names[i]}.png`; a.click();
  });
}

// ── Image zoom ────────────────────────────────────────────────────────
function zoom(img){
  document.getElementById('zoomImg').src=img.src;
  document.getElementById('zoomModal').style.display='flex';
}
document.querySelectorAll('.result-card img').forEach(i=>i.addEventListener('click',()=>zoom(i)));

// ── Clear ─────────────────────────────────────────────────────────────
function clearAll(){
  selFile=null; lastRep=''; lastData=null;
  document.getElementById('finfo').style.display='none';
  document.getElementById('results').style.display='none';
  document.getElementById('aBtn').disabled=true;
  document.getElementById('fi').value='';
}

// ── Training monitor ──────────────────────────────────────────────────
async function loadTrain(){
  try{
    const d=await (await fetch('/training')).json();
    const h=d.history||[];
    if(!h.length){
      document.getElementById('etbody').innerHTML=
        '<tr><td colspan="7" style="color:var(--muted);text-align:center;padding:20px">No training history yet</td></tr>';
      return;
    }
    const last=h[h.length-1];
    const best=h.reduce((a,b)=>(b.vd||0)>(a.vd||0)?b:a,h[0]);
    document.getElementById('st-ep').textContent=last.ep||last.epoch||'?';
    document.getElementById('st-bd').textContent=(best.vd||0).toFixed(4);
    document.getElementById('st-td').textContent=(last.td||0).toFixed(4);
    document.getElementById('st-vl').textContent=(last.vl||0).toFixed(4);
    const tb=document.getElementById('etbody'); tb.innerHTML='';
    h.slice(-25).reverse().forEach(row=>{
      const ep=row.ep||row.epoch||0;
      const ib=parseFloat(row.vd||0)>=(best.vd||0)-0.0001;
      tb.innerHTML+=`<tr class="${ib?'best':''}">
        <td>${ep}</td>
        <td>${(row.tl||0).toFixed(4)}</td>
        <td>${(row.td||0).toFixed(4)}</td>
        <td>${(row.vl||0).toFixed(4)}</td>
        <td><strong>${(row.vd||0).toFixed(4)}</strong></td>
        <td style="color:var(--green)">${(best.vd||0).toFixed(4)}</td>
        <td style="color:${(row.gap||row.td-row.vd||0)>0.05?'var(--orange)':'var(--muted)'}">${((row.gap||0)>0?'+':'')}${(row.gap||0).toFixed(3)}</td>
      </tr>`;
    });
    drawDice(h);
  }catch(e){ console.error('Train load failed',e); }
}

function drawDice(h){
  const cv=document.getElementById('dc');
  const ctx=cv.getContext('2d');
  cv.width=cv.parentElement.clientWidth-36; cv.height=200;
  const W=cv.width, H=cv.height, pad={l:42,r:20,t:18,b:32};
  ctx.clearRect(0,0,W,H);
  const eps=h.map(r=>r.ep||r.epoch||0);
  const tvd=h.map(r=>r.td||0), vvd=h.map(r=>r.vd||0);
  const maxD=Math.max(.01,...tvd,...vvd);
  const minEp=eps[0]||0, maxEp=eps[eps.length-1]||1;
  const xp=e=>pad.l+(e-minEp)/(maxEp-minEp||1)*(W-pad.l-pad.r);
  const yp=d=>H-pad.b-(d/maxD)*(H-pad.t-pad.b);
  // Grid lines
  [0,.25,.5,.75,1].forEach(v=>{
    const y=yp(v*maxD);
    ctx.strokeStyle='#1e2a3a'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.stroke();
    ctx.fillStyle='#4a5568'; ctx.font='10px sans-serif';
    ctx.fillText((v*maxD).toFixed(2),2,y+3);
  });
  // Target line 0.90
  ctx.strokeStyle='rgba(246,173,85,.4)'; ctx.lineWidth=1; ctx.setLineDash([5,4]);
  ctx.beginPath(); ctx.moveTo(pad.l,yp(0.9)); ctx.lineTo(W-pad.r,yp(0.9)); ctx.stroke();
  ctx.setLineDash([]); ctx.fillStyle='#f6ad55'; ctx.font='10px sans-serif';
  ctx.fillText('Target 0.90',W-90,yp(0.9)-4);
  // Lines
  const drawLine=(data,col,dash=[])=>{
    ctx.strokeStyle=col; ctx.lineWidth=2.5; ctx.setLineDash(dash);
    ctx.beginPath(); data.forEach((d,i)=>i===0?ctx.moveTo(xp(eps[i]),yp(d)):ctx.lineTo(xp(eps[i]),yp(d))); ctx.stroke();
    ctx.setLineDash([]);
  };
  drawLine(tvd,'rgba(99,179,237,.6)',[3,3]);
  drawLine(vvd,'#68d391');
  // Epoch labels
  const step=Math.max(1,Math.floor(eps.length/8));
  ctx.fillStyle='#4a5568'; ctx.font='10px sans-serif';
  eps.forEach((e,i)=>{if(i%step===0)ctx.fillText(e,xp(e)-6,H-8);});
  // Legend
  ctx.fillStyle='rgba(99,179,237,.6)'; ctx.fillRect(pad.l,5,10,3);
  ctx.fillStyle='#a0aec0'; ctx.font='11px sans-serif'; ctx.fillText('Train',pad.l+14,10);
  ctx.fillStyle='#68d391'; ctx.fillRect(pad.l+60,5,10,3);
  ctx.fillText('Val',pad.l+74,10);
}
setInterval(()=>{if(document.getElementById('s-training').classList.contains('active'))loadTrain();},60000);

// ── Patient history ───────────────────────────────────────────────────
async function loadHistory(){
  try{
    const hist=await (await fetch('/history')).json();
    const el=document.getElementById('histList');
    if(!hist.length){
      el.innerHTML='<p style="color:var(--muted);text-align:center;padding:40px">No history yet</p>';
      return;
    }
    const sevClass=s=>({None:'sev-none',Mild:'sev-mild',Moderate:'sev-moderate',Severe:'sev-severe'}[s]||'sev-mild');
    el.innerHTML=hist.map(r=>`
      <div class="hist-item">
        <div style="flex:1;min-width:0">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
            <strong style="font-size:14px">${r.patient?.name||r.filename||'Unknown'}</strong>
            <span class="hist-badge ${sevClass(r.severity)}">${r.severity||'?'}</span>
            ${r.scoliosis_risk&&r.scoliosis_risk!=='Normal'?`<span class="hist-badge sev-moderate">${r.scoliosis_risk}</span>`:''}
            ${r.stenosis_risk&&r.stenosis_risk.includes('Stenosis')?`<span class="hist-badge sev-severe">Stenosis</span>`:''}
          </div>
          <div style="font-size:12px;color:var(--muted);display:flex;gap:14px;flex-wrap:wrap">
            <span>📅 ${r.timestamp}</span>
            <span>🔬 ${r.disease}</span>
            <span>📊 Pfirrmann ${r.pfirrmann}/5</span>
            ${r.cobb_angle?`<span>📐 ${r.cobb_angle}° (${r.scoliosis_risk})</span>`:''}
            ${r.lordosis_type?`<span>〜 ${r.lordosis_type}</span>`:''}
            ${r.uncertainty!=null?`<span>⚡ Uncertainty: ${r.uncertainty}</span>`:''}
            ${r.patient?.age?`<span>👤 ${r.patient.age}y ${r.patient.sex||''}</span>`:''}
          </div>
        </div>
        <button class="btn btn-danger" onclick="deleteHist('${r.id}')">🗑</button>
      </div>`).join('');
  }catch(e){ console.error(e); }
}
async function deleteHist(id){
  await fetch('/history/'+id,{method:'DELETE'});
  loadHistory();
}
</script>
</body></html>"""

# ── Server startup ────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    print("=" * 52)
    print("  ATM-Net++ Web Server  v2.0")
    print("=" * 52)
    print(f"  GPU    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if CKPT and CKPT.exists():
        try:
            c = torch.load(str(CKPT), map_location="cpu")
            print(f"  Model  : epoch {c.get('epoch','?')} | best_dice={c.get('best_dice',0):.4f}")
        except: pass
    print(f"  Size   : {INFER_SIZE}×{INFER_SIZE}")
    print(f"  Open   : http://localhost:5000")
    print("=" * 52)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
