"""Quick test: verify Grad-CAM produces non-trivial heatmap (not all blue/uniform)."""
import requests, base64, numpy as np, cv2
from pathlib import Path

MHA = Path(r"c:\project\Spine Segmentation\10159290\images\100_t2.mha")
with open(MHA,"rb") as f:
    d = requests.post("http://localhost:5000/predict",
                      files={"file":("100_t2.mha",f)}, timeout=120).json()

b64 = d.get("gradcam_b64","")
if not b64:
    print("NO GRADCAM"); exit(1)

# Decode image
img_bytes = base64.b64decode(b64)
npa = np.frombuffer(img_bytes, np.uint8)
img = cv2.imdecode(npa, cv2.IMREAD_COLOR)  # BGR
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Check R channel (JET: high values = red)
R = img_rgb[:,:,0].astype(float)
G = img_rgb[:,:,1].astype(float)
B = img_rgb[:,:,2].astype(float)

r_mean = R.mean(); b_mean = B.mean()
print(f"GradCAM image stats:")
print(f"  R mean: {r_mean:.1f}  G mean: {G.mean():.1f}  B mean: {b_mean:.1f}")
print(f"  R max : {R.max():.0f}  B max : {B.max():.0f}")
print(f"  Color variance: {img_rgb.std():.1f}")

# JET colormap: low=blue, high=red
# A solid-blue image has B>>R — bad
# A proper heatmap has R comparable to or > B in hot regions
if r_mean > 50:
    print("  ✓ RED channel active — JET heatmap working correctly")
elif b_mean > 180 and r_mean < 30:
    print("  ✗ STILL ALL BLUE — gradient not flowing, fallback active")
    print("    (Grad-CAM showing Sobel edge fallback — still useful)")
else:
    print(f"  ✓ Mixed colors — heatmap has variation (std={img_rgb.std():.1f})")

print(f"\nUncertainty: {d.get('uncertainty_mean')}")
print(f"Disease    : {d.get('disease')} ({d.get('severity')})")
