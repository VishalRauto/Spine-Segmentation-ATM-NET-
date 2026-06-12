"""Hit the live server with a real MRI and print what it returns."""
import requests, json, base64, numpy as np
from pathlib import Path

MHA = Path(r"c:\project\Spine Segmentation\10159290\images\100_t2.mha")
URL = "http://localhost:5000/predict"

with open(MHA, "rb") as f:
    resp = requests.post(URL, files={"file": ("100_t2.mha", f, "application/octet-stream")}, timeout=120)

d = resp.json()
if "error" in d:
    print("ERROR:", d["error"])
    if "trace" in d: print(d["trace"])
else:
    print(f"Slices      : {d['num_slices']}")
    print(f"Disease     : {d['disease']} ({d['severity']}) {d['confidence']}%")
    print(f"Pfirrmann   : {d['pfirrmann_grade']}")
    print(f"Detected    : {d['detected_structures']}")
    print(f"Distribution:")
    for k,v in d.get('class_distribution',{}).items():
        print(f"  {k:20s}: {v['percent']:5.1f}%  ({v['pixels']} px)")
    
    # Decode mask and check color distribution
    mask_bytes = base64.b64decode(d["mask_b64"])
    import cv2
    npa = np.frombuffer(mask_bytes, np.uint8)
    mask_img = cv2.imdecode(npa, cv2.IMREAD_COLOR)  # BGR
    
    # Convert BGR to RGB and check unique colors
    mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    unique, counts = np.unique(mask_rgb.reshape(-1,3), axis=0, return_counts=True)
    total = mask_rgb.reshape(-1,3).shape[0]
    
    COLORS = {
        (0,0,0):"background",(255,50,50):"V1",(255,120,50):"V2",(255,200,50):"V3",
        (200,255,50):"V4",(100,255,50):"V5",(50,255,100):"V6",(50,255,200):"V7",
        (50,200,255):"V8",(50,100,255):"Sacrum",(150,50,255):"IVD1",(255,50,200):"IVD2",
        (255,50,100):"IVD3",(200,100,255):"IVD4",(100,200,255):"IVD5",(255,150,50):"IVD6",
        (50,255,150):"IVD7",(150,255,50):"IVD8",(220,220,220):"Canal",
    }
    
    print(f"\nMask image colors (top 10):")
    for col, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        pct = cnt/total*100
        name = COLORS.get(tuple(col), f"RGB{tuple(col)}")
        print(f"  {name:15s}: {pct:5.1f}%  ({cnt} px)")
    
    # Check if mask is solid single color (bad) or multi-color (good)
    n_unique = len(unique)
    dominant_pct = counts.max()/total*100
    print(f"\nUnique colors in mask: {n_unique}")
    print(f"Dominant color       : {dominant_pct:.1f}%")
    if n_unique == 1:
        print("❌ STILL BROKEN: Single solid color in mask")
    elif dominant_pct > 99:
        print("❌ NEARLY SOLID: >99% one color")
    elif dominant_pct > 90:
        print(f"⚠ Background dominant ({dominant_pct:.0f}%) but fg structures visible — THIS IS NORMAL for spine MRI")
        print("  The spine occupies ~5-15% of image area — mask will look mostly dark")
    else:
        print(f"✓ GOOD: {n_unique} colors, dominant={dominant_pct:.0f}% — clear multi-class output")
