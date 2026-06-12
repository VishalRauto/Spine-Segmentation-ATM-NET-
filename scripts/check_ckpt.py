import torch, sys
from pathlib import Path

OUT = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run")

for name in ["best_model.pth", "last_model.pth"]:
    p = OUT / name
    if not p.exists():
        print(f"{name}: NOT FOUND")
        continue
    c = torch.load(str(p), map_location="cpu")
    ep   = c.get("epoch", "?")
    dice = c.get("best_dice", 0.0)
    cfg  = c.get("cfg", {})
    print(f"\n{name}:")
    print(f"  epoch     = {ep}")
    print(f"  best_dice = {dice:.4f}")
    print(f"  cfg       = {cfg}")
    pc = c.get("per_class_dice", {})
    if pc:
        print("  per_class_dice:")
        for k,v in pc.items():
            flag = " <-- DOMINANT" if v > 0.90 else ""
            print(f"    {k:20s}: {v:.4f}{flag}")
    keys = list(c.get("model_state_dict", {}).keys())
    print(f"  state_dict keys: {len(keys)}")
    print(f"  first key: {keys[0] if keys else 'none'}")
    # Check if model has aux head
    has_aux = any("aux" in k for k in keys)
    has_ds3 = any("ds3" in k for k in keys)
    print(f"  has aux head: {has_aux}")
    print(f"  has ds3 head: {has_ds3}")
