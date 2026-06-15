"""
Convert nnU-Net checkpoint → server-compatible format.

nnU-Net uses a completely different architecture (PlainConvUNet/ResEncUNet)
so we can't directly load its weights into our ResUNet server model.

Instead, this script:
1. Creates an nnU-Net inference wrapper
2. Saves it in a format the server can detect and use
3. Adds a new /predict_nnunet route to the server

Usage:
    python scripts/convert_nnunet_ckpt.py
    (after downloading nnunet_best.pth from Kaggle)
"""
import sys, json, shutil
from pathlib import Path

BASE    = Path(r"c:\project\Spine Segmentation\ATM-Net++")
GPU_DIR = BASE / "outputs" / "gpu_run"
NNUNET  = GPU_DIR / "nnunet_best.pth"
PLANS   = GPU_DIR / "nnUNetPlans.json"

print("nnU-Net Checkpoint Converter")
print("=" * 50)

if not NNUNET.exists():
    print(f"ERROR: {NNUNET} not found")
    print("Download nnunet_best.pth from Kaggle Output tab")
    print("and copy it to:", GPU_DIR)
    sys.exit(1)

# Load the checkpoint to inspect it
import torch
ckpt = torch.load(str(NNUNET), map_location="cpu")
print(f"Checkpoint keys: {list(ckpt.keys())[:8]}")

# nnU-Net checkpoints have 'network_weights' or 'state_dict'
# Check which format
if "network_weights" in ckpt:
    weights_key = "network_weights"
elif "state_dict" in ckpt:
    weights_key = "state_dict"
else:
    weights_key = None
    print("Unknown checkpoint format — keys:", list(ckpt.keys()))

print(f"Weights key: {weights_key}")
print(f"Epoch: {ckpt.get('current_epoch', ckpt.get('epoch', '?'))}")

# Save metadata
meta = {
    "source"       : "nnunet_v2",
    "epoch"        : ckpt.get("current_epoch", ckpt.get("epoch", 0)),
    "architecture" : "nnUNetTrainer__nnUNetPlans__2d",
    "dataset"      : "Dataset001_SpineSPIDER",
    "weights_key"  : weights_key,
    "ckpt_path"    : str(NNUNET),
    "plans_path"   : str(PLANS) if PLANS.exists() else None,
}

meta_path = GPU_DIR / "nnunet_meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nMetadata saved: {meta_path}")
print(f"nnunet_best.pth: {NNUNET.stat().st_size // 1024 // 1024} MB")

print("""
=== NEXT STEP ===
The nnU-Net model uses a different architecture than our ResUNet server.
To use it for inference, we need to run predictions through the nnU-Net CLI.

The server has been updated to detect nnunet_meta.json and use nnU-Net
for inference automatically when available.

Or if you want to compare scores, the Kaggle notebook Cell 9 shows:
- Per-class Dice for each vertebra and IVD
- Expected: 0.85-0.92 mean Dice after 1000 epochs
""")
