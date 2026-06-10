"""
Export ATM-Net++ to ONNX format for production deployment.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/best.pth --output model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default="configs/base_config.yaml")
    p.add_argument("--output",     default="checkpoints/atmnet_pp.onnx")
    p.add_argument("--img-size",   type=int, default=512)
    p.add_argument("--opset",      type=int, default=17)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    from models.atmnet_plus_plus import ATMNetPlusPlus
    model = ATMNetPlusPlus(
        img_size=(args.img_size, args.img_size),
        in_channels=model_cfg.get("in_channels", 1),
        num_seg_classes=config.get("segmentation", {}).get("num_classes", 20),
        use_text=False,
        use_demographics=False,
        deep_supervision=False,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 1, args.img_size, args.img_size)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        opset_version=args.opset,
        input_names=["image"],
        output_names=["seg_logits"],
        dynamic_axes={
            "image":      {0: "batch"},
            "seg_logits": {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(f"[✓] Exported to {out_path}")

    # Verify
    try:
        import onnx
        model_onnx = onnx.load(str(out_path))
        onnx.checker.check_model(model_onnx)
        print(f"[✓] ONNX model verified. Size: {out_path.stat().st_size / 1e6:.1f} MB")
    except ImportError:
        print("[!] onnx package not installed — skipping verification")


if __name__ == "__main__":
    main()
