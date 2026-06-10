"""
Standalone inference script for ATM-Net++.

Usage:
    python scripts/run_inference.py --image path/to/image.mha
    python scripts/run_inference.py --image path/to/image.mha --report "Disc bulge at L4-L5"
    python scripts/run_inference.py --image path/to/image.mha --checkpoint checkpoints/best.pth
    python scripts/run_inference.py --image path/to/image.mha --save-overlay outputs/overlay.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="ATM-Net++ Inference")
    p.add_argument("--image",      required=True, help="Path to MRI file")
    p.add_argument("--checkpoint", default="checkpoints/atmnet_pp_best.pth")
    p.add_argument("--config",     default="configs/base_config.yaml")
    p.add_argument("--report",     default=None, help="Radiology report text")
    p.add_argument("--modality",   default="T2", choices=["T1", "T2", "STIR"])
    p.add_argument("--age",        type=int, default=None)
    p.add_argument("--sex",        default=None, choices=["M", "F"])
    p.add_argument("--device",     default="auto")
    p.add_argument("--tta",        action="store_true", help="Test-time augmentation")
    p.add_argument("--save-overlay",   default=None, help="Save segmentation overlay PNG")
    p.add_argument("--save-report",    default=None, help="Save PDF report")
    p.add_argument("--save-json",      default=None, help="Save JSON results")
    p.add_argument("--show",       action="store_true", help="Display results in terminal")
    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[ATM-Net++] Using device: {device}")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Build predictor
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        from inference.predictor import SpinePredictor
        predictor = SpinePredictor.from_checkpoint(str(checkpoint_path), config, device)
    else:
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}. Using untrained model.")
        from models.atmnet_plus_plus import ATMNetPlusPlus
        model_cfg = config.get("model", {})
        model = ATMNetPlusPlus(
            img_size=tuple(model_cfg.get("img_size", [512, 512])),
            in_channels=model_cfg.get("in_channels", 1),
            num_seg_classes=config.get("segmentation", {}).get("num_classes", 20),
            use_text=True,
            use_demographics=True,
        )
        from inference.predictor import SpinePredictor
        predictor = SpinePredictor(model=model, device=device, config=config, use_tta=args.tta)

    # Build demographics
    demographics = None
    if args.age or args.sex:
        demographics = {}
        if args.age: demographics["age"] = args.age
        if args.sex: demographics["sex"] = args.sex

    # Run inference
    print(f"[ATM-Net++] Processing: {args.image}")
    result = predictor.predict_from_file(
        image_path=args.image,
        report_text=args.report,
        demographics=demographics,
        modality=args.modality,
    )

    # Display results
    if args.show or not any([args.save_overlay, args.save_report, args.save_json]):
        print("\n" + "=" * 60)
        print("  ATM-Net++ RESULTS")
        print("=" * 60)
        cls = result["classification"]
        sev = result["severity"]
        lvl = result["levels"]
        print(f"  Diagnosis:       {cls['disease_name'].replace('_',' ')} ({cls['confidence']*100:.1f}%)")
        print(f"  Severity:        {sev['name']}")
        print(f"  Pfirrmann Grade: {result['pfirrmann_grade']:.1f}/5")
        print(f"  Affected Levels: {', '.join(lvl['affected']) or 'None'}")
        print(f"  Inference Time:  {result.get('inference_time_ms', 0):.0f}ms")
        print(f"  Slices:          {result.get('num_slices_processed', 0)}")
        print("\n  REPORT IMPRESSION:")
        print(f"  {result['report']['impression']}")
        print("\n  RECOMMENDATION:")
        print(f"  {result['report']['recommendation']}")
        print("=" * 60)
        print("\n  Detected Structures:", ", ".join(result["segmentation"]["detected_structures"]))

    # Save overlay
    if args.save_overlay and result["segmentation"].get("overlay_b64"):
        import base64
        overlay_bytes = base64.b64decode(result["segmentation"]["overlay_b64"])
        out_path = Path(args.save_overlay)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(overlay_bytes)
        print(f"[✓] Overlay saved: {out_path}")

    # Save JSON
    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove base64 images from JSON output (too large)
        result_json = {k: v for k, v in result.items()
                       if k not in {"gradcam_b64"} and not (isinstance(v, dict) and "overlay_b64" in v)}
        if "segmentation" in result_json:
            result_json["segmentation"] = {k: v for k, v in result_json["segmentation"].items()
                                           if k != "overlay_b64"}
            result_json["segmentation"]["mask"] = []  # Omit large mask array
        with open(out_path, "w") as f:
            json.dump(result_json, f, indent=2, default=str)
        print(f"[✓] Results saved: {out_path}")

    # Save PDF report
    if args.save_report:
        from backend.services.pdf_service import generate_pdf_report
        pdf_bytes = generate_pdf_report(result, patient_info=demographics)
        out_path = Path(args.save_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"[✓] PDF report saved: {out_path}")


if __name__ == "__main__":
    main()
