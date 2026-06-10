import torch
ckpt = torch.load(
    r"C:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run\best_model.pth",
    map_location="cpu"
)
print("Saved epoch :", ckpt.get("epoch"))
print("Best Dice   :", round(ckpt.get("best_dice", 0), 4))
pc = ckpt.get("per_class_dice", {})
if pc:
    print()
    print("Per-class Dice (best checkpoint):")
    for name, d in sorted(pc.items(), key=lambda x: -x[1]):
        bar = "#" * int(d * 30)
        print(f"  {name:<18} {d:.4f}  {bar}")
