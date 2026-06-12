import torch, json
from pathlib import Path

ckpt_path = Path(r"C:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run\best_model.pth")
hist_path = Path(r"C:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run\history.json")

if ckpt_path.exists():
    c = torch.load(str(ckpt_path), map_location="cpu")
    print(f"Checkpoint : epoch {c.get('epoch','?')} | best_dice={c.get('best_dice',0):.4f}")
    mod_time = ckpt_path.stat().st_mtime
    import datetime
    print(f"Saved at   : {datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')}")
else:
    print("No checkpoint found")

if hist_path.exists():
    h = json.load(open(hist_path))
    print(f"History    : {len(h)} epochs logged")
    last = h[-1]
    print(f"Last epoch : ep={last.get('ep','?')} | val_dice={last.get('vd',0):.4f} | train_dice={last.get('td',0):.4f}")
