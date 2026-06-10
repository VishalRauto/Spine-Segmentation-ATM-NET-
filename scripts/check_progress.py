import json
from pathlib import Path

hist_file = Path(r"C:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run\history.json")
if not hist_file.exists():
    print("No history file yet"); exit()

h = json.load(open(hist_file))
print(f"Total logged epochs: {len(h)}")
print()

# Trend every 5 epochs
print(f"  {'Ep':>4}  {'TrainDice':>10}  {'ValDice':>9}  {'BestDice':>9}")
print("  " + "-"*40)
best_so_far = 0
for r in h:
    ep = r.get("ep", r.get("epoch", 0))
    td = r.get("td", r.get("train_dice", 0))
    vd = r.get("vd", r.get("val_dice", 0))
    best_so_far = max(best_so_far, vd)
    if ep % 5 == 0 or ep == h[-1].get("ep", h[-1].get("epoch", 0)):
        star = " *" if vd == best_so_far else ""
        print(f"  {ep:>4}  {td:>10.4f}  {vd:>9.4f}  {best_so_far:>9.4f}{star}")

print()
all_vd = [r.get("vd", r.get("val_dice", 0)) for r in h]
all_td = [r.get("td", r.get("train_dice", 0)) for r in h]
best_val = max(all_vd)
first_val = all_vd[0]
latest_val = all_vd[-1]
latest_ep = h[-1].get("ep", h[-1].get("epoch", "?"))
best_ep = h[all_vd.index(best_val)].get("ep", h[all_vd.index(best_val)].get("epoch", "?"))

print("=" * 50)
print("  IMPROVEMENT SUMMARY")
print("=" * 50)
print(f"  Epochs completed  : {len(h)}")
print(f"  Latest epoch      : {latest_ep}")
print(f"  First Val Dice    : {first_val:.4f}")
print(f"  Latest Val Dice   : {latest_val:.4f}")
print(f"  Best Val Dice     : {best_val:.4f}  (epoch {best_ep})")
print(f"  Total improvement : +{best_val - first_val:.4f}  ({(best_val-first_val)/max(first_val,0.001)*100:.0f}%)")
print(f"  Train Dice latest : {all_td[-1]:.4f}")
print()
print("  STATUS:")
if best_val >= 0.90:
    print("  TARGET ACHIEVED: Dice >= 0.90!")
elif best_val >= 0.80:
    print("  Excellent — very close to target 0.90")
elif best_val >= 0.70:
    print("  Strong improvement — continue training")
elif best_val >= 0.60:
    print("  Good progress — model improving steadily")
else:
    print("  Model converging — keep training")
print("=" * 50)
