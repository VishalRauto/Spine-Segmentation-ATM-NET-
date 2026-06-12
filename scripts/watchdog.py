"""
Training Watchdog — auto-restarts train_final.py on CUDA crash.
Runs until epoch 150 or Dice > 0.90.
"""
import subprocess, sys, time, json
from pathlib import Path

PYTHON   = r"C:\Users\VISHAL RAUTO\anaconda3\python.exe"
SCRIPT   = "scripts/train_best.py"
CKPT     = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run\best_model.pth")
MAX_RESTARTS = 50
TARGET_DICE  = 0.90
TARGET_EPOCH = 150

def get_ckpt_info():
    try:
        import torch
        from pathlib import Path
        best_p = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run\best_model.pth")
        last_p = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run\last_model.pth")
        # Use whichever has a higher epoch
        best_ep, best_dice = 0, 0.0
        if best_p.exists():
            c = torch.load(str(best_p), map_location="cpu")
            best_ep = c.get("epoch", 0); best_dice = c.get("best_dice", 0.0)
        last_ep = 0
        if last_p.exists():
            c2 = torch.load(str(last_p), map_location="cpu")
            last_ep = c2.get("epoch", 0)
        # Report highest epoch seen
        return max(best_ep, last_ep), best_dice
    except:
        return 0, 0.0

restart = 0
print("=" * 55)
print("  ATM-Net++ Training Watchdog")
print("  Auto-restarts on CUDA crash")
print("=" * 55)

while restart < MAX_RESTARTS:
    ep, dice = get_ckpt_info()
    print(f"\n[Watchdog] Restart #{restart+1} | Last saved: epoch={ep}, dice={dice:.4f}")

    if ep >= TARGET_EPOCH:
        print(f"[Watchdog] Reached epoch {TARGET_EPOCH}. Training complete!")
        break
    if dice >= TARGET_DICE:
        print(f"[Watchdog] Target Dice {TARGET_DICE} achieved! Done.")
        break

    # Run training
    env = {"CUDA_LAUNCH_BLOCKING": "1",
           "PATH": r"C:\Users\VISHAL RAUTO\anaconda3;C:\Users\VISHAL RAUTO\anaconda3\Scripts;"
                   r"C:\Windows\System32;C:\Windows"}
    import os; full_env = {**os.environ, "CUDA_LAUNCH_BLOCKING": "1"}

    proc = subprocess.run(
        [PYTHON, SCRIPT],
        cwd=r"c:\project\Spine Segmentation\ATM-Net++",
        env=full_env,
    )

    ep_new, dice_new = get_ckpt_info()
    print(f"[Watchdog] Process ended. New state: epoch={ep_new}, dice={dice_new:.4f}")

    if ep_new >= TARGET_EPOCH or dice_new >= TARGET_DICE:
        break

    if ep_new <= ep:
        print("[Watchdog] No progress made. Waiting 10s before retry...")
        time.sleep(10)

    restart += 1

ep_final, dice_final = get_ckpt_info()
print(f"\n[Watchdog] Final result: epoch={ep_final}, best_dice={dice_final:.4f}")
print("Training complete." if ep_final >= TARGET_EPOCH or dice_final >= TARGET_DICE
      else f"Stopped after {restart} restarts.")
