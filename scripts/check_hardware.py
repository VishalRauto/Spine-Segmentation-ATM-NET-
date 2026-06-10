import sys, platform, subprocess
print("="*60)
print("  Hardware & Environment Check")
print("="*60)
print(f"  OS      : {platform.system()} {platform.release()}")
print(f"  Python  : {sys.version.split()[0]}")
import multiprocessing
print(f"  CPU     : {multiprocessing.cpu_count()} cores")
try:
    import psutil
    ram = psutil.virtual_memory()
    print(f"  RAM     : {ram.total//1024**3} GB total, {ram.available//1024**3} GB free")
except: pass

import torch
print(f"\n  PyTorch : {torch.__version__}")
print(f"  CUDA    : {torch.version.cuda}")
print(f"  GPU avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}   : {p.name}")
        print(f"  VRAM    : {p.total_memory//1024**2} MB")
        print(f"  Compute : {p.major}.{p.minor}")
else:
    print("  GPU     : None detected (CPU only)")
    print("\n  To enable GPU:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

# Check disk space
import shutil
total, used, free = shutil.disk_usage("c:/")
print(f"\n  Disk    : {free//1024**3} GB free of {total//1024**3} GB")
print("="*60)
