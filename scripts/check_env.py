import importlib, sys
print("Python:", sys.version)
pkgs = {
    'torch': 'torch',
    'numpy': 'numpy',
    'SimpleITK': 'SimpleITK',
    'monai': 'monai',
    'opencv': 'cv2',
    'scikit-learn': 'sklearn',
    'scipy': 'scipy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'nibabel': 'nibabel',
    'pydicom': 'pydicom',
}
missing = []
for name, mod in pkgs.items():
    try:
        m = importlib.import_module(mod)
        v = getattr(m, '__version__', 'installed')
        print(f"[OK]      {name:<20} {v}")
    except ImportError:
        print(f"[MISSING] {name}")
        missing.append(name)

print()
if missing:
    print("Missing packages:", missing)
    print("Install with:  pip install " + " ".join(missing))
else:
    print("All core packages present.")

# Check CUDA
try:
    import torch
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
except:
    pass
