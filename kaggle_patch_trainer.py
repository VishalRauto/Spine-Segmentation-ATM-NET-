# ═══════════════════════════════════════════════════════════
# PATCH CELL — Run this ONCE before Cell 7
# Directly patches the nnUNetTrainer to use 0 worker processes
# This fixes the DataLoader hang on Kaggle
# ═══════════════════════════════════════════════════════════
import importlib.util, site
from pathlib import Path

spec = importlib.util.find_spec('nnunetv2')
assert spec, 'nnunetv2 not installed'
pkg = Path(spec.origin).parent

# ── Patch 1: nnUNetTrainer.py ────────────────────────────────
tf = pkg / 'training' / 'nnUNetTrainer' / 'nnUNetTrainer.py'
src = tf.read_text()

patches = [
    # Disable torch.compile (Python 3.12)
    ('self.network = torch.compile(self.network)',
     'self.network = self.network  # PATCHED'),
    ("self.compile = ('nnUNet_compile' in os.environ "
     "and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))",
     'self.compile = False  # PATCHED'),
    # Force num_processes_segmentation_export = 0
    ('self.num_processes_segmentation_export = 8',
     'self.num_processes_segmentation_export = 0  # PATCHED'),
    ('self.num_processes_segmentation_export = 4',
     'self.num_processes_segmentation_export = 0  # PATCHED'),
]
for old, new in patches:
    if old in src:
        src = src.replace(old, new)
        print(f'  Patched nnUNetTrainer: {old[:60]}')

tf.write_text(src)
print(f'  Saved: {tf}')

# ── Patch 2: data_augmentation_moreDA.py — the real hang source ──
# nnU-Net uses batchgenerators for data augmentation
# It spawns worker threads that deadlock in Kaggle's sandbox
da_files = list(pkg.rglob('nnUNetTrainerCls*.py')) + \
           list(pkg.rglob('default_data_augmentation.py')) + \
           list(pkg.rglob('data_augmentation_moreDA.py'))

for df in da_files:
    src2 = df.read_text()
    changed = False
    for old, new in [
        ("'num_processes': get_allowed_n_proc_DA()",
         "'num_processes': 0  # PATCHED"),
        ('num_processes=get_allowed_n_proc_DA()',
         'num_processes=0  # PATCHED'),
    ]:
        if old in src2:
            src2 = src2.replace(old, new)
            changed = True
            print(f'  Patched {df.name}: {old[:50]}')
    if changed:
        df.write_text(src2)

# ── Patch 3: batchgenerators — the actual multiprocessing source ──
try:
    import batchgenerators
    bg_path = Path(batchgenerators.__file__).parent
    mt_files = list(bg_path.rglob('multithreaded_augmenter.py'))
    for mf in mt_files:
        src3 = mf.read_text()
        # Force num_processes to 1 (not 0 — needs at least 1 for batchgenerators)
        for old, new in [
            ('self.num_processes = num_processes',
             'self.num_processes = 1  # PATCHED: Kaggle fix'),
        ]:
            if old in src3 and 'PATCHED' not in src3:
                src3 = src3.replace(old, new)
                mf.write_text(src3)
                print(f'  Patched batchgenerators: {mf.name}')
except Exception as e:
    print(f'  batchgenerators patch skipped: {e}')

# ── Patch 4: sitecustomize.py ────────────────────────────────
sc = Path(site.getsitepackages()[0]) / 'sitecustomize.py'
sc.write_text("""
import os
os.environ.setdefault('nnUNet_n_proc_DA', '0')
try:
    import torch as _t
    if not getattr(_t, '_cpatch', False):
        _t.compile = lambda fn=None, *a, **kw: fn if fn else (lambda f: f)
        _t._cpatch = True
except: pass
""")
print(f'  sitecustomize.py updated')

# ── Verify ────────────────────────────────────────────────────
src_check = tf.read_text()
has_compile_patch = 'self.network = self.network  # PATCHED' in src_check
print(f'\nVerification:')
print(f'  compile patch : {has_compile_patch}')
print(f'  sitecustomize : OK')
print('\nAll patches applied. Now run the training cell.')
