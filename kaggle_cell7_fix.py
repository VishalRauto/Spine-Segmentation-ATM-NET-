# ═══════════════════════════════════════════════════════════
# CELL 7 (FIXED): Train nnU-Net — no multiprocessing hang
# Stop current Cell 7 first, then paste this entire cell
# ═══════════════════════════════════════════════════════════
import os, sys, subprocess, shutil, site, importlib.util
from pathlib import Path

NNUNET_BASE  = Path('/kaggle/working/nnunet')
RAW_DIR      = NNUNET_BASE / 'raw'
PREPROC_DIR  = NNUNET_BASE / 'preprocessed'
RESULTS_DIR  = NNUNET_BASE / 'results'
DATASET_ID   = 1
DATASET_NAME = 'Dataset001_SpineSPIDER'

env = {
    **os.environ,
    'nnUNet_raw'          : str(RAW_DIR),
    'nnUNet_preprocessed' : str(PREPROC_DIR),
    'nnUNet_results'      : str(RESULTS_DIR),
    'nnUNet_compile'      : 'false',
    'nnUNet_n_proc_DA'    : '0',   # FIX: disable multiprocessing — prevents hang
    'PYTHONDONTWRITEBYTECODE': '1',
}

# Patch nnUNetTrainer — disable compile AND multiprocessing
spec = importlib.util.find_spec('nnunetv2')
if spec:
    pkg = Path(spec.origin).parent
    tf  = pkg / 'training' / 'nnUNetTrainer' / 'nnUNetTrainer.py'
    if tf.exists():
        src = tf.read_text()
        patches = [
            ('self.network = torch.compile(self.network)',
             'self.network = self.network  # patched'),
            ("self.compile = ('nnUNet_compile' in os.environ "
             "and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))",
             'self.compile = False  # patched'),
            # KEY FIX: force num_processes to 0 — prevents DataLoader hang
            ('self.num_processes = get_allowed_n_proc_DA()',
             'self.num_processes = 0  # patched: no multiprocessing on Kaggle'),
            ('allowed_num_processes = get_allowed_n_proc_DA()',
             'allowed_num_processes = 0  # patched'),
        ]
        changed = False
        for old, new in patches:
            if old in src:
                src = src.replace(old, new)
                changed = True
                print(f'  Patched: {old[:60]}')
        if changed:
            tf.write_text(src)
            print('  Trainer saved')
        else:
            print('  Already patched or different version')

# sitecustomize safety net
sc = Path(site.getsitepackages()[0]) / 'sitecustomize.py'
sc.write_text("""
try:
    import torch as _t
    if not getattr(_t, '_cpatch', False):
        _t.compile = lambda fn=None, *a, **kw: fn if fn else (lambda f: f)
        _t._cpatch = True
except: pass
""")

# Verify preprocessing exists
plans = PREPROC_DIR / DATASET_NAME / 'nnUNetPlans.json'
assert plans.exists(), 'Run Cell 6 first'
print(f'Plans: {plans}')

# Find CLI
cli = shutil.which('nnUNetv2_train') or '/usr/local/bin/nnUNetv2_train'
cmd = [sys.executable, cli, str(DATASET_ID), '2d', '0', '--npz']
print(f'Command: {" ".join(cmd)}')
print('nnUNet_n_proc_DA=0 (no multiprocessing — fixes hang)')
print('Training started — epoch 1 should appear in ~60 seconds')
print('='*55)

proc = subprocess.Popen(
    cmd, env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1
)
try:
    for line in proc.stdout:
        print(line, end='', flush=True)
except KeyboardInterrupt:
    proc.terminate()
    print('\nInterrupted — checkpoint saved. Re-run to resume.')

proc.wait()
print(f'\nReturn code: {proc.returncode}')
if proc.returncode == 0:
    print('Training complete! Run Cell 8.')
else:
    print('Check output above. Re-run to resume from checkpoint.')
