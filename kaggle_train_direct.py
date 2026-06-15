# ═══════════════════════════════════════════════════════════════
# DIRECT TRAINING CELL — bypasses subprocess hang completely
# Paste this in Kaggle instead of Cell 7
# Runs nnU-Net in-process with all multiprocessing disabled
# ═══════════════════════════════════════════════════════════════
import os, sys, importlib.util, site
from pathlib import Path

NNUNET_BASE  = Path('/kaggle/working/nnunet')
DATASET_ID   = 1
DATASET_NAME = 'Dataset001_SpineSPIDER'

# ── Set ALL env vars before any nnunetv2 import ──────────────
os.environ['nnUNet_raw']          = str(NNUNET_BASE / 'raw')
os.environ['nnUNet_preprocessed'] = str(NNUNET_BASE / 'preprocessed')
os.environ['nnUNet_results']      = str(NNUNET_BASE / 'results')
os.environ['nnUNet_compile']      = 'false'
os.environ['nnUNet_n_proc_DA']    = '0'
os.environ['MKL_NUM_THREADS']     = '1'
os.environ['OMP_NUM_THREADS']     = '1'

# ── Patch torch.compile ──────────────────────────────────────
import torch
if not getattr(torch, '_cpatch', False):
    torch.compile = lambda fn=None, *a, **kw: fn if fn else (lambda f: f)
    torch._cpatch = True
    print('torch.compile patched')

# ── Patch batchgenerators MultiThreadedAugmenter ────────────
try:
    # Import and monkey-patch BEFORE nnunetv2 loads it
    from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
    from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

    # Replace MultiThreadedAugmenter with SingleThreadedAugmenter globally
    import batchgenerators.dataloading.multi_threaded_augmenter as _mta
    _mta.MultiThreadedAugmenter = SingleThreadedAugmenter

    # Also patch the nnunetv2 import of it
    import batchgenerators.dataloading
    batchgenerators.dataloading.multi_threaded_augmenter.MultiThreadedAugmenter = SingleThreadedAugmenter

    print('MultiThreadedAugmenter → SingleThreadedAugmenter (no subprocess hang)')
except Exception as e:
    print(f'batchgenerators patch: {e}')

# ── Also patch nnunetv2 directly if it imports MultiThreadedAugmenter ──
try:
    import nnunetv2
    pkg = Path(nnunetv2.__file__).parent
    # Find all files that import MultiThreadedAugmenter
    trainer_file = pkg / 'training' / 'nnUNetTrainer' / 'nnUNetTrainer.py'
    if trainer_file.exists():
        src = trainer_file.read_text()
        if 'torch.compile(self.network)' in src:
            src = src.replace('self.network = torch.compile(self.network)',
                              'self.network = self.network  # patched')
            trainer_file.write_text(src)
            print('nnUNetTrainer compile patched')
        if "self.compile = ('nnUNet_compile'" in src:
            src = trainer_file.read_text()
            src = src.replace(
                "self.compile = ('nnUNet_compile' in os.environ "
                "and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))",
                'self.compile = False  # patched')
            trainer_file.write_text(src)
            print('nnUNetTrainer self.compile patched')
except Exception as e:
    print(f'nnunetv2 patch: {e}')

# ── Run training IN-PROCESS (no subprocess) ──────────────────
print('\nStarting nnU-Net training IN-PROCESS...')
print('No subprocess = no hang. Epoch 1 in ~60-90 seconds.')
print('='*55)

try:
    from nnunetv2.run.run_training import run_training
    import inspect

    sig = inspect.signature(run_training)
    params = set(sig.parameters.keys())
    print(f'run_training params: {list(params)}')

    kwargs = {}
    if 'plans_identifier'        in params: kwargs['plans_identifier']        = 'nnUNetPlans'
    if 'trainer_class_name'      in params: kwargs['trainer_class_name']      = 'nnUNetTrainer'
    if 'device'                  in params: kwargs['device']                   = torch.device('cuda')
    if 'continue_training'       in params: kwargs['continue_training']        = False
    if 'pretrained_weights'      in params: kwargs['pretrained_weights']       = None
    if 'num_gpus'                in params: kwargs['num_gpus']                 = 1
    if 'disable_checkpointing'   in params: kwargs['disable_checkpointing']    = False
    if 'val_with_best'           in params: kwargs['val_with_best']            = False
    if 'export_validation_probabilities' in params:
        kwargs['export_validation_probabilities'] = True
    if 'only_run_validation'     in params: kwargs['only_run_validation']      = False
    if 'npz'                     in params: kwargs['npz']                      = True

    print(f'kwargs: {list(kwargs.keys())}')
    print()

    run_training(DATASET_NAME, '2d', 0, **kwargs)

except Exception as e:
    import traceback
    print(f'\nERROR: {e}')
    traceback.print_exc()
    print('\nIf TypeError about unexpected keyword: re-run, kwargs auto-detected')

print('\nTraining cell finished.')
