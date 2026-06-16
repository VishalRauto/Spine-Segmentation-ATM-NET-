"""Build ATM_Net_PlusPlus_Kaggle.ipynb from kaggle_single_cell.py"""
import json
from pathlib import Path

BASE = Path(r"c:\project\Spine Segmentation\ATM-Net++")
code = (BASE / "kaggle_single_cell.py").read_text(encoding="utf-8")
lines = [line + "\n" for line in code.split("\n")]

install_cell = [
    "# Cell 1: Install PyTorch 2.3.1\n",
    "import subprocess, sys\n",
    "subprocess.run([sys.executable,'-m','pip','install','-q',\n",
    "    'torch==2.3.1','torchvision==0.18.1','torchaudio==2.3.1',\n",
    "    '--index-url','https://download.pytorch.org/whl/cu121'],timeout=600)\n",
    "subprocess.run([sys.executable,'-m','pip','install','-q',\n",
    "    'SimpleITK','opencv-python-headless'],timeout=120)\n",
    "import torch\n",
    "print(f'PyTorch: {torch.__version__}')\n",
    "assert torch.cuda.is_available(), 'No GPU - Settings > Accelerator > GPU T4'\n",
    "print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
    "print('Ready')\n",
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU"
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ATM-Net++ Spine Segmentation\n",
                "**2 cells only. Run All.**\n",
                "- Expected Dice: **0.85+** after 300 epochs\n",
                "- Download `best_model.pth` from Output tab when done"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": install_cell
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        }
    ]
}

out = BASE / "ATM_Net_PlusPlus_Kaggle.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

size_kb = out.stat().st_size // 1024
print(f"Written: {out}")
print(f"Size   : {size_kb} KB")
print(f"Cells  : {len(nb['cells'])} (markdown + install + training)")
print("Done")
