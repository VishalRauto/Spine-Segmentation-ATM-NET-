"""
Pytest configuration and shared fixtures.
"""

import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import numpy as np
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (skip with -m 'not slow')")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_mri_slice():
    """512x512 synthetic T2 MRI slice."""
    np.random.seed(42)
    img = np.random.uniform(0, 1000, (512, 512)).astype(np.float32)
    # Add spine-like structure
    img[200:300, 220:280] += 500  # Vertebral body
    return img


@pytest.fixture
def sample_mask():
    """512x512 synthetic segmentation mask."""
    mask = np.zeros((512, 512), dtype=np.int64)
    mask[200:230, 220:280] = 7   # L4
    mask[240:270, 220:280] = 8   # L5
    mask[230:240, 220:280] = 16  # L4/L5 disc
    mask[270:280, 220:280] = 17  # L5/S1 disc
    return mask


@pytest.fixture
def model_small():
    """Small ATM-Net++ model for fast testing."""
    from models.atmnet_plus_plus import ATMNetPlusPlus
    return ATMNetPlusPlus(
        img_size=(128, 128),
        in_channels=1,
        num_seg_classes=5,
        feature_size=16,
        fusion_dim=64,
        use_text=False,
        use_demographics=True,
        deep_supervision=False,
    ).eval()
