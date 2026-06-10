"""
Label mapping for the SPIDER dataset to ATM-Net++ internal label space.

SPIDER Dataset Labels:
  Vertebrae: 1-19 (T1-T12=1-12, L1-L5=13-17, L6=18, S1=19)
    In practice for lumbar: T10=10, T11=11, T12=12, L1=13, L2=14, L3=15, L4=16, L5=17, S1=19
  Discs:   101-108 (T10/T11 through L5/S1)
  Canal:   201
  Spinal Cord: included in canal region or 202
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


# SPIDER vertebra label IDs (native)
SPIDER_VERTEBRA_LABELS = {
    1: "C1", 2: "C2", 3: "C3", 4: "C4", 5: "C5", 6: "C6", 7: "C7",
    8: "T1", 9: "T2", 10: "T3", 11: "T4", 12: "T5", 13: "T6",
    14: "T7", 15: "T8", 16: "T9", 17: "T10", 18: "T11", 19: "T12",
    20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5", 25: "S1",
    26: "S2",
}

# SPIDER disc label IDs (native)
SPIDER_DISC_LABELS = {
    101: "C2_C3", 102: "C3_C4", 103: "C4_C5", 104: "C5_C6",
    105: "C6_C7", 106: "C7_T1",
    107: "T1_T2", 108: "T2_T3", 109: "T3_T4", 110: "T4_T5",
    111: "T5_T6", 112: "T6_T7", 113: "T7_T8", 114: "T8_T9",
    115: "T9_T10", 116: "T10_T11", 117: "T11_T12",
    118: "T12_L1", 119: "L1_L2", 120: "L2_L3",
    121: "L3_L4", 122: "L4_L5", 123: "L5_S1",
}

SPIDER_CANAL_LABEL = 201
SPIDER_CORD_LABEL = 202

# ATM-Net++ internal class IDs (0-based, 0=background)
ATMNET_CLASSES = {
    0:  "background",
    1:  "T10",
    2:  "T11",
    3:  "T12",
    4:  "L1",
    5:  "L2",
    6:  "L3",
    7:  "L4",
    8:  "L5",
    9:  "S1",
    10: "T10_T11_disc",
    11: "T11_T12_disc",
    12: "T12_L1_disc",
    13: "L1_L2_disc",
    14: "L2_L3_disc",
    15: "L3_L4_disc",
    16: "L4_L5_disc",
    17: "L5_S1_disc",
    18: "spinal_canal",
    19: "spinal_cord",
}

NUM_CLASSES = len(ATMNET_CLASSES)


def build_spider_to_atmnet_mapping() -> Dict[int, int]:
    """
    Build pixel-value remapping from SPIDER native labels to ATM-Net++ class IDs.

    Returns:
        dict: {spider_label: atmnet_class_id}
    """
    mapping: Dict[int, int] = {}

    # Vertebrae of interest
    vert_map = {
        17: 1,   # T10 -> 1
        18: 2,   # T11 -> 2
        19: 3,   # T12 -> 3
        20: 4,   # L1  -> 4
        21: 5,   # L2  -> 5
        22: 6,   # L3  -> 6
        23: 7,   # L4  -> 7
        24: 8,   # L5  -> 8
        25: 9,   # S1  -> 9
    }
    mapping.update(vert_map)

    # Discs of interest
    disc_map = {
        116: 10,  # T10/T11 -> 10
        117: 11,  # T11/T12 -> 11
        118: 12,  # T12/L1  -> 12
        119: 13,  # L1/L2   -> 13
        120: 14,  # L2/L3   -> 14
        121: 15,  # L3/L4   -> 15
        122: 16,  # L4/L5   -> 16
        123: 17,  # L5/S1   -> 17
    }
    mapping.update(disc_map)

    # Spinal canal
    mapping[SPIDER_CANAL_LABEL] = 18

    # Spinal cord
    mapping[SPIDER_CORD_LABEL] = 19

    return mapping


SPIDER_TO_ATMNET: Dict[int, int] = build_spider_to_atmnet_mapping()

# Reverse mapping for visualization
ATMNET_TO_NAME: Dict[int, str] = ATMNET_CLASSES

# Color map for visualization (RGB)
ATMNET_COLORMAP: Dict[int, tuple] = {
    0:  (0,   0,   0),    # background - black
    1:  (255, 0,   0),    # T10        - red
    2:  (255, 85,  0),    # T11        - orange-red
    3:  (255, 170, 0),    # T12        - orange
    4:  (255, 255, 0),    # L1         - yellow
    5:  (170, 255, 0),    # L2         - yellow-green
    6:  (85,  255, 0),    # L3         - green
    7:  (0,   255, 0),    # L4         - bright green
    8:  (0,   255, 128),  # L5         - cyan-green
    9:  (0,   255, 255),  # S1         - cyan
    10: (0,   128, 255),  # T10/T11    - light blue
    11: (0,   0,   255),  # T11/T12    - blue
    12: (85,  0,   255),  # T12/L1     - blue-violet
    13: (170, 0,   255),  # L1/L2      - violet
    14: (255, 0,   255),  # L2/L3      - magenta
    15: (255, 0,   170),  # L3/L4      - pink-magenta
    16: (255, 0,   85),   # L4/L5      - hot pink
    17: (128, 0,   0),    # L5/S1      - dark red
    18: (200, 200, 200),  # spinal canal - light gray
    19: (100, 149, 237),  # spinal cord  - cornflower blue
}


def remap_spider_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remap a SPIDER segmentation mask to ATM-Net++ class IDs.

    Args:
        mask: Integer array with SPIDER native labels.

    Returns:
        Integer array with ATM-Net++ class IDs (0=background).
    """
    out = np.zeros_like(mask, dtype=np.int64)
    for src, dst in SPIDER_TO_ATMNET.items():
        out[mask == src] = dst
    return out


def create_colorized_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert ATM-Net++ class label map to an RGB image for visualization.

    Args:
        mask: (H, W) integer array with class IDs.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in ATMNET_COLORMAP.items():
        rgb[mask == class_id] = color
    return rgb


def get_class_weights(
    dataset_masks: List[np.ndarray],
    num_classes: int = NUM_CLASSES,
    smoothing: float = 0.1,
) -> np.ndarray:
    """
    Compute inverse-frequency class weights for handling class imbalance.

    Args:
        dataset_masks: List of mask arrays.
        num_classes: Total number of classes.
        smoothing: Label smoothing factor.

    Returns:
        (num_classes,) float32 weight array.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for mask in dataset_masks:
        for c in range(num_classes):
            counts[c] += np.sum(mask == c)

    # Avoid division by zero
    counts = np.maximum(counts, 1)
    freqs = counts / counts.sum()
    weights = 1.0 / (freqs + smoothing)
    weights = weights / weights.sum() * num_classes  # normalize
    return weights.astype(np.float32)
