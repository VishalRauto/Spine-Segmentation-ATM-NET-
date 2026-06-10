"""
Explainability Module for ATM-Net++.

Implements:
- Grad-CAM for segmentation and classification
- Attention rollout for Transformer layers
- Segmentation overlay visualization
- Disease localization heatmaps
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).
    Works for both classification and segmentation models.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._handles: List = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        seg_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Args:
            input_tensor: (1, C, H, W) input image
            target_class: Class index to explain. If None, uses argmax.
            seg_mask: If provided, use segmentation output for backward pass.

        Returns:
            (H, W) float32 heatmap in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # Select target
        if seg_mask is not None:
            # For segmentation: sum activations of target class region
            seg_logits = output.get("seg_logits", output) if isinstance(output, dict) else output
            if target_class is not None:
                score = seg_logits[:, target_class, :, :].mean()
            else:
                score = seg_logits.max(dim=1)[0].mean()
        else:
            # For classification
            if isinstance(output, dict):
                logits = output.get("disease", {}).get("logits",
                         output.get("logits", torch.zeros(1, 7)))
            else:
                logits = output
            if target_class is None:
                target_class = logits.argmax(dim=-1).item()
            score = logits[0, target_class]

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self._gradients is None or self._activations is None:
            logger.warning("GradCAM: No gradients/activations captured.")
            h, w = input_tensor.shape[2:]
            return np.zeros((h, w), dtype=np.float32)

        # Pool gradients over spatial dims
        gradients = self._gradients  # (1, C, H', W') or (1, C, L)
        activations = self._activations  # (1, C, H', W')

        if gradients.dim() == 4:
            weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        elif gradients.dim() == 3:
            # Transformer attention (1, L, C) format
            weights = gradients.mean(dim=1, keepdim=True)
            activations = activations.unsqueeze(-1)
            weights = weights.unsqueeze(-1)
        else:
            weights = gradients.mean(dim=-1, keepdim=True)
            activations = activations.unsqueeze(-1)
            weights = weights.unsqueeze(-1)

        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Upsample to input size
        h, w = input_tensor.shape[2:]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def __del__(self):
        self.remove_hooks()


class AttentionRollout:
    """
    Attention rollout visualization for Transformer-based models.
    Computes the flow of attention from the last layer back to the input.
    (Abnar & Zuidema, 2020)
    """

    def __init__(self, model: nn.Module, attention_layer_pattern: str = "attn"):
        self.model = model
        self.attention_layer_pattern = attention_layer_pattern
        self._attention_maps: List[torch.Tensor] = []
        self._handles: List = []

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if self.attention_layer_pattern in name.lower():
                if hasattr(module, "attn_drop") or hasattr(module, "attention"):
                    handle = module.register_forward_hook(self._attention_hook)
                    self._handles.append(handle)

    def _attention_hook(self, module, input, output):
        if isinstance(output, tuple) and len(output) >= 2:
            attn = output[1]  # (B, heads, seq_len, seq_len)
            if attn is not None:
                self._attention_maps.append(attn.detach())

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate attention rollout map.

        Returns:
            (H, W) float32 attention map
        """
        self._attention_maps = []
        self._register_hooks()

        with torch.no_grad():
            _ = self.model(input_tensor)

        for h in self._handles:
            h.remove()
        self._handles = []

        if not self._attention_maps:
            h, w = input_tensor.shape[2:]
            return np.ones((h, w), dtype=np.float32) / (h * w)

        # Rollout computation
        result = self._attention_maps[0].mean(dim=1)  # avg over heads: (B, L, L)
        for attn in self._attention_maps[1:]:
            attn_avg = attn.mean(dim=1)
            # Add residual: A = A + I
            eye = torch.eye(attn_avg.shape[-1], device=attn_avg.device).unsqueeze(0)
            attn_avg = (attn_avg + eye) / 2
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            result = torch.bmm(attn_avg, result)

        # Take attention from CLS to all patches
        rollout = result[0, 0, 1:]  # (num_patches,)

        # Reshape to 2D
        h, w = input_tensor.shape[2:]
        num_patches = rollout.shape[0]
        patch_h = patch_w = int(num_patches ** 0.5)
        if patch_h * patch_w != num_patches:
            logger.warning(f"Non-square patch grid: {num_patches}")
            patch_h = patch_w = max(1, int(num_patches ** 0.5))

        rollout_2d = rollout[:patch_h * patch_w].reshape(patch_h, patch_w).cpu().numpy()
        rollout_2d = cv2.resize(rollout_2d, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize
        r_min, r_max = rollout_2d.min(), rollout_2d.max()
        if r_max > r_min:
            rollout_2d = (rollout_2d - r_min) / (r_max - r_min)

        return rollout_2d.astype(np.float32)


class ExplainabilityVisualizer:
    """
    Creates overlay visualizations for reports and dashboard.
    """

    @staticmethod
    def create_heatmap_overlay(
        image: np.ndarray,
        heatmap: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Overlay a heatmap on an image.

        Args:
            image: (H, W) or (H, W, 3) float32 or uint8
            heatmap: (H, W) float32 in [0, 1]
            colormap: OpenCV colormap
            alpha: Blend factor for heatmap

        Returns:
            (H, W, 3) uint8 overlay
        """
        # Convert image to uint8 RGB
        if image.dtype != np.uint8:
            img = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img = image.copy()

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)

        # Resize heatmap to match
        if heatmap.shape != img.shape[:2]:
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Apply colormap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap_rgb, alpha, 0)
        return overlay

    @staticmethod
    def create_segmentation_overlay(
        image: np.ndarray,
        mask: np.ndarray,
        colormap: Dict[int, Tuple[int, int, int]],
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay segmentation mask on image.

        Args:
            image: (H, W) float32 normalized image
            mask: (H, W) int64 class label mask
            colormap: {class_id: (R, G, B)} color dict
            alpha: Blend alpha

        Returns:
            (H, W, 3) uint8 overlay
        """
        # Convert image
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        if img_uint8.ndim == 2:
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_uint8.copy()

        # Build color mask
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in colormap.items():
            color_mask[mask == class_id] = color

        # Blend only non-background
        blend_mask = (mask > 0).astype(np.float32)[..., np.newaxis]
        overlay = (img_rgb * (1 - alpha * blend_mask) + color_mask * alpha * blend_mask)
        return overlay.astype(np.uint8)

    @staticmethod
    def encode_image_b64(image: np.ndarray) -> str:
        """Encode numpy image array to base64 string for API response."""
        import base64
        _, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                 if image.ndim == 3 else image)
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    @staticmethod
    def save_visualization(
        image: np.ndarray,
        heatmap: np.ndarray,
        mask: np.ndarray,
        output_path: str,
        colormap: Optional[Dict] = None,
    ) -> None:
        """Save a multi-panel visualization figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from datasets.preprocessing.label_mapper import ATMNET_COLORMAP

        if colormap is None:
            colormap = ATMNET_COLORMAP

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("MRI Input")
        axes[0].axis("off")

        # Grad-CAM
        overlay = ExplainabilityVisualizer.create_heatmap_overlay(image, heatmap)
        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        # Segmentation
        seg_overlay = ExplainabilityVisualizer.create_segmentation_overlay(image, mask, colormap)
        axes[2].imshow(seg_overlay)
        axes[2].set_title("Segmentation Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved visualization to {output_path}")
