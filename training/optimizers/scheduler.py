"""
Learning rate schedulers and optimizer utilities for ATM-Net++.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, OneCycleLR


def build_optimizer(model: torch.nn.Module, config: dict) -> Optimizer:
    """
    Build optimizer with layer-wise learning rate decay (LLRD).
    Backbone gets smaller LR than task heads.
    """
    opt_cfg = config.get("training", {}).get("optimizer", {})
    base_lr = float(opt_cfg.get("lr", 1e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-5))

    # Separate parameters into groups:
    # 1. Backbone (Swin encoder) — lower LR
    # 2. Text encoder — lower LR (mostly frozen)
    # 3. Task heads + fusion — base LR
    no_decay = {"bias", "LayerNorm.weight", "BatchNorm.weight",
                "norm.weight", "norm1.weight", "norm2.weight"}

    backbone_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = "image_encoder" in name or "swinViT" in name
        is_text = "text_encoder" in name
        if is_backbone or is_text:
            backbone_params.append((name, param))
        else:
            other_params.append((name, param))

    param_groups = [
        {
            "params": [p for n, p in backbone_params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": base_lr * 0.1,
        },
        {
            "params": [p for n, p in backbone_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": base_lr * 0.1,
        },
        {
            "params": [p for n, p in other_params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": base_lr,
        },
        {
            "params": [p for n, p in other_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": base_lr,
        },
    ]

    # Filter empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    name = opt_cfg.get("name", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
    elif name == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
    elif name == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(param_groups)
        except ImportError:
            return torch.optim.AdamW(param_groups)
    else:
        return torch.optim.AdamW(param_groups)


def build_scheduler(
    optimizer: Optimizer,
    config: dict,
    steps_per_epoch: int,
) -> object:
    """
    Build learning rate scheduler from config.
    Supports: cosine_warmup, onecycle, linear_warmup, cosine.
    """
    train_cfg = config.get("training", {})
    sched_cfg = train_cfg.get("scheduler", {})
    max_epochs = train_cfg.get("epochs", 200)
    name = sched_cfg.get("name", "cosine_warmup").lower()
    warmup_epochs = sched_cfg.get("warmup_epochs", 10)
    min_lr = float(sched_cfg.get("min_lr", 1e-6))

    if name == "cosine_warmup":
        def lr_lambda(current_step: int) -> float:
            warmup_steps = warmup_epochs * steps_per_epoch
            total_steps = max_epochs * steps_per_epoch
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(cosine_decay, min_lr / optimizer.param_groups[0]["lr"])

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif name == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=[g["lr"] * 10 for g in optimizer.param_groups],
            epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100,
        )

    elif name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=min_lr,
        )

    else:
        return CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=min_lr)


class GradientStats:
    """Track gradient norms during training for debugging."""

    def __init__(self):
        self._norms: List[float] = []

    def record(self, model: torch.nn.Module):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self._norms.append(total_norm ** 0.5)

    @property
    def mean_norm(self) -> float:
        return float(sum(self._norms) / len(self._norms)) if self._norms else 0.0

    @property
    def max_norm(self) -> float:
        return float(max(self._norms)) if self._norms else 0.0

    def reset(self):
        self._norms = []
