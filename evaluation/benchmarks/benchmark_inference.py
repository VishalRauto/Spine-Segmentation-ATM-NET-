"""
Inference speed benchmarking for ATM-Net++.
Measures latency and throughput on CPU/GPU.

Usage:
    python evaluation/benchmarks/benchmark_inference.py
    python evaluation/benchmarks/benchmark_inference.py --device cuda --img-size 512 --n-runs 100
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device",   default="auto")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--n-runs",   type=int, default=50)
    p.add_argument("--warmup",   type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


def benchmark_model(
    model: torch.nn.Module,
    device: torch.device,
    img_size: int = 512,
    n_runs: int = 50,
    warmup: int = 10,
    batch_size: int = 1,
) -> dict:
    model.eval()
    dummy = torch.randn(batch_size, 1, img_size, img_size, device=device)

    latencies = []

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    return {
        "mean_ms":   round(statistics.mean(latencies), 2),
        "std_ms":    round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 2),
        "min_ms":    round(min(latencies), 2),
        "max_ms":    round(max(latencies), 2),
        "p50_ms":    round(statistics.median(latencies), 2),
        "p95_ms":    round(sorted(latencies)[int(0.95 * len(latencies))], 2),
        "fps":       round(1000 / statistics.mean(latencies) * batch_size, 2),
        "n_runs":    n_runs,
        "batch_size": batch_size,
        "img_size":  img_size,
        "device":    str(device),
    }


def benchmark_memory(model: torch.nn.Module, device: torch.device, img_size: int) -> dict:
    result = {}
    if device.type != "cuda":
        result["note"] = "Memory profiling only available on CUDA"
        return result

    torch.cuda.reset_peak_memory_stats(device)
    dummy = torch.randn(1, 1, img_size, img_size, device=device)
    with torch.no_grad():
        _ = model(dummy)
    result["peak_vram_mb"] = round(torch.cuda.max_memory_allocated(device) / 1e6, 1)
    result["reserved_vram_mb"] = round(torch.cuda.memory_reserved(device) / 1e6, 1)
    return result


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total_params":     total,
        "trainable_params": trainable,
        "frozen_params":    frozen,
        "total_M":          round(total / 1e6, 2),
        "trainable_M":      round(trainable / 1e6, 2),
    }


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  ATM-Net++ Inference Benchmark")
    print(f"{'='*60}")
    print(f"  Device:     {device}")
    print(f"  Image size: {args.img_size}×{args.img_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Runs:       {args.n_runs} (+ {args.warmup} warmup)")
    print(f"{'='*60}\n")

    from models.atmnet_plus_plus import ATMNetPlusPlus
    model = ATMNetPlusPlus(
        img_size=(args.img_size, args.img_size),
        in_channels=1,
        num_seg_classes=20,
        feature_size=48,
        use_text=False,
        use_demographics=True,
        deep_supervision=False,
    ).to(device)

    # Parameter count
    params = count_parameters(model)
    print("Model Parameters:")
    for k, v in params.items():
        print(f"  {k:22s}: {v:,}" if isinstance(v, int) else f"  {k:22s}: {v}M")

    # Latency benchmark
    print("\nLatency Benchmark:")
    results = benchmark_model(
        model, device,
        img_size=args.img_size,
        n_runs=args.n_runs,
        warmup=args.warmup,
        batch_size=args.batch_size,
    )
    for k, v in results.items():
        print(f"  {k:20s}: {v}")

    # Memory benchmark
    print("\nMemory Usage:")
    mem = benchmark_memory(model, device, args.img_size)
    for k, v in mem.items():
        print(f"  {k:20s}: {v}")

    print(f"\n{'='*60}")
    print(f"  Summary: {results['mean_ms']:.1f}ms mean latency, {results['fps']:.1f} FPS")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
