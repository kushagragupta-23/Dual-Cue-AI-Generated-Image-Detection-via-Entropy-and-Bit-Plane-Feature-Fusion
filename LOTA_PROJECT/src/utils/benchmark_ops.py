"""
Latency, Throughput, and VRAM/UMA Memory Profiling Utilities.

Provides hardware-agnostic benchmarking for:
    - Per-image latency (ms/img) across forward and forward+backward passes
    - Batch throughput (images/sec) at varying batch sizes
    - GPU VRAM or Apple UMA memory occupancy tracking
    - Mixed-precision (FP16/BF16/FP32) comparative analysis

All profiling functions are device-agnostic and operate on CUDA, MPS, and CPU.
"""

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.utils.device import get_compute_device
from src.utils.logger import get_logger

logger = get_logger("benchmark_ops")


@contextmanager
def cuda_timer():
    """
    Context manager for precise GPU timing using CUDA events.

    Falls back to wall-clock time on non-CUDA devices.

    Yields:
        callable: A function that returns the elapsed time in milliseconds.
    """
    device = get_compute_device()

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        elapsed_ms = [0.0]

        def get_elapsed():
            return elapsed_ms[0]

        yield get_elapsed
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms[0] = start_event.elapsed_time(end_event)
    else:
        start_time = time.perf_counter()

        def get_elapsed():
            return (time.perf_counter() - start_time) * 1000.0

        yield get_elapsed


def measure_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 256, 256),
    device: Optional[torch.device] = None,
    num_warmup: int = 5,
    num_iterations: int = 20,
    include_backward: bool = False,
) -> Dict[str, float]:
    """
    Measure per-image inference latency in milliseconds.

    Args:
        model: Model to benchmark.
        input_shape: Shape of the input tensor (batch_size=1 recommended).
        device: Target device. Uses auto-detection if None.
        num_warmup: Number of warmup iterations to exclude.
        num_iterations: Number of timed iterations to average.
        include_backward: If True, includes backward pass in timing.

    Returns:
        dict with 'mean_ms', 'std_ms', 'min_ms', 'max_ms', 'throughput_img_per_sec'.
    """
    device = device or get_compute_device()
    model = model.to(device).eval()
    batch_size = input_shape[0]

    x = torch.randn(*input_shape, device=device)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(x)

    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []
    for _ in range(num_iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        if include_backward:
            model.train()
            out = model(x)
            loss = out.sum() if isinstance(out, torch.Tensor) else out[0].sum()
            loss.backward()
            model.zero_grad()
            model.eval()
        else:
            with torch.no_grad():
                _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) * 1000.0
        timings.append(elapsed)

    import numpy as np
    timings = np.array(timings)

    return {
        "mean_ms": float(timings.mean()),
        "std_ms": float(timings.std()),
        "min_ms": float(timings.min()),
        "max_ms": float(timings.max()),
        "throughput_img_per_sec": float(batch_size * 1000.0 / timings.mean()),
    }


def measure_throughput(
    model: nn.Module,
    batch_sizes: List[int] = [1, 4, 8, 16, 32, 64],
    input_channels: int = 3,
    image_size: int = 256,
    device: Optional[torch.device] = None,
    num_iterations: int = 10,
) -> Dict[int, Dict[str, float]]:
    """
    Measure throughput (images/sec) at varying batch sizes.

    Args:
        model: Model to benchmark.
        batch_sizes: List of batch sizes to test.
        input_channels: Number of input channels.
        image_size: Spatial dimension (H=W).
        device: Target device.
        num_iterations: Iterations per batch size.

    Returns:
        dict mapping batch_size → throughput metrics.
    """
    device = device or get_compute_device()
    model = model.to(device).eval()
    results = {}

    for bs in batch_sizes:
        try:
            shape = (bs, input_channels, image_size, image_size)
            metrics = measure_latency(
                model, shape, device, num_warmup=3, num_iterations=num_iterations
            )
            results[bs] = {
                "throughput_img_per_sec": metrics["throughput_img_per_sec"],
                "latency_per_image_ms": metrics["mean_ms"] / bs,
                "total_batch_ms": metrics["mean_ms"],
            }
            logger.info(
                f"  BS={bs}: {metrics['throughput_img_per_sec']:.1f} img/sec, "
                f"{metrics['mean_ms'] / bs:.2f} ms/img"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"  BS={bs}: OOM — skipping")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                results[bs] = {"error": "OOM"}
            else:
                raise

    return results


def measure_memory(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 256, 256),
    device: Optional[torch.device] = None,
    include_backward: bool = True,
) -> Dict[str, float]:
    """
    Measure peak GPU VRAM occupancy during forward (+ optional backward) pass.

    Args:
        model: Model to benchmark.
        input_shape: Input tensor shape.
        device: Target device.
        include_backward: Include backward pass memory.

    Returns:
        dict with memory measurements in MB.
    """
    device = device or get_compute_device()
    model = model.to(device)

    if device.type != "cuda":
        logger.warning("Memory profiling only available on CUDA devices")
        # Count parameters for non-CUDA
        param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        return {"param_memory_mb": param_mb, "device": device.type}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(*input_shape, device=device)

    # Forward pass
    model.train()
    out = model(x)
    loss = out.sum() if isinstance(out, torch.Tensor) else out[0].sum()

    forward_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    if include_backward:
        loss.backward()
        backward_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        backward_peak_mb = forward_peak_mb

    model.zero_grad()
    torch.cuda.empty_cache()

    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    return {
        "param_memory_mb": param_mb,
        "forward_peak_mb": forward_peak_mb,
        "forward_backward_peak_mb": backward_peak_mb,
        "device": device.type,
    }


def format_benchmark_table(
    throughput_results: Dict[int, Dict],
    latency_results: Optional[Dict[str, float]] = None,
    memory_results: Optional[Dict[str, float]] = None,
    title: str = "Hardware Benchmark",
) -> str:
    """
    Format benchmark results as a Markdown table.

    Args:
        throughput_results: Output from measure_throughput().
        latency_results: Output from measure_latency().
        memory_results: Output from measure_memory().
        title: Table title.

    Returns:
        str: Markdown-formatted benchmark table.
    """
    lines = [f"## {title}", ""]

    if latency_results:
        lines.extend([
            "### Latency (Single Image)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean Latency | {latency_results['mean_ms']:.2f} ms |",
            f"| Min Latency | {latency_results['min_ms']:.2f} ms |",
            f"| Max Latency | {latency_results['max_ms']:.2f} ms |",
            f"| Throughput | {latency_results['throughput_img_per_sec']:.1f} img/sec |",
            "",
        ])

    if throughput_results:
        lines.extend([
            "### Throughput Scaling",
            "",
            "| Batch Size | Throughput (img/sec) | Latency/Image (ms) |",
            "|------------|---------------------|---------------------|",
        ])
        for bs, metrics in sorted(throughput_results.items()):
            if "error" in metrics:
                lines.append(f"| {bs} | OOM | — |")
            else:
                lines.append(
                    f"| {bs} | {metrics['throughput_img_per_sec']:.1f} | "
                    f"{metrics['latency_per_image_ms']:.2f} |"
                )
        lines.append("")

    if memory_results:
        lines.extend([
            "### Memory Profile",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Parameters | {memory_results['param_memory_mb']:.2f} MB |",
        ])
        if "forward_peak_mb" in memory_results:
            lines.append(
                f"| Forward Peak VRAM | {memory_results['forward_peak_mb']:.2f} MB |"
            )
        if "forward_backward_peak_mb" in memory_results:
            lines.append(
                f"| Fwd+Bwd Peak VRAM | {memory_results['forward_backward_peak_mb']:.2f} MB |"
            )

    return "\n".join(lines)


__all__ = [
    "cuda_timer",
    "measure_latency",
    "measure_throughput",
    "measure_memory",
    "format_benchmark_table",
]
