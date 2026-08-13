"""
GPU Throughput Scaling Benchmarks for HydraFusion-Net.

Measures:
  - Inference latency (ms per image)
  - Throughput (images per second)
  - Peak VRAM usage (MB)
  - Scaling across batch sizes (1, 2, 4, 8, 16, 32)

Supports FP32, FP16, and BF16 precision modes.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("benchmark")


@dataclass
class BenchmarkResult:
    """Container for a single benchmark measurement."""
    batch_size: int
    precision: str
    latency_ms: float  # Average ms per image
    throughput_ips: float  # Images per second
    peak_vram_mb: float  # Peak GPU memory in MB
    num_warmup: int = 10
    num_iterations: int = 50


def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
    precisions: List[str] = ["fp32", "fp16"],
    image_size: int = 256,
    num_warmup: int = 10,
    num_iterations: int = 50,
) -> List[BenchmarkResult]:
    """
    Run throughput benchmarks across batch sizes and precision modes.

    Args:
        model: HydraFusion-Net model on the target device.
        batch_sizes: List of batch sizes to benchmark.
        precisions: List of precision modes ("fp32", "fp16", "bf16").
        image_size: Input image resolution.
        num_warmup: Number of warmup iterations (not timed).
        num_iterations: Number of timed iterations.

    Returns:
        List of BenchmarkResult for each (batch_size, precision) combination.
    """
    model.eval()
    results: List[BenchmarkResult] = []

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    for precision in precisions:
        for bs in batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            amp_dtype = dtype_map.get(precision, torch.float32)
            use_amp = precision != "fp32" and device.type == "cuda"

            # Create dummy input
            dummy_input = torch.randn(bs, 3, image_size, image_size, device=device)
            dummy_input = dummy_input * 255.0  # Match [0, 255] convention

            try:
                # Warmup
                with torch.no_grad():
                    for _ in range(num_warmup):
                        if use_amp:
                            with torch.amp.autocast("cuda", dtype=amp_dtype):
                                model(dummy_input, stage=2)
                        else:
                            model(dummy_input, stage=2)

                # Timed iterations
                torch.cuda.synchronize()
                t_start = time.perf_counter()

                with torch.no_grad():
                    for _ in range(num_iterations):
                        if use_amp:
                            with torch.amp.autocast("cuda", dtype=amp_dtype):
                                model(dummy_input, stage=2)
                        else:
                            model(dummy_input, stage=2)

                torch.cuda.synchronize()
                t_end = time.perf_counter()

                total_time = t_end - t_start
                total_images = num_iterations * bs
                latency_ms = (total_time / total_images) * 1000.0
                throughput = total_images / total_time
                peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

                result = BenchmarkResult(
                    batch_size=bs,
                    precision=precision,
                    latency_ms=round(latency_ms, 3),
                    throughput_ips=round(throughput, 1),
                    peak_vram_mb=round(peak_vram, 1),
                    num_warmup=num_warmup,
                    num_iterations=num_iterations,
                )
                results.append(result)

                logger.info(
                    f"  BS={bs:3d} | {precision:4s} | "
                    f"Latency={latency_ms:.2f} ms/img | "
                    f"Throughput={throughput:.1f} img/s | "
                    f"VRAM={peak_vram:.0f} MB"
                )

            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    f"  BS={bs:3d} | {precision:4s} | OOM — skipped"
                )
                torch.cuda.empty_cache()
                continue

    return results


def format_benchmark_table(results: List[BenchmarkResult]) -> str:
    """Format benchmark results as a Markdown table."""
    lines = [
        "### GPU Throughput Benchmarks\n",
        "| Batch Size | Precision | Latency (ms/img) | Throughput (img/s) | Peak VRAM (MB) |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        lines.append(
            f"| {r.batch_size} | {r.precision} | {r.latency_ms:.2f} | "
            f"{r.throughput_ips:.1f} | {r.peak_vram_mb:.0f} |"
        )
    return "\n".join(lines)
