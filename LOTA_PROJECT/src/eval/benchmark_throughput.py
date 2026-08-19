"""
GPU (FP16/BF16) and Apple Metal Throughput Scaling Benchmarks.

Provides automated cross-device throughput benchmarking:
    - Single-image latency profiling at FP32, FP16, BF16
    - Batch scaling curves (BS=1 to 64)
    - VRAM/UMA occupancy monitoring
    - Comparative Markdown report generation
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.utils.device import get_compute_device
from src.utils.benchmark_ops import (
    measure_latency,
    measure_throughput,
    measure_memory,
    format_benchmark_table,
)
from src.utils.logger import get_logger

logger = get_logger("benchmark_throughput")


class ThroughputBenchmarkSuite:
    """
    Automated throughput benchmarking across precision modes and batch sizes.

    Runs systematic benchmarks on a model:
        1. FP32 baseline latency and throughput
        2. FP16 AMP latency and throughput (CUDA only)
        3. BF16 AMP latency and throughput (CUDA only)
        4. Memory profiling at target batch size
        5. Comparative report generation

    Args:
        model: Model to benchmark.
        device: Compute device. Auto-detected if None.
        image_size: Input spatial size (default 256).
        input_channels: Number of input channels (default 3).
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        image_size: int = 256,
        input_channels: int = 3,
    ):
        self.model = model
        self.device = device or get_compute_device()
        self.image_size = image_size
        self.input_channels = input_channels

    def _make_amp_model(self, precision: str):
        """Create a wrapper that runs the model under AMP autocast."""
        model = self.model.to(self.device)
        device = self.device

        if precision == "fp32" or device.type != "cuda":
            return model

        dtype = torch.float16 if precision == "fp16" else torch.bfloat16

        class AMPWrapper(nn.Module):
            def __init__(self, inner_model, cast_dtype):
                super().__init__()
                self.inner = inner_model
                self.cast_dtype = cast_dtype

            def forward(self, x):
                with torch.amp.autocast(device_type="cuda", dtype=self.cast_dtype):
                    return self.inner(x)

        return AMPWrapper(model, dtype)

    def run_precision_benchmark(
        self,
        precisions: Optional[List[str]] = None,
        batch_size: int = 1,
        num_iterations: int = 20,
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark latency across multiple precision modes.

        Args:
            precisions: List of precision modes ('fp32', 'fp16', 'bf16').
            batch_size: Batch size for latency measurement.
            num_iterations: Number of iterations per precision.

        Returns:
            dict mapping precision → latency metrics.
        """
        if precisions is None:
            precisions = ["fp32"]
            if self.device.type == "cuda":
                precisions.extend(["fp16", "bf16"])

        results = {}
        input_shape = (batch_size, self.input_channels, self.image_size, self.image_size)

        for prec in precisions:
            logger.info(f"Benchmarking {prec.upper()} precision...")
            try:
                wrapped = self._make_amp_model(prec)
                metrics = measure_latency(
                    wrapped, input_shape, self.device,
                    num_iterations=num_iterations,
                )
                results[prec] = metrics
                logger.info(
                    f"  {prec.upper()}: {metrics['mean_ms']:.2f}ms, "
                    f"{metrics['throughput_img_per_sec']:.1f} img/sec"
                )
            except Exception as e:
                logger.warning(f"  {prec.upper()} benchmark failed: {e}")
                results[prec] = {"error": str(e)}

        return results

    def run_scaling_benchmark(
        self,
        batch_sizes: Optional[List[int]] = None,
        num_iterations: int = 10,
    ) -> Dict[int, Dict[str, float]]:
        """
        Benchmark throughput at varying batch sizes.

        Args:
            batch_sizes: Batch sizes to test.
            num_iterations: Iterations per batch size.

        Returns:
            dict mapping batch_size → throughput metrics.
        """
        batch_sizes = batch_sizes or [1, 4, 8, 16, 32, 64]
        logger.info(f"Scaling benchmark: batch sizes {batch_sizes}")

        return measure_throughput(
            self.model,
            batch_sizes=batch_sizes,
            input_channels=self.input_channels,
            image_size=self.image_size,
            device=self.device,
            num_iterations=num_iterations,
        )

    def run_memory_benchmark(
        self,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Profile peak memory at target batch size.

        Args:
            batch_size: Batch size for memory profiling.

        Returns:
            Memory metrics dict.
        """
        input_shape = (batch_size, self.input_channels, self.image_size, self.image_size)
        return measure_memory(self.model, input_shape, self.device)

    def run_full_benchmark(
        self,
        batch_sizes: Optional[List[int]] = None,
        precisions: Optional[List[str]] = None,
        num_iterations: int = 10,
    ) -> Dict[str, Dict]:
        """
        Run complete benchmark suite.

        Returns:
            dict with 'precision', 'scaling', 'memory' results.
        """
        return {
            "precision": self.run_precision_benchmark(precisions, num_iterations=num_iterations),
            "scaling": self.run_scaling_benchmark(batch_sizes, num_iterations=num_iterations),
            "memory": self.run_memory_benchmark(),
        }

    @staticmethod
    def compile_report(
        results: Dict[str, Dict],
        model_name: str = "DualCueAIGIDModel",
        save_path: Optional[Path] = None,
    ) -> str:
        """
        Compile benchmark results into a Markdown report.

        Args:
            results: Output from run_full_benchmark().
            model_name: Name for the report title.
            save_path: Optional path to save the report.

        Returns:
            str: Markdown report string.
        """
        lines = [f"# Throughput Benchmark Report: {model_name}", ""]

        # Precision comparison
        if "precision" in results:
            lines.extend([
                "## Precision Comparison",
                "",
                "| Precision | Mean Latency (ms) | Throughput (img/s) |",
                "|-----------|------------------|--------------------|",
            ])
            for prec, metrics in results["precision"].items():
                if "error" in metrics:
                    lines.append(f"| {prec.upper()} | Error: {metrics['error']} | — |")
                else:
                    lines.append(
                        f"| {prec.upper()} | {metrics['mean_ms']:.2f} | "
                        f"{metrics['throughput_img_per_sec']:.1f} |"
                    )
            lines.append("")

        # Scaling
        if "scaling" in results:
            table = format_benchmark_table(results["scaling"], title="Batch Scaling")
            lines.append(table)
            lines.append("")

        # Memory
        if "memory" in results:
            mem = results["memory"]
            lines.extend([
                "## Memory Profile",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Parameters | {mem.get('param_memory_mb', 0):.2f} MB |",
                f"| Device | {mem.get('device', 'unknown')} |",
            ])
            if "forward_peak_mb" in mem:
                lines.append(f"| Forward Peak VRAM | {mem['forward_peak_mb']:.2f} MB |")
            if "forward_backward_peak_mb" in mem:
                lines.append(f"| Fwd+Bwd Peak VRAM | {mem['forward_backward_peak_mb']:.2f} MB |")

        report = "\n".join(lines)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(report, encoding="utf-8")
            logger.info(f"Saved benchmark report to {save_path}")

        return report


__all__ = ["ThroughputBenchmarkSuite"]
