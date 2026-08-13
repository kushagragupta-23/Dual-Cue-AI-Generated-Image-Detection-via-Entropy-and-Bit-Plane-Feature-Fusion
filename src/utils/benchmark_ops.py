"""
Hardware Benchmarking Operations — Roadmap Alias.

Exposes benchmark_inference and BenchmarkResult.
"""

from src.eval.benchmark_throughput import benchmark_inference, BenchmarkResult, format_benchmark_table

__all__ = ["benchmark_inference", "BenchmarkResult", "format_benchmark_table"]
