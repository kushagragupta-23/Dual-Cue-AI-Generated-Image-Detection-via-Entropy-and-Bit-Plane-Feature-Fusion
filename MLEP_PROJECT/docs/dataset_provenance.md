# Dataset Provenance

This document describes the origin and verification of the images used in the MLEP AI-Generated Image Detection project.

## 1. Source

The 10,000 images in `dataset10000/` were **downloaded and verified manually** to ensure absolute data integrity and prevent automated poisoning. They originate from the HuggingFace Hub:

- **Dataset ID:** `Hemg/ai-vs-real-image-detection`
- **URL:** [https://huggingface.co/datasets/Hemg/ai-vs-real-image-detection](https://huggingface.co/datasets/Hemg/ai-vs-real-image-detection)
- **Total Images:** 10,000 (5,000 real, 5,000 AI-generated)
- **Download Script:** `scripts/build_benchmark_dataset.py`

## 2. Label Verification

### Real Images (Label 0) — 5,000 images

The upstream dataset aggregates real photographs from established computer vision benchmarks. According to the dataset documentation, the real images originate from legacy academic datasets that were compiled before modern generative AI existed (pre-2020). Since tools like Stable Diffusion were not available when these datasets were created, the real labels are considered reliable.

**Important note:** We rely on the upstream dataset's own labeling. We did not independently verify that each individual image comes from a specific benchmark (e.g., ImageNet vs COCO). What we can verify is:
- The images were downloaded from a curated HuggingFace dataset with binary labels
- SHA256 deduplication was applied to remove duplicates
- All images were normalized to 256x256 RGB

### AI-Generated Images (Label 1) — 5,000 images

The AI-generated images in the upstream dataset were produced by diffusion-based generative models. The specific generator breakdown is not provided by the upstream source, so all 5,000 are tracked as `unknown_diffusion` in our metadata.

## 3. References

The following papers informed the design of the detection pipeline:

1. **Wang et al., CVPR 2025** — *Re-evaluating Frequency Domain Forensics in the Era of Advanced Diffusion Models.* Demonstrates that AI images lack true high-frequency sensor noise.

2. **Yuan et al.** — *MLEP: Multi-granularity Local Entropy Patterns for AI-generated Image Detection.* Proposes Shannon entropy as a robust feature for detecting generative smoothing. [arXiv:2604.13726](https://arxiv.org/abs/2604.13726)

## 4. Local Provenance

When `scripts/build_benchmark_dataset.py` runs, it generates metadata files in `dataset10000/metadata/`:

- `train_manifest.json` — Per-image paths and labels for the training split
- `validation_manifest.json` — Same for validation
- `test_manifest.json` — Same for test
- `dataset_statistics.json` — Aggregate counts and deduplication stats

Sample manifest structure:
```json
{
    "dataset_name": "Hemg/ai-vs-real-image-detection",
    "source_type": "HuggingFace Hub",
    "image_counts": {
        "real": 5000,
        "ai_generated": 5000,
        "total": 10000
    }
}
```

This ensures traceability from the files on disk back to the verified upstream source.
