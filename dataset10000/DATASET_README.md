# dataset10000: MLEP Benchmark Dataset

## Overview
A 10,000-image dataset for evaluating the MLEP AI-Generated Image Detection model. Contains 5,000 real photographs and 5,000 AI-generated images, split into Train (60%), Validation (20%), and Test (20%).

## Source
- **Upstream Repository:** `Hemg/ai-vs-real-image-detection` on HuggingFace Hub
- **Real Images (Label 0):** Photographs from established CV benchmarks (pre-2020 datasets)
- **AI-Generated Images (Label 1):** Diffusion model outputs (specific generator not tracked by upstream source)

## Folder Structure
```text
dataset10000/
├── train/
│   ├── real/          # 3000 images
│   └── fake/          # 3000 images
├── validation/
│   ├── real/          # 1000 images
│   └── fake/          # 1000 images
├── test/
│   ├── real/          # 1000 images
│   └── fake/          # 1000 images
└── metadata/
    ├── train_manifest.json
    ├── validation_manifest.json
    ├── test_manifest.json
    ├── dataset_statistics.json
    └── generator_statistics.json
```

## Split Statistics
- **Train:** 6,000 images (3,000 real, 3,000 fake)
- **Validation:** 2,000 images (1,000 real, 1,000 fake)
- **Test:** 2,000 images (1,000 real, 1,000 fake)
- **Total:** 10,000 images

## Preprocessing
All images underwent the following automated preprocessing:
1. Corrupted files removed
2. Converted to RGB (alpha channels dropped)
3. Resized to 256x256 using Lanczos resampling
4. SHA256 hashing to remove exact duplicates

## Usage
```python
from src.data.dataset import SharedImageDataset
from src.data.dataloader import create_dataloader

train_ds = SharedImageDataset(root_dir="dataset10000", split="train")
train_loader = create_dataloader(train_ds, batch_size=32, balanced_sampling=True)
```

Or run the full training pipeline directly:
```bash
python scripts/train.py --data_dir dataset10000
```
