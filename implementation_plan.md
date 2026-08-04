# Goal Description
The objective is to completely pivot the project's architecture to focus **exclusively on the MLEP (Multi-granularity Local Entropy Patterns) branch**. 

I have just deeply researched Aishwarya's repository (`https://github.com/aishwaryanevrekar/...`). **She has 100% deleted MLEP from her repository.** Her `src/models` directory only contains `lota.py`, and her scripts are strictly LOTA-only. 

To perfectly mirror her repository setup and strictly divide the work, we must do the exact same thing in reverse: completely strip out the LOTA branch and keep 0% of her code in this repository.

## User Review Required
> [!WARNING]
> This is a destructive action that will permanently delete the LOTA architecture and its associated research implementations from the codebase, perfectly matching Aishwarya's setup. Please review the deletion list below and confirm you are ready to pivot to an MLEP-Only codebase.

## Proposed Changes

### Core Models
- **[DELETE]** `src/models/lota.py` (The entire LOTA extractor and MGPS scoring module).
- **[DELETE]** `src/models/fusion_head.py` (The Cross-Modal attention network used to fuse MLEP and LOTA).
- **[MODIFY]** `src/models/dual_cue_detector.py` -> Rename and rewrite as `mlep_detector.py`. It will only instantiate the `MLEPExtractor`, the ResNet-18 backbone, and a linear classifier.
- **[MODIFY]** `src/models/__init__.py` to remove LOTA and Fusion imports.
- **[MODIFY]** `src/models/backbones.py` to remove any LOTA references.

### Training Scripts
- **[DELETE]** `scripts/train_ablation.py` (No longer needed since there is only one model).
- **[MODIFY]** `scripts/train.py` to train the new standalone `MLEPDetector` instead of the `DualCueDetector`.

### Documentation & Cleanup
- **[MODIFY]** `README.md` to remove references to LOTA and reflect the new MLEP-centric architecture.
- **[COMMAND]** Run terminal commands to delete old output directories (`outputs/project_run_training`, `outputs/ablation_study`) that contain LOTA-polluted weights.

## Verification Plan
1. Delete and rewrite the targeted files.
2. Purge the old output directories.
3. Run `scripts/train.py`.
4. Verify the neural network trains smoothly using ONLY the Shannon Entropy (MLEP) features.
