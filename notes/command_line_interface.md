# Command-Line Interface for single_dataset_ablation.py

**Date**: 21 February 2026  
**Purpose**: Enable selective execution of experiment groups to manage large dataset (05_online_payment.csv: 6.3M rows, 481MB) resource usage

## AI-Assisted Modifications

The following improvements to `scripts/single_dataset_ablation.py` were implemented with AI assistance (GitHub Copilot) to address memory and runtime constraints:

### Changes Made

1. **Added argparse for command-line control**
   - Location: Lines 1, 59-86
   - Added `--experiment` flag with choices: oversampling, model, anomaly, calibration, pairwise, all
   - **Supports multiple experiments**: Can specify multiple experiment groups in one command
   - Added `--skip-significance` flag to skip statistical testing
   - Default behavior: runs all experiments (backward compatible)

2. **Refactored experiment generation**
   - Location: Lines 88-92, 237-266
   - Changed from sequential lists to dictionary-based conditional loading
   - Structure: `experiments_to_run = {}`
   - Expands "all" to full experiment list: `["oversampling", "model", "anomaly", "calibration", "pairwise"]`
   - Only loads experiments specified by `--experiment` flag

3. **Updated experiment execution loop**
   - Location: Lines 308-377
   - Changed from sequential execution to dictionary-based iteration
   - Collects results in `results_dfs` dictionary keyed by experiment type
   - Progress logging for each experiment group

4. **Fixed significance testing**
   - Location: Lines 379-424
   - All 4 significance sections now properly check if results exist before running
   - Proper indentation and variable references throughout
   - Only runs on experiment types that were executed
   - Fixed pairwise result capture (was missing in original)

## Usage Examples

### Run single experiment groups (recommended for large datasets)
```bash
# Run only oversampling experiments (8 experiments)
python scripts/single_dataset_ablation.py --experiment oversampling

# Run model comparison experiments (5 experiments)
python scripts/single_dataset_ablation.py --experiment model

# Run anomaly detection experiments (3 experiments)
python scripts/single_dataset_ablation.py --experiment anomaly

# Run calibration experiments (3 experiments)
python scripts/single_dataset_ablation.py --experiment calibration

# Run pairwise combinations (40 experiments)
python scripts/single_dataset_ablation.py --experiment pairwise
```

### Run multiple experiment groups in one command (NEW)
```bash
# Run oversampling AND model experiments (13 total)
python scripts/single_dataset_ablation.py --experiment oversampling model

# Run anomaly, calibration, and pairwise (46 total)
python scripts/single_dataset_ablation.py --experiment anomaly calibration pairwise

# Run all experiments except pairwise (19 total)
python scripts/single_dataset_ablation.py --experiment oversampling model anomaly calibration
```

### Skip significance testing to save time
```bash
# Run experiments but skip statistical testing
python scripts/single_dataset_ablation.py --experiment oversampling --skip-significance

# Multiple experiments without significance testing
python scripts/single_dataset_ablation.py --experiment oversampling model --skip-significance
```

### Run all experiments (original behavior)
```bash
# Runs all 59 experiments with significance testing
python scripts/single_dataset_ablation.py --experiment all

# Or simply (default behavior):
python scripts/single_dataset_ablation.py
```

## Performance Benefits

**For 05_online_payment.csv (6.3M rows)**:
- Breaking experiments into groups allows memory to be released between runs
- Each group can run in separate sessions (e.g., run overnight one group at a time)
- Expected memory usage: ~10-15GB per experiment group vs ~20GB for all experiments
- Estimated runtime per group: 1-3 hours vs 10+ hours for all

## Experiment Breakdown

| Flag | Count | Description |
|------|-------|-------------|
| `oversampling` | 8 | None, SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, KMeansSMOTE, SMOTEN, SMOTENC |
| `model` | 5 | LogisticRegression, DecisionTree, XGBoost, RandomForest, GradientBoosting |
| `anomaly` | 3 | None, IsolationForest, LOF |
| `calibration` | 3 | None, isotonic, platt |
| `pairwise` | 40 | 10 oversampling methods × 4 GAN types (CTGAN, CondWGAN-GP, SDV-CTGAN, CopulaGAN) |
| **Total** | **59** | All experiments |

## Technical Notes

- **Backward compatibility**: Script runs identically to original when called without arguments
- **Multi-experiment support**: Can specify multiple experiment groups in one command (e.g., `--experiment oversampling model anomaly`)
- **Environment variables**: Core configuration (GAN parameters, seeds, etc.) still managed via `.env` files
- **Temporal methodology**: All temporal split settings preserved (no shuffle, forward-chaining)
- **Results structure**: Each experiment group saves separate JSON configs and CSV results
- **Significance testing**: Automatically adapts to only test experiments that were run
- **Pairwise results**: Now properly captured (fixed bug in original script)

## Lecturer Documentation

This modification was made with AI assistance to address practical constraints while maintaining scientific rigor:

1. **Problem**: 6.3M row dataset causing excessive memory usage (20GB+) and runtime (10+ hours)
2. **Solution**: Command-line interface to split 59 experiments into manageable groups
3. **Enhancement**: Multi-experiment support allows flexible combinations (e.g., run failed experiments together)
4. **Validation**: All experiment logic, bootstrap samples, and temporal methodology unchanged
5. **Tool used**: GitHub Copilot (Claude Sonnet 4.5)
6. **Human oversight**: Code reviewed and tested before integration
7. **Integration**: Ported to professor's refactored `single_dataset_ablation.py` with .env config system

The core experimental methodology remains identical; only the execution scheduling has been made flexible.
