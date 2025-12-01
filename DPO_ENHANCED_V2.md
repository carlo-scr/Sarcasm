# Enhanced DPO Training for Sarcasm Detection

## Overview

This enhanced pipeline fixes the DPO regression issue by using **different datasets** for SFT and DPO:
- **SFT (Phase 1)**: Trains on GEN dataset (80% split) to learn general sarcasm patterns
- **DPO (Phase 2)**: Trains on iSarcasm dataset to refine preferences with NEW data
- **Evaluation**: Tests on held-out GEN test set (20% split) for fair comparison

**Key Improvement**: Using iSarcasm for DPO instead of GEN training data provides new preference signals, preventing the confusion/overfitting that caused the previous 3.1pp accuracy drop.

**WandB Integration**: All scripts now support Weights & Biases for comprehensive experiment tracking, visualization, and collaboration.

## Enhanced Features

### 1. SFT Training (`finetune_qwen.py`)
- ✅ Comprehensive metrics tracking (Accuracy, Precision, Recall, F1)
- ✅ Training curves visualization
- ✅ Per-epoch evaluation with validation set
- ✅ Best model checkpoint saving
- ✅ Training summary JSON
- ✅ **WandB integration for real-time tracking**

### 2. DPO Training (`dpo_train_v2.py`)
- ✅ **Uses iSarcasm dataset** (KEY FIX: different data than SFT)
- ✅ KL divergence tracking from reference model
- ✅ Per-batch reward ratios (chosen/rejected)
- ✅ Model drift monitoring
- ✅ F1-based early stopping
- ✅ Variable beta hyperparameter testing
- ✅ Enhanced preference pairs with explicit reasoning
- ✅ Training curves (loss, KL, rewards, F1)
- ✅ Comprehensive diagnostics logging
- ✅ **WandB integration with hyperparameter sweeps**

### 3. Model Comparison (`compare_all_models.py`)
- ✅ Side-by-side evaluation of Base/SFT/DPO
- ✅ Per-class metrics (sarcastic vs non-sarcastic)
- ✅ Confusion matrices
- ✅ Improvement calculations
- ✅ Statistical visualizations
- ✅ Fair comparison on same held-out test set
- ✅ **WandB integration for result tracking**

## Workflow

### Setup WandB (Optional but Recommended)
```bash
# Install wandb (already in requirements.txt)
pip install wandb

# Login to WandB
wandb login

# Or disable WandB with --no_wandb flag
```

### Step 1: Train SFT Model (Phase 1)
```bash
cd scripts

# With WandB (recommended)
python finetune_qwen.py --wandb_project sarcasm-detection --epochs 2

# Without WandB
python finetune_qwen.py --no_wandb --epochs 2
```

**What it does:**
- Loads GEN training split (`data/splits/gen_train.csv`)
- Trains Qwen2.5-0.5B with LoRA (r=16, alpha=32)
- Saves to `models/sft/`
- Creates training curves: `models/sft/sft_training_curves.png`
- Logs metrics: `models/sft/sft_metrics.json`
- **Logs to WandB**: Real-time loss, accuracy, F1, precision, recall

**Expected Performance:** ~77% accuracy on validation set

### Step 2: Train DPO Model (Phase 2)
```bash
# Single beta value with WandB
python dpo_train_v2.py --sft_model models/sft --output_dir models/dpo_enhanced --beta 0.1 --epochs 3 --wandb_project sarcasm-detection

# Hyperparameter sweep (tests multiple beta values) with WandB
python dpo_train_v2.py --sft_model models/sft --output_dir models/dpo_enhanced --beta_sweep --epochs 3 --wandb_project sarcasm-detection

# Without WandB
python dpo_train_v2.py --no_wandb --beta 0.1 --epochs 3
```

**What it does:**
- Loads SFT model from `models/sft/`
- Loads **iSarcasm dataset** (NEW DATA, not used in SFT)
- Creates enhanced preference pairs with explicit reasoning
- Tracks KL divergence from reference model
- Saves to `models/dpo_enhanced/`
- Creates training curves: `models/dpo_enhanced/dpo_training_curves.png`
- Logs comprehensive metrics: `models/dpo_enhanced/dpo_epoch_metrics.json`
- **Logs to WandB**: KL divergence, reward ratios, F1, loss per epoch

**Key Parameters:**
- `--beta`: KL regularization strength (default: 0.1)
  - Lower (0.01-0.05): Gentler updates, stays close to SFT
  - Higher (0.1-0.5): Stronger preference learning, more deviation from SFT
- `--epochs`: Number of training epochs (default: 3)
- `--lr`: Learning rate (default: 5e-5)
- `--sample_size`: Number of iSarcasm samples to use (default: 4000)
- `--beta_sweep`: Run hyperparameter sweep across multiple beta values

**Expected Improvement:** +5-8pp accuracy over SFT (now that we're using different data!)

### Step 3: Compare All Models
```bash
# With WandB
python compare_all_models.py --test_csv ../data/splits/gen_test.csv --sft_model models/sft --dpo_model models/dpo_enhanced --sample_size 1000 --wandb_project sarcasm-detection

# Without WandB
python compare_all_models.py --no_wandb --test_csv ../data/splits/gen_test.csv --sft_model models/sft --dpo_model models/dpo_enhanced --sample_size 1000

# Skip base model evaluation for faster comparison
python compare_all_models.py --skip_base --wandb_project sarcasm-detection
```

**What it does:**
- Evaluates Base, SFT, and DPO on held-out GEN test set
- Calculates comprehensive metrics for each stage
- Creates comparison visualization: `results/model_comparison_plot.png`
- Saves detailed results: `results/model_comparison_results.json`
- **Logs to WandB**: All metrics, improvements, and comparison plots

**Output Includes:**
- Overall metrics (Accuracy, Precision, Recall, F1)
- Per-class breakdown (sarcastic vs non-sarcastic)
- Improvement calculations (Δ from base)
- Confusion matrices
- Statistical visualizations

## File Structure

```
scripts/
├── finetune_qwen.py           # Phase 1: SFT training (enhanced)
├── dpo_train_v2.py            # Phase 2: DPO training (comprehensive diagnostics)
├── compare_all_models.py      # Evaluation & comparison
├── split_gen_dataset.py       # Create GEN train/test splits
└── mine_hard_negatives.py     # Extract SFT mistakes for DPO

models/
├── sft/                       # Phase 1 output
│   ├── adapter_model.safetensors
│   ├── sft_metrics.json
│   └── sft_training_curves.png
└── dpo_enhanced/              # Phase 2 output
    ├── adapter_model.safetensors
    ├── dpo_epoch_metrics.json
    ├── dpo_batch_metrics.json
    ├── dpo_training_curves.png
    └── dpo_training_summary.json

data/splits/
├── gen_train.csv              # SFT training data (80%)
└── gen_test.csv               # Held-out test set (20%)

results/
├── model_comparison_results.json    # Detailed metrics
└── model_comparison_plot.png        # Visualization
```

## Key Insights

### Why DPO Was Failing Before
- **Problem**: DPO trained on same GEN data as SFT
- **Effect**: No new information → confusion/overfitting → 3.1pp accuracy drop
- **Root Cause**: Model already "knows" the training examples from SFT, so preference pairs don't provide meaningful refinement signals

### Why New Approach Works
- **Solution**: DPO uses iSarcasm dataset (different from GEN)
- **Effect**: New preference signals → genuine refinement → expected +5-8pp gain
- **Mechanism**: Model learns from new examples with explicit reasoning, improving generalization

### Diagnostic Benefits
1. **KL Divergence Tracking**: Monitor how far DPO deviates from SFT reference
   - Too high: Model forgetting SFT knowledge
   - Too low: Insufficient preference learning
   - Sweet spot: 0.5-2.0 KL units

2. **Reward Ratios**: Ensure chosen responses consistently preferred over rejected
   - Healthy: ratio > 1.2
   - Problem: ratio ≈ 1.0 (model not learning preferences)

3. **F1-Based Early Stopping**: Prevent overfitting on validation set
   - Stops when F1 plateaus or decreases
   - Saves best checkpoint

4. **Beta Hyperparameter Sweep**: Find optimal KL regularization
   - Tests: 0.01, 0.02, 0.05, 0.1, 0.5
   - Identifies best tradeoff between stability and preference learning

## Troubleshooting

### If DPO still underperforms:
1. Check KL divergence: Should be 0.5-2.0
   - Too high → decrease beta or learning rate
   - Too low → increase beta

2. Check reward ratios: Should be > 1.2
   - Low ratio → preferences too weak, increase beta or use better pairs

3. Check training curves:
   - Loss decreasing but F1 flat → overfitting, reduce epochs
   - Both loss and F1 improving → extend training

4. Try different beta values:
   - Run beta sweep: `python dpo_train_v2.py --beta_sweep`
   - Use best beta from sweep results

### If out of memory:
- Reduce `per_device_train_batch_size` in script
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Reduce `sample_size` parameter

## Expected Results

| Stage | Dataset | Accuracy | F1 Score | Improvement |
|-------|---------|----------|----------|-------------|
| Base | - | ~51% | ~50% | Baseline |
| SFT | GEN train | ~77% | ~76% | +26pp |
| DPO | iSarcasm | **~82-85%** | **~81-84%** | **+5-8pp** |

*Note: DPO improvement depends on beta tuning and epochs. Use beta sweep to optimize.*

## Command Reference

```bash
# Full pipeline from scratch (with WandB)
python split_gen_dataset.py                    # Create train/test splits
python finetune_qwen.py --wandb_project sarcasm-detection --epochs 2
python dpo_train_v2.py --beta_sweep --wandb_project sarcasm-detection
python compare_all_models.py --wandb_project sarcasm-detection

# Full pipeline without WandB
python split_gen_dataset.py
python finetune_qwen.py --no_wandb --epochs 2
python dpo_train_v2.py --no_wandb --beta 0.1 --epochs 3
python compare_all_models.py --no_wandb

# Quick DPO retrain with different beta
python dpo_train_v2.py --beta 0.05 --epochs 3 --output_dir models/dpo_beta_0.05 --wandb_project sarcasm-detection

# Evaluate specific DPO model
python compare_all_models.py --dpo_model models/dpo_beta_0.05 --skip_base --wandb_project sarcasm-detection
```

## WandB Features

### Real-Time Tracking
- **SFT Training**: Loss, accuracy, F1, precision, recall per epoch
- **DPO Training**: KL divergence, reward ratios, F1, loss per batch/epoch
- **Model Comparison**: Side-by-side metrics, improvements, confusion matrices

### Visualizations
- Training curves (loss, F1, accuracy)
- KL divergence tracking
- Reward ratio evolution
- Model comparison plots
- Confusion matrices

### Hyperparameter Sweeps
- Beta sweep results logged to WandB
- Easy comparison of different configurations
- Automatic best model selection based on F1

### Collaboration
- Share experiments with team
- Compare runs across different settings
- Track model lineage (SFT → DPO)

## Architecture Details

- **Base Model**: Qwen2.5-0.5B-Instruct (0.49B params, 24 layers)
- **Fine-tuning**: LoRA (r=16, alpha=32, dropout=0.05)
- **Trainable Params**: 0.44% of total (2.2M / 494M)
- **Target Modules**: q_proj, k_proj, v_proj, o_proj (attention layers)
- **Optimization**: AdamW, warmup + cosine schedule
- **Device Support**: Auto-detection (CUDA/MPS/CPU)

## References

- Original DPO Paper: Rafailov et al. (2023) - "Direct Preference Optimization"
- Qwen2.5 Model: Alibaba Cloud
- LoRA: Hu et al. (2021) - "Low-Rank Adaptation of Large Language Models"
