# Sarcasm Detection Training Workflow

## Overview
Two-phase training pipeline for sarcasm detection using Qwen2.5-0.5B-Instruct:
1. **Phase 1 (SFT)**: Supervised Fine-Tuning on large SARC dataset
2. **Phase 2 (DPO)**: Direct Preference Optimization on iSarcasm dataset

## Data Setup

### 1. Create Train/Test Splits
```bash
python create_splits.py
```

This creates:
- `data/splits/isarcasm_train.csv` - 2,774 samples (80%) for DPO training
- `data/splits/isarcasm_test.csv` - 694 samples (20%) held out for evaluation

**Important**: Test set is never used during training to prevent data leakage.

## Training Pipeline

### Phase 1: Supervised Fine-Tuning (SFT)
```bash
python scripts/finetune_qwen.py
```

- Trains on SARC dataset (5,000 samples)
- Uses LoRA for efficient fine-tuning
- Output: `models/sft/`
- Time: ~30-60 minutes

### Phase 2: Direct Preference Optimization (DPO)
```bash
python scripts/dpo_train.py
```

- Trains on iSarcasm **training split only** (2,774 samples)
- Starts from SFT checkpoint
- Enhanced preference pairs with explicit reasoning
- Beta = 0.5 for strong preference learning
- Output: `models/dpo_enhanced/`
- Time: ~20-40 minutes

## Evaluation

### Evaluate All Models
```bash
python scripts/evaluate_all_stages.py
```

Evaluates three models on the **held-out test set**:
1. Base Model (zero-shot)
2. SFT Model (after Phase 1)
3. DPO Model (after Phase 2)

Output: `comparative_results.json`

## Expected Performance

- **Base Model**: ~49% accuracy (baseline)
- **SFT Model**: ~63% accuracy (+14 pts)
- **DPO Enhanced**: ~68% accuracy (+5 pts target)

Key metrics tracked: Accuracy, Precision, Recall, F1

## Data Flow

```
isarcasm2022.csv (3,468 samples)
         |
         v
create_splits.py
         |
         +---> isarcasm_train.csv (2,774) --> dpo_train.py --> models/dpo_enhanced/
         |
         +---> isarcasm_test.csv (694) --> evaluate_all_stages.py
```

## Files

- `create_splits.py` - Create train/test splits
- `scripts/finetune_qwen.py` - Phase 1 SFT training
- `scripts/dpo_train.py` - Phase 2 DPO training  
- `scripts/evaluate_all_stages.py` - Comparative evaluation
- `data/splits/` - Train/test split storage
- `models/` - Trained model storage
- `train_colab.ipynb` - Google Colab notebook for GPU trainingEADME.md` for details.

## Files

- `create_splits.py` - Create train/test splits
- `finetune_qwen.py` - Phase 1 SFT training
- `dpo_train.py` - Phase 2 DPO training  
- `evaluate_all_stages.py` - Comparative evaluation
- `data/splits/` - Train/test split storage
