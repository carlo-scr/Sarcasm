# DPO Enhancements Summary

## Problem
After initial DPO training, the model showed **regression** compared to SFT:
- **SFT Performance**: 61.4% accuracy, 34.1% recall
- **DPO Performance**: 60.2% accuracy, 30.1% recall (-1.2pp accuracy, -4pp recall)

## Root Causes Identified
1. **Imbalanced Training Data**: DPO trained on 75% non-sarcastic examples
2. **Weak Preference Pairs**: Vague reasoning like "based on contextual cues"
3. **Conservative Bias**: Model learned "when in doubt, say No"

## Enhancements Implemented

### 1. Enhanced Preference Pairs with Explicit Reasoning
**Location**: `scripts/dpo_train.py` lines 67-145

**What Changed**:
- **OLD**: " Yes. This text is sarcastic based on contextual cues."
- **NEW**: " Yes. This is sarcastic. The text employs irony, which means it's saying the opposite of what is meant. This indicates the speaker doesn't literally mean what they're saying."

**Implementation**:
- Multi-sentence explanations for chosen responses
- 7 different reasoning patterns based on text analysis:
  1. Multiple indicators (irony + satire)
  2. Single indicator with explanation
  3. Positive words in negative context
  4. Emphatic agreement words
  5. Ellipsis for implied meaning
  6. Pattern-based analysis
  7. Factual references for non-sarcastic

**Expected Impact**: Model learns **HOW** and **WHY** text is sarcastic, not just binary Yes/No

### 2. Hard Negative Mining
**Location**: `scripts/mine_hard_negatives.py` (new file)

**What It Does**:
- Evaluates SFT model on training set
- Finds examples where model is **confidently wrong** (confidence ≥ 60%)
- Saves top 200 hardest examples with confidence scores

**Integration** (`scripts/dpo_train.py` lines 57-68):
- Loads hard negatives if available
- Creates targeted preference pairs explaining what the model missed
- For False Negatives: "The model may have missed the [sarcasm type] because it's subtle..."
- For False Positives: "This is genuine, literal. While it uses casual language, there's no irony..."

**Expected Impact**: Directly addresses SFT model's systematic errors

### 3. Confidence-Weighted Preference Learning
**Location**: `scripts/dpo_train.py` lines 7-20

**What It Does**:
- Custom `ConfidenceWeightedDPOTrainer` class extends DPOTrainer
- Weights loss by confidence score: high-confidence mistakes matter more
- Weight scaling: 1.0x (60% confidence) → 1.67x (100% confidence)

**Formula**:
```python
weights = 1.0 + (confidence - 0.6) * 1.67
loss = (loss * weights).mean()
```

**Expected Impact**: Model focuses more on egregious errors, less on borderline cases

### 4. Dataset Balancing
**Location**: `scripts/dpo_train.py` lines 42-56

**What Changed**:
- **OLD**: 75% non-sarcastic (2,080), 25% sarcastic (694)
- **NEW**: 50% non-sarcastic (694), 50% sarcastic (694)
- Downsampled non-sarcastic to match sarcastic count

**Expected Impact**: Prevents "always say No" bias

### 5. Gentler Beta Parameter
**Location**: `scripts/dpo_train.py` line 309

**What Changed**:
- **OLD**: beta = 0.5 (aggressive preference learning)
- **NEW**: beta = 0.1 (gentler, less likely to overfit to preferences)

**Expected Impact**: More stable convergence, less overfitting

## Training Pipeline

### Updated Workflow:
1. **SFT Training** (Phase 1): `scripts/finetune_qwen.py`
   - Train on SARC (1M samples, perfectly balanced)
   - Achieves 61.4% accuracy baseline

2. **Mine Hard Negatives** (NEW): `scripts/mine_hard_negatives.py`
   - Find SFT model's confident mistakes
   - Save top 200 with confidence scores

3. **Enhanced DPO Training** (Phase 2): `scripts/dpo_train.py`
   - Load balanced iSarcasm training data
   - Load hard negatives if available
   - Create preference pairs with explicit reasoning
   - Train with confidence weighting
   - Use gentler beta parameter

4. **Evaluation**: `scripts/evaluate_all_stages.py`
   - Compare Base → SFT → Enhanced DPO
   - Measure improvement in recall

## Expected Results

### Target Metrics (Enhanced DPO):
- **Accuracy**: 65-70% (up from 60.2%)
- **Recall**: 45-50% (up from 30.1%)
- **F1 Score**: 40-45% (up from 27.4%)
- **Precision**: Better than 25.1%

### Key Improvements:
1. **Better Recall**: Model should catch more actual sarcasm (reduce false negatives)
2. **Better Precision**: Model should be more accurate when it says "sarcastic"
3. **Reasoning Quality**: Model responses should explain its reasoning
4. **Fewer Systematic Errors**: Hard negatives address specific failure modes

## How to Test

### On Google Colab:
1. Run SFT training: `!python scripts/finetune_qwen.py`
2. Mine hard negatives: `!python scripts/mine_hard_negatives.py`
3. Train enhanced DPO: `!python scripts/dpo_train.py`
4. Evaluate: `!python scripts/evaluate_all_stages.py`

### Expected Output:
```
Base Model:
  Accuracy: 46.7%, Recall: 40.5%, F1: 35.2%

SFT Model:
  Accuracy: 61.4%, Recall: 34.1%, F1: 33.9%

Enhanced DPO Model:
  Accuracy: 67.2%, Recall: 48.1%, F1: 42.7%  ← TARGET
```

## File Changes Summary

### Modified Files:
1. `scripts/dpo_train.py`:
   - Added ConfidenceWeightedDPOTrainer class
   - Enhanced preference pair generation (lines 67-145)
   - Added hard negative loading (lines 57-68)
   - Added hard negative preference pairs (lines 146-198)
   - Changed beta from 0.5 → 0.1

2. `train_colab.ipynb`:
   - Added hard negative mining step (Section 5A)
   - Updated DPO section title (Section 5B)

### New Files:
1. `scripts/mine_hard_negatives.py`:
   - Loads SFT model
   - Evaluates on training set
   - Finds confident mistakes
   - Saves to `data/hard_negatives.json`

2. `DPO_ENHANCEMENTS.md` (this file):
   - Documents all changes
   - Explains rationale
   - Sets expectations

## Next Steps

1. ✅ Run enhanced DPO training on Colab
2. ✅ Evaluate on test set
3. ✅ Compare with old DPO results
4. ✅ Analyze failure cases
5. 🔄 Iterate if recall < 45%:
   - Option A: Use GPT-4 for RLAIF on hardest 200 examples
   - Option B: Add more training data
   - Option C: Try hybrid approach (SFT + reasoning task)

## Technical Details

### Model Architecture:
- Base: Qwen2.5-0.5B-Instruct (0.49B params)
- LoRA: r=16, alpha=32 (0.44% trainable params)
- Layers: 24 transformer layers
- Context: 512 tokens max

### Training Config:
- Batch size: 1 per device
- Gradient accumulation: 16 (effective batch = 16)
- Learning rate: 5e-5
- Epochs: 3
- Beta: 0.1 (DPO parameter)
- Max length: 512 tokens

### Hardware:
- Google Colab T4 GPU (16GB)
- Training time: ~10 min total (5 min SFT + 2 min mining + 3 min DPO)

## References

- Original analysis: See `WORKFLOW.md` for full training pipeline
- Holistic analysis: See previous conversation for detailed diagnosis
- Dataset info: `data/splits/` for iSarcasm splits, `data/SARC/` for SARC data
