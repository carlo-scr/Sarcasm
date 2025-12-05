# Sarcasm Detection with Small Language Models

[![CS229](https://img.shields.io/badge/Stanford-CS229-8C1515?style=flat-square)](https://cs229.stanford.edu/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Qwen](https://img.shields.io/badge/Qwen2.5-0.5B-6366F1?style=flat-square)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

> **CS229 Machine Learning Final Project** - Training efficient small language models for sarcasm detection through Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).

## 📊 Key Results

| Model | Accuracy | F1 Score | Latency (ms) | Parameters |
|-------|----------|----------|--------------|------------|
| **GPT-4** (zero-shot) | 79.7% | 78.8% | 631 ± 261 | ~1.76T |
| Qwen2.5-0.5B (zero-shot) | 49.6% | 42.2% | 140 ± 7 | 0.5B |
| **+ SFT** | 69.8% | 71.6% | 171 ± 12 | 0.5B |
| **+ DPO** | **73.8%** | **76.2%** | 165 ± 2 | 0.5B |

**Highlights:**
- 📈 **+24.2pp** accuracy improvement from base to DPO
- ⚡ **3.8× faster** inference than GPT-4
- 🎯 **96.6%** of GPT-4's F1 score with **3,500× fewer parameters**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Two-Phase Training Pipeline                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │   Base Model  │ ──▶ │   SFT Model   │ ──▶ │   DPO Model   │       │
│  │  Qwen2.5-0.5B │     │   + LoRA      │     │   + LoRA      │       │
│  └──────────────┘     └──────────────┘     └──────────────┘        │
│         │                     │                     │               │
│         ▼                     ▼                     ▼               │
│      49.6%                 69.8%                 73.8%              │
│                                                                      │
│  Phase 1: SFT on IAC-V2         Phase 2: DPO on iSarcasm            │
│  (4,000 samples)                (286 preference pairs)              │
│  Learning: Pattern recognition  Learning: Error correction           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
.
├── data/
│   ├── GEN-sarc-notsarc.csv          # IAC-V2 dataset (in-distribution)
│   ├── isarcasm2022.csv               # iSarcasm dataset (DPO preferences)
│   ├── SARC/                          # SARC dataset (out-of-distribution)
│   │   └── train-balanced-sarcasm.csv
│   └── splits/                        # Train/test splits
│       ├── gen_train.csv
│       └── gen_test.csv
├── models/
│   ├── sft/                           # SFT model checkpoint (LoRA)
│   └── qwen_dpo_mistakes/             # DPO model checkpoint (LoRA)
├── scripts/
│   ├── Qwen2.5/
│   │   ├── finetune_qwen.py           # SFT training script
│   │   ├── dpo_train_v2.py            # DPO training script
│   │   ├── evaluate_all_stages_qwen.py # IAC-V2 evaluation
│   │   ├── evaluate_all_stages_sarc.py # SARC evaluation
│   │   └── measure_latency.py         # Latency benchmarking
│   ├── evaluate_variance.py           # 5-run variance evaluation
│   └── create_splits.py               # Dataset splitting utilities
├── results/
│   ├── comparative_results.json       # IAC-V2 evaluation results
│   ├── comparative_results_sarc.json  # SARC evaluation results
│   └── latency_results.json           # Latency benchmarks
├── visualizations/
│   ├── results_visualization.ipynb    # Figure generation notebook
│   └── fig_*.pdf                      # Generated figures
├── notebooks/
│   └── train_colab.ipynb              # Google Colab training notebook
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or Apple Silicon Mac
- ~8GB GPU memory for training, ~4GB for inference

### Installation

```bash
# Clone the repository
git clone https://github.com/carlo-scr/Sarcasm.git
cd Sarcasm

# Create conda environment (recommended)
conda create -n sarcasm python=3.10
conda activate sarcasm

# Install dependencies
pip install -r requirements.txt
```

### Run Evaluation (Pre-trained Models)

```bash
# Evaluate all models on IAC-V2 test set
cd scripts/Qwen2.5
python evaluate_all_stages_qwen.py

# Evaluate on SARC (out-of-distribution)
python evaluate_all_stages_sarc.py
```

### Train from Scratch

```bash
# Phase 1: Supervised Fine-Tuning
cd scripts/Qwen2.5
python finetune_qwen.py

# Phase 2: Direct Preference Optimization
python dpo_train_v2.py
```

## 📈 Detailed Results

### In-Distribution (IAC-V2 Test Set, n=1000)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| GPT-4 | 79.7% | 82.9% | 75.0% | 78.8% |
| Base | 49.6% ± 2.9% | 49.5% | 36.8% | 42.2% ± 2.6% |
| SFT | 69.8% ± 2.7% | 67.6% | 76.0% | 71.6% ± 2.0% |
| **DPO** | **73.8% ± 2.9%** | 69.8% | 84.0% | **76.2% ± 2.5%** |

### Out-of-Distribution (SARC, n=1000)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Base | 51.0% | 51.1% | 47.2% | 49.1% ± 1.7% |
| SFT | 52.6% | 51.6% | 83.6% | 63.8% ± 2.2% |
| **DPO** | **56.4%** | 53.8% | 90.8% | **67.6% ± 2.2%** |

### Latency Comparison (per sample)

| Model | Mean (ms) | Std (ms) | Speedup vs GPT-4 |
|-------|-----------|----------|------------------|
| GPT-4 | 631.4 | 261.5 | 1.0× |
| Base Qwen | 140.0 | 7.4 | 4.5× |
| SFT | 170.6 | 11.9 | 3.7× |
| **DPO** | 165.0 | 2.2 | **3.8×** |

## 🔬 Methodology

### Phase 1: Supervised Fine-Tuning (SFT)

- **Dataset**: IAC-V2 (GEN-sarc-notsarc), 4,000 balanced samples
- **Method**: LoRA fine-tuning (rank=16, alpha=32)
- **Target modules**: q_proj, k_proj, v_proj, o_proj
- **Training**: 2 epochs, batch size 4, lr=2e-4

### Phase 2: Direct Preference Optimization (DPO)

- **Dataset**: iSarcasm2022 preference pairs (286 samples)
- **Method**: DPO on top of SFT checkpoint
- **Beta**: 0.1
- **Key insight**: Uses SFT model's mistakes to create preference pairs

### Preference Pair Generation

```
Chosen: Correct prediction with reasoning
Rejected: Common error pattern

Example:
Text: "Oh great, another meeting that could've been an email."
Chosen: "Yes" (correctly identifies sarcasm)
Rejected: "No" (common failure mode)
```

## 📊 Reproducing Results

### Generate Visualizations

```bash
# Open the visualization notebook
cd visualizations
jupyter notebook results_visualization.ipynb
```

This generates all paper figures:
- `fig_model_comparison.pdf` - Accuracy comparison
- `fig_f1_comparison.pdf` - F1 score comparison
- `fig_confusion_matrices.pdf` - 2×2 confusion matrix grid
- `fig_id_vs_ood.pdf` - Generalization analysis
- `fig_efficiency_frontier.pdf` - Accuracy vs latency plot

### Run Variance Analysis

```bash
# 5-run evaluation for error bars
cd scripts
python evaluate_variance.py       # IAC-V2
python evaluate_variance_sarc.py  # SARC
```

## 🗃️ Datasets

| Dataset | Size | Source | Use |
|---------|------|--------|-----|
| IAC-V2 (GEN) | 4,000 | [Internet Argument Corpus](https://nlds.soe.ucsc.edu/iac2) | SFT training + ID evaluation |
| iSarcasm | 4,014 | [iSarcasmEval](https://github.com/iabufarha/iSarcasmEval) | DPO preference pairs |
| SARC | 1M+ | [SARC](https://nlp.cs.princeton.edu/SARC/) | OOD evaluation |

## 📄 Model Checkpoints

Pre-trained LoRA adapters are available in the `models/` directory:

| Model | Path | Size |
|-------|------|------|
| SFT | `models/sft/` | ~5MB |
| DPO | `models/qwen_dpo_mistakes/` | ~5MB |

To load a model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "models/qwen_dpo_mistakes")
```

## 🔧 Configuration

### Training Hyperparameters

| Parameter | SFT | DPO |
|-----------|-----|-----|
| Learning rate | 2e-4 | 5e-5 |
| Batch size | 4 | 4 |
| Epochs | 2 | 3 |
| LoRA rank | 16 | 16 |
| LoRA alpha | 32 | 32 |
| Gradient accumulation | 4 | 4 |
| Max length | 256 | 256 |
| Beta (DPO only) | - | 0.1 |

### Hardware Requirements

| Task | GPU Memory | Time |
|------|------------|------|
| Inference | 2-4 GB | ~0.17s/sample |
| SFT Training | 6-8 GB | ~30 min |
| DPO Training | 6-8 GB | ~20 min |

## 🙏 Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen2.5) for the base model
- [iSarcasmEval](https://github.com/iabufarha/iSarcasmEval) for the iSarcasm dataset
- [TRL Library](https://github.com/huggingface/trl) for DPO implementation
- Stanford CS229 course staff

## 📝 License

This project is for educational purposes as part of Stanford CS229. The datasets are subject to their respective licenses.
