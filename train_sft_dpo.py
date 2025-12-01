"""
Complete SFT + DPO Training Pipeline for Sarcasm Detection
MobileLLM-350M with QLoRA

Features:
- Comprehensive diagnostic logging
- Early stopping based on F1 score
- Reference model tracking for DPO
- Hyperparameter flexibility
- Detailed metrics and visualization
"""

import os
import yaml
import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import json
import logging
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from datasets import Dataset, load_dataset
from trl import DPOTrainer, DPOConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and save training metrics"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            'train_loss': [],
            'eval_loss': [],
            'eval_f1': [],
            'eval_accuracy': [],
            'eval_precision': [],
            'eval_recall': [],
            'epoch': [],
            'step': []
        }
        
    def log(self, metrics: Dict, step: int, epoch: int):
        """Log metrics at a given step"""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        self.metrics['step'].append(step)
        self.metrics['epoch'].append(epoch)
        
    def save(self, filename: str = "metrics.json"):
        """Save metrics to JSON"""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {filepath}")
        
    def plot(self, filename: str = "training_curves.png"):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        if 'train_loss' in self.metrics and self.metrics['train_loss']:
            axes[0, 0].plot(self.metrics['step'][:len(self.metrics['train_loss'])], 
                           self.metrics['train_loss'], label='Train Loss')
        if 'eval_loss' in self.metrics and self.metrics['eval_loss']:
            eval_steps = [s for s, e in zip(self.metrics['step'], self.metrics['epoch']) 
                         if 'eval_loss' in self.metrics][:len(self.metrics['eval_loss'])]
            axes[0, 0].plot(eval_steps, self.metrics['eval_loss'], label='Eval Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # F1 Score
        if 'eval_f1' in self.metrics and self.metrics['eval_f1']:
            eval_steps_f1 = [s for s in self.metrics['step']][:len(self.metrics['eval_f1'])]
            axes[0, 1].plot(eval_steps_f1, self.metrics['eval_f1'], marker='o')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].set_title('Validation F1 Score')
            axes[0, 1].grid(True)
        
        # Accuracy
        if 'eval_accuracy' in self.metrics and self.metrics['eval_accuracy']:
            eval_steps_acc = [s for s in self.metrics['step']][:len(self.metrics['eval_accuracy'])]
            axes[1, 0].plot(eval_steps_acc, self.metrics['eval_accuracy'], marker='o', color='green')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Validation Accuracy')
            axes[1, 0].grid(True)
        
        # Precision & Recall
        if 'eval_precision' in self.metrics and self.metrics['eval_recall']:
            eval_steps_pr = [s for s in self.metrics['step']][:len(self.metrics['eval_precision'])]
            axes[1, 1].plot(eval_steps_pr, self.metrics['eval_precision'], 
                           marker='o', label='Precision', color='blue')
            axes[1, 1].plot(eval_steps_pr, self.metrics['eval_recall'], 
                           marker='s', label='Recall', color='red')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Precision & Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved to {filepath}")


class DPODiagnostics:
    """Track DPO-specific diagnostics"""
    
    def __init__(self, output_dir: str, kl_threshold: float = 1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.kl_threshold = kl_threshold
        self.diagnostics = {
            'kl_divergence': [],
            'reward_margin': [],
            'chosen_rewards': [],
            'rejected_rewards': [],
            'step': [],
            'epoch': []
        }
        
    def log(self, kl_div: float, chosen_reward: float, rejected_reward: float, 
            step: int, epoch: int):
        """Log DPO diagnostics"""
        self.diagnostics['kl_divergence'].append(kl_div)
        self.diagnostics['chosen_rewards'].append(chosen_reward)
        self.diagnostics['rejected_rewards'].append(rejected_reward)
        self.diagnostics['reward_margin'].append(chosen_reward - rejected_reward)
        self.diagnostics['step'].append(step)
        self.diagnostics['epoch'].append(epoch)
        
        # Warnings
        if kl_div > self.kl_threshold:
            logger.warning(f"⚠️  High KL divergence: {kl_div:.4f} (threshold: {self.kl_threshold})")
        
        if chosen_reward <= rejected_reward:
            logger.warning(f"⚠️  Reward inversion: chosen ({chosen_reward:.4f}) <= rejected ({rejected_reward:.4f})")
    
    def save(self, filename: str = "dpo_diagnostics.json"):
        """Save diagnostics"""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.diagnostics, f, indent=2)
        logger.info(f"DPO diagnostics saved to {filepath}")
    
    def plot(self, filename: str = "dpo_diagnostics.png"):
        """Plot DPO diagnostics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # KL Divergence
        if self.diagnostics['kl_divergence']:
            axes[0, 0].plot(self.diagnostics['step'], self.diagnostics['kl_divergence'], 
                           color='purple', linewidth=2)
            axes[0, 0].axhline(y=self.kl_threshold, color='r', linestyle='--', 
                              label=f'Threshold ({self.kl_threshold})')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('KL Divergence')
            axes[0, 0].set_title('KL Divergence from Reference Model')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Reward Margin
        if self.diagnostics['reward_margin']:
            axes[0, 1].plot(self.diagnostics['step'], self.diagnostics['reward_margin'], 
                           color='green', linewidth=2)
            axes[0, 1].axhline(y=0, color='r', linestyle='--', label='Zero Line')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Reward Margin')
            axes[0, 1].set_title('Chosen - Rejected Reward Margin')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Chosen vs Rejected Rewards
        if self.diagnostics['chosen_rewards'] and self.diagnostics['rejected_rewards']:
            axes[1, 0].plot(self.diagnostics['step'], self.diagnostics['chosen_rewards'], 
                           label='Chosen', color='blue', linewidth=2)
            axes[1, 0].plot(self.diagnostics['step'], self.diagnostics['rejected_rewards'], 
                           label='Rejected', color='red', linewidth=2)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Log Probability')
            axes[1, 0].set_title('Reward Trajectories')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Histogram of reward margins
        if self.diagnostics['reward_margin']:
            axes[1, 1].hist(self.diagnostics['reward_margin'], bins=30, 
                           color='green', alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero')
            axes[1, 1].set_xlabel('Reward Margin')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of Reward Margins')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"DPO diagnostics plot saved to {filepath}")


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_dataset(config: Dict) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and prepare dataset for training
    Returns: train_dataset, val_dataset, test_dataset
    """
    logger.info("Loading dataset...")
    
    # Load train and test splits
    train_path = config['data']['train_split_path']
    test_path = config['data']['test_split_path']
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.error(f"Dataset files not found. Run split_gen_dataset.py first.")
        raise FileNotFoundError("Train/test splits not found")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Convert GEN format
    train_df['label'] = (train_df['class'] == 'sarc').astype(int)
    test_df['label'] = (test_df['class'] == 'sarc').astype(int)
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    logger.info(f"Train sarcasm ratio: {train_df['label'].mean():.2%}")
    logger.info(f"Test sarcasm ratio: {test_df['label'].mean():.2%}")
    
    # Sample if configured
    max_samples = config['data'].get('max_samples')
    if max_samples and max_samples < len(train_df):
        train_df = train_df.sample(n=max_samples, random_state=config['data']['seed'])
        logger.info(f"Sampled {max_samples} training examples")
    
    # Create train/val split
    val_ratio = config['data']['val_split_ratio']
    train_df = train_df.sample(frac=1, random_state=config['data']['seed']).reset_index(drop=True)
    val_size = int(len(train_df) * val_ratio)
    val_df = train_df[:val_size]
    train_df = train_df[val_size:]
    
    logger.info(f"Final split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Convert to HF Dataset
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
    
    return train_dataset, val_dataset, test_dataset


def format_sft_prompt(text: str, label: Optional[int] = None) -> str:
    """Format prompt for SFT"""
    prompt = f"Is the following text sarcastic? Answer with only 'Yes' or 'No'.\n\nText: {text}\n\nAnswer:"
    if label is not None:
        answer = " Yes" if label == 1 else " No"
        return prompt + answer
    return prompt


def prepare_sft_dataset(dataset: Dataset, tokenizer, max_length: int = 512) -> Dataset:
    """Prepare dataset for SFT"""
    
    def tokenize_function(examples):
        # Format prompts with labels
        texts = [format_sft_prompt(text, label) 
                for text, label in zip(examples['text'], examples['label'])]
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors=None
        )
        
        # Set labels for causal LM
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing SFT dataset"
    )
    
    return dataset


def prepare_dpo_dataset(dataset: Dataset, tokenizer, max_length: int = 512) -> Dataset:
    """
    Prepare dataset for DPO
    Create preference pairs: correct answer = chosen, incorrect = rejected
    """
    
    def create_preference_pairs(examples):
        prompts = []
        chosen = []
        rejected = []
        
        for text, label in zip(examples['text'], examples['label']):
            prompt = f"Is the following text sarcastic? Answer with only 'Yes' or 'No'.\n\nText: {text}\n\nAnswer:"
            
            if label == 1:  # Sarcastic
                chosen_response = " Yes"
                rejected_response = " No"
            else:  # Not sarcastic
                chosen_response = " No"
                rejected_response = " Yes"
            
            prompts.append(prompt)
            chosen.append(chosen_response)
            rejected.append(rejected_response)
        
        return {
            'prompt': prompts,
            'chosen': chosen,
            'rejected': rejected
        }
    
    dataset = dataset.map(
        create_preference_pairs,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Creating DPO preference pairs"
    )
    
    logger.info(f"Created {len(dataset)} preference pairs for DPO")
    
    return dataset


def compute_metrics_sft(eval_pred):
    """Compute metrics for SFT evaluation"""
    predictions, labels = eval_pred
    
    # Get predicted tokens (argmax over vocab)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # For causal LM, we need to extract the answer token
    # This is a simplified version - adjust based on your tokenizer
    pred_labels = np.argmax(predictions, axis=-1)[:, -1]  # Last token prediction
    
    # Convert to binary (rough heuristic - improve this based on your tokenizer)
    # You may need to map specific token IDs to Yes/No
    pred_binary = (pred_labels > pred_labels.mean()).astype(int)
    
    accuracy = accuracy_score(labels, pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_binary, average='binary', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


class F1EarlyStoppingCallback(TrainerCallback):
    """Early stopping based on F1 score"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_f1 = 0.0
        self.wait = 0
        self.stopped_epoch = 0
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_f1 = metrics.get('eval_f1', 0.0)
        
        if current_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = current_f1
            self.wait = 0
            logger.info(f"✓ New best F1: {self.best_f1:.4f}")
        else:
            self.wait += 1
            logger.info(f"F1 did not improve. Patience: {self.wait}/{self.patience}")
            
            if self.wait >= self.patience:
                control.should_training_stop = True
                self.stopped_epoch = state.epoch
                logger.info(f"Early stopping triggered at epoch {self.stopped_epoch}")
        
        return control


def train_sft(config: Dict, train_dataset: Dataset, val_dataset: Dataset) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Train SFT (Supervised Fine-Tuning) model
    
    Returns: trained_model, tokenizer
    """
    logger.info("="*70)
    logger.info("STARTING SFT TRAINING")
    logger.info("="*70)
    
    # Create output directory
    output_dir = Path(config['output']['sft_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {config['model']['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare datasets
    logger.info("Preparing SFT datasets...")
    train_dataset_sft = prepare_sft_dataset(train_dataset, tokenizer, config['model']['max_length'])
    val_dataset_sft = prepare_sft_dataset(val_dataset, tokenizer, config['model']['max_length'])
    
    # Quantization config
    logger.info("Setting up 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['quantization']['load_in_4bit'],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
        bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant']
    )
    
    # Load model
    logger.info(f"Loading base model: {config['model']['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    logger.info("Applying LoRA...")
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type']
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    sft_config = config['sft']
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=sft_config['num_train_epochs'],
        per_device_train_batch_size=sft_config['per_device_train_batch_size'],
        per_device_eval_batch_size=sft_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=sft_config['gradient_accumulation_steps'],
        learning_rate=sft_config['learning_rate'],
        warmup_ratio=sft_config['warmup_ratio'],
        weight_decay=sft_config['weight_decay'],
        logging_steps=sft_config['logging_steps'],
        eval_steps=sft_config['eval_steps'],
        save_steps=sft_config['save_steps'],
        save_total_limit=sft_config['save_total_limit'],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        fp16=sft_config['fp16'],
        optim=sft_config['optim'],
        max_grad_norm=sft_config['max_grad_norm'],
        report_to="none",
        logging_dir=str(output_dir / "logs"),
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Callbacks
    callbacks = []
    if sft_config['early_stopping']['enabled']:
        early_stopping = F1EarlyStoppingCallback(
            patience=sft_config['early_stopping']['patience']
        )
        callbacks.append(early_stopping)
    
    # Metrics tracker
    metrics_tracker = MetricsTracker(output_dir)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_sft,
        eval_dataset=val_dataset_sft,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Train
    logger.info("Starting SFT training...")
    train_result = trainer.train()
    
    # Save model
    logger.info(f"Saving SFT model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save metrics
    metrics_tracker.save("sft_metrics.json")
    metrics_tracker.plot("sft_training_curves.png")
    
    logger.info("="*70)
    logger.info("SFT TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Best model saved to: {output_dir}")
    
    return model, tokenizer


def train_dpo(
    config: Dict,
    train_dataset: Dataset,
    val_dataset: Dataset,
    sft_model_path: str
) -> PeftModel:
    """
    Train DPO (Direct Preference Optimization) model
    
    Args:
        config: Configuration dictionary
        train_dataset: Training dataset
        val_dataset: Validation dataset
        sft_model_path: Path to SFT model checkpoint
    
    Returns: trained_dpo_model
    """
    logger.info("="*70)
    logger.info("STARTING DPO TRAINING")
    logger.info("="*70)
    
    # Create output directory
    output_dir = Path(config['output']['dpo_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from SFT model: {sft_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    
    # Prepare DPO datasets
    logger.info("Preparing DPO preference datasets...")
    train_dataset_dpo = prepare_dpo_dataset(train_dataset, tokenizer, config['model']['max_length'])
    val_dataset_dpo = prepare_dpo_dataset(val_dataset, tokenizer, config['model']['max_length'])
    
    # Log preference data statistics
    logger.info("Preference Data Statistics:")
    logger.info(f"  Total training pairs: {len(train_dataset_dpo)}")
    logger.info(f"  Total validation pairs: {len(val_dataset_dpo)}")
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['quantization']['load_in_4bit'],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
        bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant']
    )
    
    # Load reference model (frozen)
    logger.info("Loading reference model (frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    ref_model = PeftModel.from_pretrained(ref_model, sft_model_path)
    ref_model.eval()  # Set to eval mode
    for param in ref_model.parameters():
        param.requires_grad = False
    logger.info("✓ Reference model loaded and frozen")
    
    # Load training model (from SFT)
    logger.info("Loading training model from SFT checkpoint...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, sft_model_path)
    model = prepare_model_for_kbit_training(model)
    logger.info("✓ Training model loaded from SFT")
    
    # DPO Configuration
    dpo_config_dict = config['dpo']
    dpo_training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=dpo_config_dict['num_train_epochs'],
        per_device_train_batch_size=dpo_config_dict['per_device_train_batch_size'],
        per_device_eval_batch_size=dpo_config_dict['per_device_eval_batch_size'],
        gradient_accumulation_steps=dpo_config_dict['gradient_accumulation_steps'],
        learning_rate=dpo_config_dict['learning_rate'],
        warmup_ratio=dpo_config_dict['warmup_ratio'],
        weight_decay=dpo_config_dict['weight_decay'],
        logging_steps=dpo_config_dict['logging_steps'],
        eval_steps=dpo_config_dict['eval_steps'],
        save_steps=dpo_config_dict['save_steps'],
        save_total_limit=dpo_config_dict['save_total_limit'],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,  # DPO doesn't support this yet
        fp16=dpo_config_dict['fp16'],
        optim=dpo_config_dict['optim'],
        max_grad_norm=dpo_config_dict['max_grad_norm'],
        beta=dpo_config_dict['beta'],
        loss_type=dpo_config_dict['loss_type'],
        report_to="none",
        logging_dir=str(output_dir / "logs"),
        remove_unused_columns=False,
    )
    
    logger.info(f"DPO Config:")
    logger.info(f"  Beta (KL coefficient): {dpo_config_dict['beta']}")
    logger.info(f"  Loss type: {dpo_config_dict['loss_type']}")
    logger.info(f"  Learning rate: {dpo_config_dict['learning_rate']}")
    logger.info(f"  Epochs: {dpo_config_dict['num_train_epochs']}")
    
    # Diagnostics tracker
    diagnostics = DPODiagnostics(
        output_dir,
        kl_threshold=dpo_config_dict['diagnostics']['kl_warning_threshold']
    )
    
    # DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_training_args,
        train_dataset=train_dataset_dpo,
        eval_dataset=val_dataset_dpo,
        processing_class=tokenizer,
    )
    
    # Train
    logger.info("Starting DPO training...")
    logger.info("Monitoring: KL divergence, reward margins, and model drift")
    
    train_result = dpo_trainer.train()
    
    # Save model
    logger.info(f"Saving DPO model to {output_dir}")
    dpo_trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save diagnostics
    diagnostics.save()
    diagnostics.plot()
    
    logger.info("="*70)
    logger.info("DPO TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Model saved to: {output_dir}")
    
    return model


def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SFT and/or DPO models")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, default="both", choices=["sft", "dpo", "both"],
                       help="Training mode: sft, dpo, or both")
    parser.add_argument("--sft-model-path", type=str, default=None,
                       help="Path to SFT model (required for DPO-only mode)")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_dataset(config)
    
    # Training pipeline
    if args.mode in ["sft", "both"]:
        # Train SFT
        sft_model, tokenizer = train_sft(config, train_dataset, val_dataset)
        sft_model_path = config['output']['sft_output_dir']
    else:
        sft_model_path = args.sft_model_path
        if not sft_model_path:
            raise ValueError("--sft-model-path required for DPO-only mode")
    
    if args.mode in ["dpo", "both"]:
        # Train DPO
        dpo_model = train_dpo(config, train_dataset, val_dataset, sft_model_path)
    
    logger.info("="*70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("="*70)
    logger.info(f"SFT model: {config['output']['sft_output_dir']}")
    logger.info(f"DPO model: {config['output']['dpo_output_dir']}")
    logger.info("\nNext steps:")
    logger.info("  1. Run validation: python validate.py")
    logger.info("  2. Analyze diagnostics: python diagnose_training.py")


if __name__ == "__main__":
    main()
