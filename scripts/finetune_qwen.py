"""
Fine-tune Qwen2.5-0.5B on GEN-sarc-notsarc dataset using LoRA (Phase 1: SFT).
This script trains on the GEN training split to learn general sarcasm patterns.
Phase 2 (DPO) will use different data (iSarcasm) for preference alignment.

Enhanced with:
- Comprehensive metrics tracking (F1, Precision, Recall)
- Training curves visualization
- Configurable hyperparameters
- Better logging and diagnostics
"""

import pandas as pd
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import wandb

class MetricsLogger:
    """Logger for tracking SFT training metrics and creating visualizations."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.history = []
        os.makedirs(output_dir, exist_ok=True)
    
    def log(self, epoch, metrics_dict):
        """Log metrics for an epoch."""
        entry = {'epoch': epoch, 'timestamp': datetime.now().isoformat(), **metrics_dict}
        self.history.append(entry)
        
        # Save to JSON
        with open(f"{self.output_dir}/sft_metrics.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Epoch {epoch} Summary:")
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
        print(f"{'='*70}\n")
    
    def plot_curves(self):
        """Create training curve visualizations."""
        if len(self.history) < 2:
            return
        
        epochs = [h['epoch'] for h in self.history]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss curves
        if 'train_loss' in self.history[0]:
            train_losses = [h.get('train_loss') for h in self.history]
            axes[0, 0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
        if 'eval_loss' in self.history[0]:
            eval_losses = [h.get('eval_loss') for h in self.history]
            axes[0, 0].plot(epochs, eval_losses, 'r-o', label='Eval Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('Loss', fontsize=11)
        axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1 Score
        if 'eval_f1' in self.history[0]:
            f1_scores = [h.get('eval_f1') for h in self.history]
            axes[0, 1].plot(epochs, f1_scores, 'g-o', label='F1 Score', linewidth=2)
            axes[0, 1].set_xlabel('Epoch', fontsize=11)
            axes[0, 1].set_ylabel('F1 Score', fontsize=11)
            axes[0, 1].set_title('Validation F1 Score', fontsize=12, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Precision & Recall
        if 'eval_precision' in self.history[0] and 'eval_recall' in self.history[0]:
            precisions = [h.get('eval_precision') for h in self.history]
            recalls = [h.get('eval_recall') for h in self.history]
            axes[1, 0].plot(epochs, precisions, 'b-o', label='Precision', linewidth=2)
            axes[1, 0].plot(epochs, recalls, 'r-o', label='Recall', linewidth=2)
            axes[1, 0].set_xlabel('Epoch', fontsize=11)
            axes[1, 0].set_ylabel('Score', fontsize=11)
            axes[1, 0].set_title('Precision & Recall', fontsize=12, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy
        if 'eval_accuracy' in self.history[0]:
            accuracies = [h.get('eval_accuracy') for h in self.history]
            axes[1, 1].plot(epochs, accuracies, 'purple', marker='o', label='Accuracy', linewidth=2)
            axes[1, 1].set_xlabel('Epoch', fontsize=11)
            axes[1, 1].set_ylabel('Accuracy', fontsize=11)
            axes[1, 1].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"{self.output_dir}/sft_training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved training curves to {plot_path}")

def load_and_prepare_data(csv_path, tokenizer, max_length=256, sample_size=None):
    """Load and prepare the dataset for training."""
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check dataset format
    if 'class' in df.columns and 'text' in df.columns:
        # GEN dataset format
        print("Detected GEN-sarc-notsarc dataset format")
        # Convert to standard format
        df['label'] = (df['class'] == 'sarc').astype(int)
        text_col = 'text'
        label_col = 'label'
    elif 'comment' in df.columns:
        # SARC dataset format
        text_col = 'comment'
        label_col = 'label'
        print("Detected SARC dataset format")
    elif 'tweet' in df.columns:
        # iSarcasm dataset format
        df = df.set_index(df.columns[0]) if df.columns[0] == 'Unnamed: 0' else df
        text_col = 'tweet'
        label_col = 'sarcastic'
        print("Detected iSarcasm dataset format")
    else:
        raise ValueError("Unknown dataset format")
    
    # Sample if specified
    if sample_size and len(df) > sample_size:
        # Sample while maintaining class balance
        sarc_df = df[df[label_col] == 1]
        notsarc_df = df[df[label_col] == 0]
        
        n_per_class = sample_size // 2
        sarc_sample = sarc_df.sample(n=min(n_per_class, len(sarc_df)), random_state=42)
        notsarc_sample = notsarc_df.sample(n=min(n_per_class, len(notsarc_df)), random_state=42)
        
        df = pd.concat([sarc_sample, notsarc_sample]).sample(frac=1, random_state=42)
        print(f"Sampled {len(df)} examples from dataset (balanced)")
    
    print(f"Total samples: {len(df)}")
    print(f"Sarcastic: {df[label_col].sum()}, Non-sarcastic: {len(df) - df[label_col].sum()}")
    
    # Create prompts with labels
    def create_training_prompt(row):
        text = row[text_col]
        label = "Yes" if row[label_col] == 1 else "No"
        
        prompt = f"""Is the following text sarcastic? Answer with only 'Yes' or 'No'.

Text: {text}

Answer: {label}"""
        return prompt
    
    df['text'] = df.apply(create_training_prompt, axis=1)
    
    # Split into train/validation (80/20)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Convert to HF Dataset
    train_dataset = Dataset.from_pandas(train_df[['text']])
    val_dataset = Dataset.from_pandas(val_df[['text']])
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    return train_dataset, val_dataset


def compute_metrics_sft(eval_pred, tokenizer):
    """
    Compute comprehensive metrics for SFT evaluation.
    Extracts predictions from generated text and compares with labels.
    """
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    # predictions shape: (batch_size, seq_length)
    # We need to extract the sarcastic/not_sarcastic classification from generated text
    
    decoded_preds = []
    decoded_labels = []
    
    for pred_ids, label_ids in zip(predictions, labels):
        # Skip padding tokens
        pred_ids = pred_ids[pred_ids != -100]
        label_ids = label_ids[label_ids != -100]
        
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).lower()
        label_text = tokenizer.decode(label_ids, skip_special_tokens=True).lower()
        
        # Extract classification from text
        pred_class = 1 if 'sarcastic' in pred_text and 'not_sarcastic' not in pred_text else 0
        label_class = 1 if 'sarcastic' in label_text and 'not_sarcastic' not in label_text else 0
        
        decoded_preds.append(pred_class)
        decoded_labels.append(label_class)
    
    # Calculate metrics
    accuracy = accuracy_score(decoded_labels, decoded_preds)
    precision = precision_score(decoded_labels, decoded_preds, zero_division=0)
    recall = recall_score(decoded_labels, decoded_preds, zero_division=0)
    f1 = f1_score(decoded_labels, decoded_preds, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def setup_lora_model(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """Setup model with LoRA adapters."""
    print(f"Loading model: {model_name}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention layers
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing for memory efficiency
    model.enable_input_require_grads()
    
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, val_dataset, output_dir="./qwen_sarcasm_lora", num_epochs=2, use_wandb=True, wandb_project="sarcasm-sft"):
    """Train the model with LoRA and comprehensive metrics tracking."""
    
    # Setup WandB
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=f"sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "model": "Qwen2.5-0.5B-Instruct",
                "method": "SFT",
                "dataset": "GEN-sarc-notsarc (train split)",
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "num_epochs": num_epochs,
                "learning_rate": 2e-4,
                "batch_size": 2,
                "gradient_accumulation_steps": 8,
                "effective_batch_size": 16
            }
        )
    
    # Setup metrics logger
    metrics_logger = MetricsLogger(output_dir)
    
    # Training arguments (optimized for memory efficiency)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,  # Reduced to 2 for memory
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size of 16
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=50,
        eval_strategy="epoch",  # Evaluate every epoch for metrics tracking
        save_strategy="epoch",  # Save every epoch
        save_total_limit=2,  # Keep best 2 checkpoints
        fp16=False,  # Disable FP16 on MPS (can cause issues)
        report_to="wandb" if use_wandb else "none",
        load_best_model_at_end=True,  # Keep best model based on eval loss
        metric_for_best_model="eval_loss",  # Could change to "f1" if compute_metrics works
        greater_is_better=False,
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        dataloader_num_workers=0,  # Disable parallel loading to save memory
        max_grad_norm=1.0,
        disable_tqdm=False,
        logging_first_step=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Custom callback for logging metrics per epoch
    class MetricsCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            if state.epoch is not None:
                epoch = int(state.epoch)
                metrics_logger.log(epoch, metrics)
                
                # Log to WandB
                if use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        **metrics
                    })
        
        def on_train_end(self, args, state, control, **kwargs):
            # Plot training curves at the end
            metrics_logger.plot_curves()
            
            # Log final plots to WandB
            if use_wandb and os.path.exists(f"{output_dir}/sft_training_curves.png"):
                wandb.log({"training_curves": wandb.Image(f"{output_dir}/sft_training_curves.png")})
    
    # Trainer (note: compute_metrics difficult with generative models, using callback instead)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[MetricsCallback()],
    )
    
    # Train
    print("\nStarting training...")
    print(f"Total epochs: {num_epochs}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Output directory: {output_dir}\n")
    
    trainer.train()
    
    # Save final model
    print(f"\n✓ Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training summary
    summary = {
        "final_epoch": int(trainer.state.epoch),
        "total_steps": trainer.state.global_step,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "training_completed": datetime.now().isoformat()
    }
    
    with open(f"{output_dir}/training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Training completed!")
    print(f"  Best checkpoint: {summary['best_model_checkpoint']}")
    print(f"  Best eval loss: {summary['best_metric']:.4f}")
    
    # Log final summary to WandB
    if use_wandb:
        wandb.log({
            "final/best_eval_loss": summary['best_metric'],
            "final/total_steps": summary['total_steps']
        })
        wandb.finish()
    
    return trainer

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT Training with WandB Integration")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="sarcasm-detection", help="WandB project name")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--output_dir", type=str, default="models/sft", help="Output directory")
    
    args = parser.parse_args()
    use_wandb = not args.no_wandb
    
    # Setup
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # PHASE 1: Train on GEN dataset training split
    gen_train_path = "data/splits/gen_train.csv"
    output_dir = args.output_dir
    
    print("="*70)
    print("PHASE 1: Supervised Fine-Tuning on GEN Dataset (Training Split)")
    print("="*70)
    print("Strategy: Train on GEN training split for sarcasm detection")
    print("Test set (data/splits/gen_test.csv) is held-out for evaluation only")
    print(f"WandB: {'Enabled' if use_wandb else 'Disabled'}")
    print("Next Phase: DPO on same training data for preference alignment")
    print("="*70)
    
    # Check if split exists
    if not os.path.exists(gen_train_path):
        print(f"\n❌ Training split not found at {gen_train_path}")
        print("Run 'python scripts/split_gen_dataset.py' first to create train/test splits")
        return
    
    # Load model and tokenizer
    model, tokenizer = setup_lora_model(model_name)
    
    # Prepare data
    # For full training, set sample_size=None
    # For faster training, use sample_size (e.g., 2000, 4000)
    train_dataset, val_dataset = load_and_prepare_data(
        gen_train_path, 
        tokenizer,
        sample_size=4000  # Using 4k samples for reasonable training time
        # Set to None to use full training set (~5k samples)
    )
    
    # Train
    trainer = train_model(
        model, 
        tokenizer, 
        train_dataset, 
        val_dataset, 
        output_dir,
        num_epochs=args.epochs,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project
    )
    
    print("\nTraining complete!")
    print(f"Model saved to: {output_dir}")
    print(f"Test set remains untouched at: data/splits/gen_test.csv")

if __name__ == "__main__":
    main()
