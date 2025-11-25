"""
Fine-tune Qwen2.5-0.5B on SARC dataset using LoRA (Phase 1: SFT).
This script trains on the large SARC dataset to learn general sarcasm patterns.
Phase 2 (DPO) will use iSarcasm for preference alignment.
"""

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import json

def load_and_prepare_data(csv_path, tokenizer, max_length=256, sample_size=None):
    """Load and prepare the dataset for training."""
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check if it's SARC or iSarcasm format
    if 'comment' in df.columns:
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
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} examples from dataset")
    
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

def train_model(model, tokenizer, train_dataset, val_dataset, output_dir="./qwen_sarcasm_lora"):
    """Train the model with LoRA."""
    
    # Training arguments (optimized for memory efficiency)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=2,  # Reduced to 2 for memory
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size of 16
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=1,  # Only keep 1 checkpoint to save disk space
        fp16=False,  # Disable FP16 on MPS (can cause issues)
        report_to="none",
        load_best_model_at_end=False,  # Disable to save memory
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        dataloader_num_workers=0,  # Disable parallel loading to save memory
        max_grad_norm=1.0,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return trainer

def main():
    # Setup
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # PHASE 1: Train on SARC dataset (large volume for pattern learning)
    sarc_path = "data/SARC/train-balanced-sarcasm.csv"
    output_dir = "models/sft"
    
    print("="*70)
    print("PHASE 1: Supervised Fine-Tuning on SARC Dataset")
    print("="*70)
    print("Strategy: Train on large SARC dataset for general sarcasm patterns")
    print("Next Phase: DPO on iSarcasm for preference alignment")
    print("="*70)
    
    # Load model and tokenizer
    model, tokenizer = setup_lora_model(model_name)
    
    # Prepare data (you can adjust sample_size to control training time)
    # Set sample_size=None to use full dataset, or e.g., 100000 for subset
    train_dataset, val_dataset = load_and_prepare_data(
        sarc_path, 
        tokenizer,
        sample_size=5000  # Using 5k samples for memory-constrained systems (~15-30 min)
        # Increase to 10k (30-60 min) or 20k (1-2 hrs) if you have more memory
    )
    
    # Train
    trainer = train_model(model, tokenizer, train_dataset, val_dataset, output_dir)
    
    print("\nTraining complete!")
    print(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
