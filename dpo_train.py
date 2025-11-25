"""
Direct Preference Optimization (DPO) for sarcasm detection - Phase 2.
This script uses iSarcasm dataset for preference alignment after SFT on SARC.

ENHANCED VERSION - Quick Fixes Applied:
1. Richer preference pairs with explicit reasoning
   - "Yes" → "Yes. This text is sarcastic, showing irony."
   - "No" → "No. This is a straightforward, literal statement."
2. Stronger beta parameter (0.1 → 0.5)
   - Model learns more aggressively from preference differences
3. Better prompt instructions with sarcasm definition
4. Increased max_length (384 → 512) for richer responses

Expected improvement: +3-5 percentage points in accuracy and F1
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset 
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig
import json 
from sklearn.model_selection import train_test_split

def prepare_dpo_dataset(csv_path, train_only=True):
    """
    ENHANCED: Prepare iSarcasm dataset with stronger, more informative preference pairs.
    
    Improvements:
    1. Richer reasoning in chosen responses
    2. More specific rejection reasons (common error patterns)
    3. Confidence levels based on number of sarcasm indicators
    
    Args:
        csv_path: Path to iSarcasm CSV file
        train_only: If True, return only the training split (80%). If False, return full dataset.
    """
    print(f"Loading iSarcasm dataset for DPO from: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    
    # Split into train (80%) and test (20%) with stratification
    df_train, df_test = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['sarcastic']
    )
    
    # Save test set indices for evaluation script to use
    test_indices = df_test.index.tolist()
    with open('isarcasm_test_indices.json', 'w') as f:
        json.dump(test_indices, f)
    print(f"Saved {len(test_indices)} test indices to isarcasm_test_indices.json")
    
    # Use only training split for DPO
    if train_only:
        df = df_train
        print(f"\nUsing TRAIN split for DPO training:")
    else:
        print(f"\nUsing FULL dataset (not recommended - causes data leakage):")
    
    print(f"Total samples: {len(df)}")
    print(f"Sarcastic: {df['sarcastic'].sum()}, Non-sarcastic: {len(df) - df['sarcastic'].sum()}")
    
    dpo_data = []
    
    for _, row in df.iterrows():
        text = row['tweet']
        is_sarcastic = row['sarcastic']
        
        # Base prompt with more explicit instructions
        prompt = f"""Is the following text sarcastic? Sarcasm often involves irony, exaggeration, or saying the opposite of what is meant. Answer with 'Yes' or 'No'.

Text: {text}

Answer:"""
        
        # Create ENHANCED chosen/rejected pairs
        if is_sarcastic == 1:
            # Collect sarcasm indicators
            indicators = []
            if row.get('irony', 0) == 1:
                indicators.append('irony')
            if row.get('satire', 0) == 1:
                indicators.append('satire')
            if row.get('overstatement', 0) == 1:
                indicators.append('exaggeration')
            if row.get('understatement', 0) == 1:
                indicators.append('understatement')
            if row.get('rhetorical_question', 0) == 1:
                indicators.append('rhetorical question')
            
            # ENHANCED: Richer chosen with explicit reasoning
            if len(indicators) >= 2:
                # Strong sarcasm signal (multiple cues)
                chosen = f" Yes. This text contains multiple sarcastic cues: {', '.join(indicators)}."
            elif len(indicators) == 1:
                # Clear sarcasm signal (single cue)
                chosen = f" Yes. This text is sarcastic, showing {indicators[0]}."
            else:
                # Sarcasm without specific type annotation
                chosen = " Yes. This text is sarcastic based on contextual cues."
            
            # ENHANCED: Rejected with specific error reasoning
            rejected = " No. This appears to be a literal statement without sarcastic intent."
            
        else:
            # Non-sarcastic text
            # ENHANCED: Chosen with reasoning why it's NOT sarcastic
            chosen = " No. This is a straightforward, literal statement."
            
            # ENHANCED: Rejected with specific false positive error
            # (model incorrectly sees sarcasm where there is none)
            rejected = " Yes. This text seems sarcastic."
        
        dpo_data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        })
    
    print(f"Created {len(dpo_data)} DPO preference pairs")
    return Dataset.from_list(dpo_data)

def load_model_for_dpo(base_model_name="Qwen/Qwen2.5-0.5B-Instruct", adapter_path=None):
    """Load model and apply LoRA if adapter path provided."""
    print(f"Loading model: {base_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float32,  # Changed from torch_dtype to dtype
        device_map="auto"
    )
    
    # Load fine-tuned adapter if provided
    if adapter_path:
        print(f"Loading fine-tuned adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        # Apply fresh LoRA config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def train_dpo(csv_path, output_dir="./qwen_sarcasm_dpo", adapter_path=None):
    """Train model using DPO."""
    
    # Prepare dataset
    print("Preparing DPO dataset...")
    dataset = prepare_dpo_dataset(csv_path)
    
    # Split train/val
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Load model
    model, tokenizer = load_model_for_dpo(adapter_path=adapter_path)
    
    # For newer TRL versions, we need a reference model
    ref_model = None  # DPOTrainer will create a copy if needed
    
    # ENHANCED DPO Configuration
    # Key changes: Higher beta (0.1 → 0.5) for stronger preference learning
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced for memory
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Maintain effective batch size
        learning_rate=5e-5,
        warmup_steps=50,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=1,
        bf16=False,  # Disable for MPS
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_length=512,  # Increased to accommodate richer responses
        max_prompt_length=256,
        beta=0.5,  # ENHANCED: Increased from 0.1 to 0.5 for stronger preference signal
        # Higher beta = model learns more aggressively from preference differences
    )
    
    # DPO Trainer (updated for TRL v0.25+)
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    print("\nStarting DPO training...")
    dpo_trainer.train()
    
    # Save
    print(f"\nSaving DPO model to {output_dir}")
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return dpo_trainer

def main():
    # PHASE 2: DPO on iSarcasm dataset
    isarcasm_path = "data/isarcasm2022.csv"
    sft_adapter_path = "./qwen_sarc_sft"  # From Phase 1 SFT
    output_dir = "./qwen_sarcasm_dpo_enhanced"  # Enhanced version with richer preferences
    
    print("="*70)
    print("PHASE 2: Direct Preference Optimization on iSarcasm Dataset")
    print("="*70)
    print("Strategy: Refine SARC-trained model with high-quality iSarcasm preferences")
    print(f"Loading SFT checkpoint from: {sft_adapter_path}")
    print("="*70)
    
    # Train DPO starting from SARC SFT model
    train_dpo(
        isarcasm_path, 
        output_dir=output_dir, 
        adapter_path=sft_adapter_path
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Phase 1 (SFT): {sft_adapter_path}")
    print(f"Phase 2 (DPO): {output_dir}")
    print("\nWorkflow Summary:")
    print("  1. SFT on SARC → Learn general sarcasm patterns")
    print("  2. DPO on iSarcasm → Refine with expert preferences")
    print("="*70)

if __name__ == "__main__":
    main()
