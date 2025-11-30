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
import os

class ConfidenceWeightedDPOTrainer(DPOTrainer):
    """Extended DPO trainer that uses confidence scores from hard negatives."""
    
    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        """Override to add confidence weighting to DPO loss."""
        # Get base loss from parent class
        loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval)
        
        # Apply confidence weighting if available
        if 'confidence' in batch:
            confidence = batch['confidence']
            # Scale loss by confidence: high-confidence mistakes matter more
            # confidence ranges from 0.6 to 1.0, so we normalize to 1.0-1.67x weight
            weights = 1.0 + (confidence - 0.6) * 1.67  # 0.6 → 1.0x, 1.0 → 1.67x
            loss = (loss * weights).mean()
        
        return loss, metrics 
def prepare_dpo_dataset(train_csv_path):
    """
    Prepare iSarcasm TRAINING dataset with enhanced preference pairs.
    
    Args:
        train_csv_path: Path to training split CSV (data/splits/isarcasm_train.csv)
    
    Returns:
        Dataset with prompt/chosen/rejected pairs
    """
    print(f"Loading iSarcasm training data from: {train_csv_path}")
    df = pd.read_csv(train_csv_path, index_col=0)
    
    print(f"Original training samples: {len(df)}")
    print(f"  Sarcastic: {df['sarcastic'].sum()} ({df['sarcastic'].mean():.1%})")
    print(f"  Non-sarcastic: {len(df) - df['sarcastic'].sum()} ({1-df['sarcastic'].mean():.1%})")
    
    # BALANCE DATASET: Oversample sarcastic examples to 50/50 ratio
    sarcastic_df = df[df['sarcastic'] == 1]
    non_sarcastic_df = df[df['sarcastic'] == 0]
    
    # Sample non-sarcastic to match sarcastic count
    non_sarcastic_balanced = non_sarcastic_df.sample(n=len(sarcastic_df), random_state=42)
    
    # Combine for balanced dataset
    df_balanced = pd.concat([sarcastic_df, non_sarcastic_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # LOAD HARD NEGATIVES if available
    hard_negatives_path = "data/hard_negatives.json"
    hard_negatives_loaded = []
    if os.path.exists(hard_negatives_path):
        print(f"\n✓ Loading hard negatives from: {hard_negatives_path}")
        with open(hard_negatives_path, 'r') as f:
            hard_negatives_loaded = json.load(f)
        print(f"  Loaded {len(hard_negatives_loaded)} hard negative examples")
        
        # Limit to top 200 highest-confidence mistakes to avoid overwhelming the dataset
        hard_negatives_loaded = hard_negatives_loaded[:200]
        print(f"  Using top {len(hard_negatives_loaded)} for training")
    else:
        print(f"\n⚠ No hard negatives found at {hard_negatives_path}")
        print("  Run scripts/mine_hard_negatives.py to generate them")
    
    print(f"\nBalanced training samples: {len(df_balanced)}")
    print(f"  Sarcastic: {df_balanced['sarcastic'].sum()} ({df_balanced['sarcastic'].mean():.1%})")
    print(f"  Non-sarcastic: {len(df_balanced) - df_balanced['sarcastic'].sum()} ({1-df_balanced['sarcastic'].mean():.1%})")
    
    dpo_data = []
    
    for _, row in df_balanced.iterrows():
        text = row['tweet']
        is_sarcastic = row['sarcastic']
        
        # Base prompt with more explicit instructions
        prompt = f"""Is the following text sarcastic? Sarcasm often involves irony, exaggeration, or saying the opposite of what is meant. Answer with 'Yes' or 'No'.

Text: {text}

Answer:"""
        
        # Create ENHANCED V2 chosen/rejected pairs with EXPLICIT REASONING
        if is_sarcastic == 1:
            # Collect sarcasm indicators
            indicators = []
            indicator_explanations = []
            
            if row.get('irony', 0) == 1:
                indicators.append('irony')
                indicator_explanations.append('saying the opposite of what is meant')
            if row.get('satire', 0) == 1:
                indicators.append('satire')
                indicator_explanations.append('mocking through exaggeration')
            if row.get('overstatement', 0) == 1:
                indicators.append('exaggeration')
                indicator_explanations.append('overstating for effect')
            if row.get('understatement', 0) == 1:
                indicators.append('understatement')
                indicator_explanations.append('minimizing something significant')
            if row.get('rhetorical_question', 0) == 1:
                indicators.append('rhetorical question')
                indicator_explanations.append('asking a question with an obvious answer')
            
            # V2 ENHANCEMENT: Multi-sentence reasoning with HOW and WHY
            if len(indicators) >= 2:
                # Strong sarcasm with multiple cues
                chosen = f" Yes. This text is sarcastic because it uses {indicators[0]} ({indicator_explanations[0]}) and {indicators[1]} ({indicator_explanations[1]}). The combination of these techniques creates a clear sarcastic tone that contradicts the literal meaning."
            elif len(indicators) == 1:
                # Clear sarcasm with single cue
                chosen = f" Yes. This is sarcastic. The text employs {indicators[0]}, which means it's {indicator_explanations[0]}. This indicates the speaker doesn't literally mean what they're saying."
            else:
                # Sarcasm without specific type - analyze text patterns
                # Look for common sarcasm patterns in the text
                text_lower = text.lower()
                if any(word in text_lower for word in ['love', 'great', 'amazing', 'perfect', 'wonderful']):
                    chosen = " Yes. This is sarcastic. The text uses positive words in a context where they likely express frustration or negativity, indicating the speaker means the opposite of what they're saying."
                elif any(word in text_lower for word in ['sure', 'totally', 'definitely', 'obviously', 'clearly']):
                    chosen = " Yes. This is sarcastic. The text uses emphatic agreement words ('sure', 'totally', etc.) which in context likely express doubt or disagreement, a common sarcastic pattern."
                elif '...' in text or '…' in text:
                    chosen = " Yes. This is sarcastic. The ellipsis suggests trailing off or implied meaning that contradicts the literal statement, a typical marker of sarcasm."
                else:
                    chosen = " Yes. This is sarcastic. The text's tone and context suggest the speaker means something different from the literal words, using conversational sarcasm to express their true sentiment."
            
            # V2 ENHANCEMENT: Rejected explains WHY someone might miss the sarcasm
            rejected = " No. While the text might seem unusual, there are no clear linguistic markers of sarcasm such as irony, exaggeration, or contradiction. It should be interpreted as a literal statement expressing the speaker's actual view."
            
        else:
            # Non-sarcastic text
            # V2 ENHANCEMENT: Explain what makes it genuinely non-sarcastic
            text_lower = text.lower()
            if any(word in text_lower for word in ['http', 'www', 'link', '@']):
                chosen = " No. This is a straightforward statement. It contains factual references (links, mentions) that indicate literal communication rather than sarcastic expression."
            elif len(text.split()) < 5:
                chosen = " No. This is a brief, literal statement. The text is too short and direct to convey the layered meaning required for sarcasm."
            elif any(word in text_lower for word in ['thank', 'please', 'sorry', 'congratulations']):
                chosen = " No. This appears to be a genuine expression of sentiment (thanks, politeness, etc.) without the contradictory tone that characterizes sarcasm."
            else:
                chosen = " No. This is a straightforward statement. The text conveys its meaning directly without irony, exaggeration, or the contradictory tone that defines sarcasm."
            
            # V2 ENHANCEMENT: Rejected shows what sarcasm WOULD look like
            rejected = " Yes. This could be sarcastic because... wait, actually there are no markers of sarcasm here. This is genuine communication."
        
        dpo_data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        })
    
    # ADD HARD NEGATIVE PREFERENCES with higher-quality explanations
    if hard_negatives_loaded:
        print(f"\nAdding {len(hard_negatives_loaded)} hard negative preference pairs...")
        for hn in hard_negatives_loaded:
            text = hn['text']
            true_label = hn['true_label']
            predicted_label = hn['predicted_label']
            confidence = hn['confidence']
            
            prompt = f"""Is the following text sarcastic? Sarcasm often involves irony, exaggeration, or saying the opposite of what is meant. Answer with 'Yes' or 'No'.

Text: {text}

Answer:"""
            
            # Create preference pair targeting the mistake
            if true_label == 1 and predicted_label == 0:
                # FALSE NEGATIVE: Model missed sarcasm
                # Chosen: Explain WHY it IS sarcastic (what the model missed)
                sarcasm_indicators = hn.get('sarcasm_type', {})
                active_types = [k for k, v in sarcasm_indicators.items() if v == 1]
                
                if active_types:
                    type_str = active_types[0].replace('_', ' ')
                    chosen = f" Yes. This is sarcastic. The model may have missed the {type_str} because it's subtle, but careful analysis reveals the speaker is expressing the opposite of what the literal words suggest. Pay attention to tone and context clues."
                else:
                    chosen = " Yes. This is sarcastic. Though subtle, the text uses conversational sarcasm where the speaker's true intent contradicts their literal words. Context and tone are key to recognizing this."
                
                # Rejected: The model's wrong reasoning (too literal)
                rejected = " No. This appears to be a straightforward statement without sarcasm markers."
                
            else:
                # FALSE POSITIVE: Model saw sarcasm where there was none
                # Chosen: Explain WHY it's NOT sarcastic
                chosen = " No. This is a genuine, literal statement. While it might use casual language or strong words, there's no irony, exaggeration, or contradiction that would indicate sarcasm. The speaker means exactly what they're saying."
                
                # Rejected: The model's wrong reasoning (over-interpreting)
                rejected = " Yes. This text shows sarcasm based on the word choice and tone."
            
            # Add with confidence weighting (higher confidence = more important to learn from)
            dpo_data.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
                'confidence': confidence  # Will use for weighted training
            })
    
    print(f"Created {len(dpo_data)} DPO preference pairs (including hard negatives)")
    return Dataset.from_list(dpo_data)

def load_model_for_dpo(base_model_name="Qwen/Qwen2.5-0.5B-Instruct", adapter_path=None):
    """Load model and apply LoRA if adapter path provided."""
    print(f"Loading model: {base_model_name}")
    
    # Auto-detect device and set optimal dtype
    if torch.cuda.is_available():
        dtype = torch.float16  # Faster on CUDA
        print("  Device: CUDA (using float16)")
    elif torch.backends.mps.is_available():
        dtype = torch.float32  # MPS compatibility
        print("  Device: MPS (using float32)")
    else:
        dtype = torch.float32  # CPU fallback
        print("  Device: CPU (using float32)")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
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
    # Auto-detect bf16/fp16 support
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8  # A100/H100
    use_fp16 = torch.cuda.is_available() and not use_bf16  # Other CUDA GPUs (T4, V100)
    
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced for memory
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Maintain effective batch size
        learning_rate=5e-5,
        warmup_steps=50,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=1,
        bf16=use_bf16,  # Auto-enable for A100/H100
        fp16=use_fp16,  # Auto-enable for T4/V100
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_length=512,  # Increased to accommodate richer responses
        max_prompt_length=256,
        beta=0.1,  # LOWERED: From 0.5 to 0.1 for gentler preference learning (less aggressive bias)
        disable_tqdm=False,
        logging_first_step=False,
        # Higher beta = model learns more aggressively from preference differences
    )
    
    # DPO Trainer with confidence weighting
    dpo_trainer = ConfidenceWeightedDPOTrainer(
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
    train_csv_path = "data/splits/isarcasm_train.csv"  # Pre-split training data
    sft_adapter_path = "models/sft"  # From Phase 1 SFT
    output_dir = "models/dpo_enhanced"  # Enhanced version
    
    print("="*70)
    print("PHASE 2: Direct Preference Optimization (DPO)")
    print("="*70)
    print("Strategy: Refine SARC-trained model with iSarcasm preferences")
    print(f"Training data: {train_csv_path}")
    print(f"Base model: {sft_adapter_path}")
    print("="*70)
    
    # Train DPO starting from SARC SFT model
    train_dpo(
        train_csv_path, 
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
