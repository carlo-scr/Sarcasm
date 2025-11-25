"""
Direct Preference Optimization (DPO) with LLM Judge - Phase 2 Alternative.
This script uses a larger LLM to judge model-generated responses and create preferences.

APPROACH:
1. Small model generates multiple candidate responses for each tweet
2. Judge LLM evaluates which response is better
3. Use judge's rankings to create DPO preference pairs
4. Train small model to align with judge's preferences

This explores if using an LLM judge leads to better generalization than ground truth labels.
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset 
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig
import json 
from tqdm import tqdm

def load_judge_model(judge_model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    """Load a larger LLM to serve as judge."""
    print(f"Loading judge model: {judge_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        judge_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        load_in_8bit=True,
    )
    
    return model, tokenizer

def load_small_model(base_model_name="Qwen/Qwen2.5-0.5B-Instruct", adapter_path=None):
    """Load the small model for generating candidates."""
    print(f"Loading small model: {base_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    if adapter_path:
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def generate_candidate_responses(small_model, tokenizer, text, num_candidates=2):
    """Generate multiple candidate responses from the small model."""
    messages = [
        {"role": "user", "content": f"Is the following text sarcastic?\n\nText: {text}\n\nAnswer:"}
    ]
    
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    candidates = []
    
    for i in range(num_candidates):
        inputs = tokenizer(prompt, return_tensors="pt").to(small_model.device)
        
        with torch.no_grad():
            outputs = small_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7 + (i * 0.1),  # Vary temperature for diversity
                top_p=0.8,
                top_k=20,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        response = response.strip()
        
        # Remove thinking tags if present
        if '</think>' in response:
            response = response.split('</think>')[-1].strip()
        
        candidates.append(response)
    
    return candidates

def extract_label(response):
    """Extract 'Yes' or 'No' label from response."""
    response_lower = response.lower()
    if 'yes' in response_lower:
        return 1
    elif 'no' in response_lower:
        return 0
    else:
        return -1  # Unclear

def judge_responses(judge_model, judge_tokenizer, text, response_a, response_b):
    """Use judge LLM to rank two responses."""
    judge_prompt = f"""You are an expert judge. Compare two responses to a sarcasm detection question.

Question: Is the following text sarcastic? Answer with 'Yes' or 'No'.
Text: {text}

Response A: {response_a}
Response B: {response_b}

Which response is better? Respond with only 'A' or 'B'."""
    
    messages = [{"role": "user", "content": judge_prompt}]
    
    try:
        prompt = judge_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except TypeError:
        prompt = judge_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    inputs = judge_tokenizer(prompt, return_tensors="pt").to(judge_model.device)
    
    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.2,
            do_sample=False,
            pad_token_id=judge_tokenizer.eos_token_id
        )
    
    judgment = judge_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    judgment = judgment.strip().upper()
    
    if 'A' in judgment:
        return 'A'
    elif 'B' in judgment:
        return 'B'
    else:
        return 'A'  # Default to A if unclear

def prepare_dpo_dataset_with_judge(train_csv_path, small_model, small_tokenizer, judge_model, judge_tokenizer, num_candidates=2):
    """
    Prepare DPO dataset using LLM judge to rank model-generated responses.
    Also tracks judge accuracy, F1, precision, recall, and win rate.
    
    Args:
        train_csv_path: Path to training split CSV
        small_model: Model to generate candidates
        small_tokenizer: Tokenizer for small model
        judge_model: LLM judge model
        judge_tokenizer: Tokenizer for judge
        num_candidates: Number of candidates to generate per sample
    
    Returns:
        Tuple of (Dataset with prompt/chosen/rejected pairs, metrics dict)
    """
    print(f"Loading iSarcasm training data from: {train_csv_path}")
    df = pd.read_csv(train_csv_path, index_col=0)
    
    print(f"Training samples: {len(df)}")
    print(f"  Sarcastic: {df['sarcastic'].sum()} ({df['sarcastic'].mean():.1%})")
    print(f"  Non-sarcastic: {len(df) - df['sarcastic'].sum()} ({1-df['sarcastic'].mean():.1%})")
    
    dpo_data = []
    
    # Metrics tracking
    judge_correct = 0
    judge_total = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    judge_picks_correct = 0  # Win rate: how often judge picks the correct answer
    
    print(f"\nGenerating candidates and judge rankings...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Judge ranking"):
        text = row['tweet']
        ground_truth_label = row['sarcastic']
        
        # Create prompt
        prompt = f"""Is the following text sarcastic? Sarcasm often involves irony, exaggeration, or saying the opposite of what is meant.

Text: {text}

Answer:"""
        
        # Generate candidates
        candidates = generate_candidate_responses(small_model, small_tokenizer, text, num_candidates=num_candidates)
        
        # Skip if we don't have at least 2 candidates
        if len(candidates) < 2:
            continue
        
        # Extract labels from candidates
        labels = [extract_label(cand) for cand in candidates]
        
        # Judge the candidates
        chosen_idx = 0  # Default to first candidate
        rejected_idx = 1
        
        if len(candidates) >= 2:
            judgment = judge_responses(judge_model, judge_tokenizer, text, candidates[0], candidates[1])
            if judgment == 'B':
                chosen_idx = 1
                rejected_idx = 0
        
        # Extract labels
        chosen_label = labels[chosen_idx]
        rejected_label = labels[rejected_idx]
        
        # Track judge metrics
        judge_total += 1
        
        # Did judge pick the correct answer?
        if chosen_label == ground_truth_label and chosen_label != -1:
            judge_picks_correct += 1
            judge_correct += 1
            # Count TP/TN
            if chosen_label == 1:
                true_positives += 1
            else:
                true_negatives += 1
        elif chosen_label != -1:
            # Count FP/FN
            if chosen_label == 1:
                false_positives += 1
            else:
                false_negatives += 1
        
        # Create preference pair (hybrid: correct answer + judge preference)
        if chosen_label == ground_truth_label and chosen_label != -1:
            # Judge picked the correct answer - make it the preferred choice
            chosen = " " + candidates[chosen_idx]
            rejected = " " + candidates[rejected_idx]
        elif rejected_label == ground_truth_label and rejected_label != -1:
            # Judge picked the wrong answer - swap them
            chosen = " " + candidates[rejected_idx]
            rejected = " " + candidates[chosen_idx]
        else:
            # Both or neither are clear - use judge's preference as tiebreaker
            chosen = " " + candidates[chosen_idx]
            rejected = " " + candidates[rejected_idx]
        
        dpo_data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        })
    
    print(f"Created {len(dpo_data)} DPO preference pairs (judge-ranked)")
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = judge_correct / judge_total if judge_total > 0 else 0
    win_rate = judge_picks_correct / judge_total if judge_total > 0 else 0
    
    metrics = {
        'total_pairs': judge_total,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'win_rate': win_rate,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'correct_choices': judge_picks_correct
    }
    
    print("\nJudge Performance Metrics:")
    print(f"  Total pairs: {metrics['total_pairs']}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall: {metrics['recall']:.2%}")
    print(f"  F1 Score: {metrics['f1_score']:.2%}")
    print(f"  Win Rate (picks correct answer): {metrics['win_rate']:.2%}")
    print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}")
    print(f"  TN: {metrics['true_negatives']}, FN: {metrics['false_negatives']}")
    
    return Dataset.from_list(dpo_data), metrics

def load_model_for_dpo(base_model_name="Qwen/Qwen2.5-0.5B-Instruct", adapter_path=None):
    """Load model and apply LoRA if adapter path provided."""
    print(f"Loading model for DPO: {base_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float32,
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

def train_dpo(dataset, metrics, output_dir="./qwen_sarcasm_dpo_judge", base_model_name="Qwen/Qwen2.5-0.5B-Instruct", adapter_path=None):
    """Train model using DPO with judge-ranked preferences."""
    
    # Split train/val
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Load model
    model, tokenizer = load_model_for_dpo(base_model_name=base_model_name, adapter_path=adapter_path)
    
    ref_model = None
    
    # DPO Configuration (same as before)
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        warmup_steps=50,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=1,
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_length=512,
        max_prompt_length=256,
        beta=0.5,
    )
    
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    print("\nStarting DPO training with judge-ranked preferences...")
    dpo_trainer.train()
    
    print(f"\nSaving DPO model to {output_dir}")
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save judge metrics
    metrics_file = f"{output_dir}/judge_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved judge metrics to {metrics_file}")
    
    return dpo_trainer

def main():
    print("="*70)
    print("PHASE 2: DPO with LLM Judge")
    print("="*70)
    print("Strategy: Use judge LLM to rank model-generated responses")
    print("="*70)
    
    train_csv_path = "data/splits/isarcasm_train.csv"
    sft_adapter_path = "models/sft"
    output_dir = "models/dpo_judge"
    judge_model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Open-source, no auth required
    
    # Load models
    print("\nLoading models...")
    small_model, small_tokenizer = load_small_model(adapter_path=sft_adapter_path)
    judge_model, judge_tokenizer = load_judge_model(judge_model_name)
    
    # Prepare dataset with judge
    print("\nPreparing dataset with judge rankings...")
    dataset, judge_metrics = prepare_dpo_dataset_with_judge(
        train_csv_path,
        small_model,
        small_tokenizer,
        judge_model,
        judge_tokenizer,
        num_candidates=2
    )
    
    # Free GPU memory from generation models
    del small_model, judge_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Train DPO
    train_dpo(
        dataset,
        judge_metrics,
        output_dir=output_dir,
        adapter_path=sft_adapter_path
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Judge-based DPO model saved to: {output_dir}")
    print(f"Judge metrics saved to: {output_dir}/judge_metrics.json")
    print("\nComparison:")
    print("  Ground truth DPO: models/dpo_enhanced")
    print("  Judge-based DPO: models/dpo_judge")
    print("\nRun evaluate_all_stages.py to compare performance")
    print("="*70)

if __name__ == "__main__":
    main()
