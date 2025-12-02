"""
Mine preference pairs from MobileLLM SFT model predictions for DPO training.

This script runs the SFT model on training data and creates preference pairs
where the model made mistakes, allowing DPO to learn from its errors.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from tqdm import tqdm
import json
import os
from datetime import datetime
import numpy as np


def mine_preferences(
    sft_model_path="models/mobilellm_sft",
    data_path="data/isarcasm2022.csv",
    output_path="data/mobilellm_sft_mistakes/sft_mistakes_only.json",
    sample_size=500,
    filter_strategy="all"  # "all", "mistakes_only", "confident_mistakes"
):
    """
    Run SFT model on iSarcasm data and create preference pairs.
    
    Args:
        sft_model_path: Path to SFT model
        data_path: Path to iSarcasm data (different from SFT training data!)
        output_path: Where to save preference pairs
        sample_size: Number of samples to process (None for all)
        filter_strategy: How to filter pairs:
            - "all": Create pairs for all samples
            - "mistakes_only": Only create pairs where SFT was wrong
            - "confident_mistakes": Only confident wrong predictions
    """
    base_model_name = "facebook/MobileLLM-R1.5-360M"
    
    # Fix relative paths
    if not os.path.exists(data_path):
        data_path = os.path.join("..", data_path)
    if not os.path.exists(sft_model_path):
        sft_model_path = os.path.join("..", sft_model_path)
    
    print(f"\n{'='*80}")
    print(f"MINING PREFERENCE PAIRS FROM MobileLLM SFT MODEL")
    print(f"{'='*80}")
    print(f"SFT Model: {sft_model_path}")
    print(f"Data: {data_path} (iSarcasm - NEW data, not used in SFT!)")
    print(f"Output: {output_path}")
    print(f"Filter Strategy: {filter_strategy}")
    print(f"{'='*80}\n")
    
    # Load iSarcasm data
    df = pd.read_csv(data_path)
    
    # iSarcasm has 'tweet' and 'sarcastic' columns
    print(f"iSarcasm dataset: {len(df)} samples")
    print(f"  Sarcastic: {df['sarcastic'].sum()}")
    print(f"  Non-sarcastic: {len(df) - df['sarcastic'].sum()}")
    
    # Balance the dataset
    if sample_size:
        sarc_df = df[df['sarcastic'] == 1]
        notsarc_df = df[df['sarcastic'] == 0]
        samples_per_class = sample_size // 2
        
        sarc_sample = sarc_df.sample(n=min(samples_per_class, len(sarc_df)), random_state=42)
        notsarc_sample = notsarc_df.sample(n=min(samples_per_class, len(notsarc_df)), random_state=42)
        
        df = pd.concat([sarc_sample, notsarc_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"\nBalanced sample: {len(df)} samples ({len(sarc_sample)} sarc, {len(notsarc_sample)} not-sarc)")
    
    print(f"Processing {len(df)} samples...")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, sft_model_path)
    model.eval()
    
    # Process samples
    preference_pairs = []
    correct = 0
    incorrect = 0
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Mining"):
            text = row['tweet']  # iSarcasm uses 'tweet' column
            true_label = row['sarcastic']  # iSarcasm uses 'sarcastic' column
            
            prompt = f"""Is the following text sarcastic? Answer with only 'Yes' or 'No'.

Text: {text}

Answer:"""
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model.generate(
                **inputs, 
                max_new_tokens=10, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )
            
            response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # Extract answer
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip().lower()
            else:
                answer = response.strip().lower()
            
            # Determine prediction
            if answer.startswith("yes"):
                pred = 1
            elif answer.startswith("no"):
                pred = 0
            else:
                pred = 0
            
            is_correct = (pred == true_label)
            if is_correct:
                correct += 1
            else:
                incorrect += 1
            
            # Create preference pair based on strategy
            should_include = False
            
            if filter_strategy == "all":
                should_include = True
            elif filter_strategy == "mistakes_only":
                should_include = not is_correct
            elif filter_strategy == "confident_mistakes":
                # Only include if model was wrong (confident mistakes give cleaner signal)
                should_include = not is_correct
            
            if should_include:
                # Create preference pair
                correct_answer = "Yes" if true_label == 1 else "No"
                wrong_answer = "No" if true_label == 1 else "Yes"
                
                preference_pairs.append({
                    'prompt': prompt,
                    'chosen': correct_answer,
                    'rejected': wrong_answer,
                    'text': text,
                    'true_label': true_label,
                    'sft_prediction': pred,
                    'sft_correct': is_correct
                })
    
    # Save preferences
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(preference_pairs, f, indent=2)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'sft_model': sft_model_path,
        'data_source': data_path,
        'total_samples': len(df),
        'correct_predictions': correct,
        'incorrect_predictions': incorrect,
        'sft_accuracy': correct / len(df),
        'filter_strategy': filter_strategy,
        'preference_pairs_created': len(preference_pairs)
    }
    
    summary_path = output_path.replace('.json', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"MINING COMPLETE")
    print(f"{'='*80}")
    print(f"SFT Accuracy: {correct}/{len(df)} ({correct/len(df)*100:.1f}%)")
    print(f"Preference pairs created: {len(preference_pairs)}")
    print(f"Saved to: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"{'='*80}\n")
    
    return preference_pairs, summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Mine preference pairs from MobileLLM SFT model on iSarcasm")
    parser.add_argument("--sft_model", type=str, default="models/mobilellm_sft", help="Path to SFT model")
    parser.add_argument("--data", type=str, default="data/isarcasm2022.csv", help="Path to iSarcasm data")
    parser.add_argument("--output", type=str, default="data/mobilellm_sft_mistakes/sft_mistakes_only.json", help="Output path")
    parser.add_argument("--sample_size", type=int, default=500, help="Number of balanced samples to process")
    parser.add_argument("--filter", type=str, default="confident_mistakes", 
                       choices=["all", "mistakes_only", "confident_mistakes"],
                       help="Filter strategy for preference pairs")
    
    args = parser.parse_args()
    
    mine_preferences(
        sft_model_path=args.sft_model,
        data_path=args.data,
        output_path=args.output,
        sample_size=args.sample_size,
        filter_strategy=args.filter
    )


if __name__ == "__main__":
    main()
