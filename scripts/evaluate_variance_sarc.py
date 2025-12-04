"""
Evaluate models multiple times on SARC data with different random samples to calculate variance/std/min/max.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path

def evaluate_model(model, tokenizer, test_df, num_samples=250, seed=42):
    """Evaluate model on a random sample of test data."""
    # Sample the data
    np.random.seed(seed)
    sample_df = test_df.sample(n=num_samples, random_state=seed)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    predictions = []
    labels = sample_df['label'].tolist()  # Labels are already 0 and 1
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Evaluating"):
        text = row['comment']  # Column name is 'comment' in SARC dataset
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that detects sarcasm in text."},
            {"role": "user", "content": f"Is the following text sarcastic? Answer with only 'Yes' or 'No'.\n\nText: {text}"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.7,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        # Parse response
        response_lower = response.lower()
        if 'yes' in response_lower:
            pred = 1
        elif 'no' in response_lower:
            pred = 0
        else:
            pred = 0  # Default to non-sarcastic
        
        predictions.append(pred)
    
    # Calculate metrics
    correct = sum([p == l for p, l in zip(predictions, labels)])
    accuracy = correct / len(labels)
    
    tp = sum([p == 1 and l == 1 for p, l in zip(predictions, labels)])
    fp = sum([p == 1 and l == 0 for p, l in zip(predictions, labels)])
    fn = sum([p == 0 and l == 1 for p, l in zip(predictions, labels)])
    tn = sum([p == 0 and l == 0 for p, l in zip(predictions, labels)])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }

def load_base_model():
    """Load base Qwen model."""
    print("\n=== Loading Base Qwen2.5-0.5B ===")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def load_sft_model():
    """Load SFT model."""
    print("\n=== Loading Qwen SFT ===")
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = "models/sft"
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    
    return model, tokenizer

def load_dpo_model():
    """Load DPO model (qwen_dpo_mistakes)."""
    print("\n=== Loading Qwen DPO (Mistakes) ===")
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    dpo_adapter = "models/qwen_dpo_mistakes"
    
    tokenizer = AutoTokenizer.from_pretrained(dpo_adapter)
    
    # DPO adapter is trained on top of SFT, so just load base + DPO
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, dpo_adapter)
    model = model.merge_and_unload()
    
    return model, tokenizer

def calculate_statistics(results):
    """Calculate mean, std, min, max, variance for each metric."""
    stats = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        values = [r[metric] for r in results]
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'variance': np.var(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    return stats

def main():
    # Load SARC test data
    test_df = pd.read_csv('data/SARC/train-balanced-sarcasm.csv')
    print(f"Total SARC test samples available: {len(test_df)}")
    
    num_runs = 5
    num_samples_per_run = 250
    
    all_results = {
        'base': [],
        'sft': [],
        'dpo': []
    }
    
    # Evaluate Base Model
    print("\n" + "="*60)
    print("EVALUATING BASE MODEL (5 runs)")
    print("="*60)
    model, tokenizer = load_base_model()
    for run in range(num_runs):
        print(f"\n--- Run {run+1}/5 (seed={run*100}) ---")
        results = evaluate_model(model, tokenizer, test_df, num_samples=num_samples_per_run, seed=run*100)
        all_results['base'].append(results)
        print(f"Accuracy: {results['accuracy']:.2f}%, F1: {results['f1']:.2f}%")
    del model
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    
    # Evaluate SFT Model
    print("\n" + "="*60)
    print("EVALUATING SFT MODEL (5 runs)")
    print("="*60)
    model, tokenizer = load_sft_model()
    for run in range(num_runs):
        print(f"\n--- Run {run+1}/5 (seed={run*100}) ---")
        results = evaluate_model(model, tokenizer, test_df, num_samples=num_samples_per_run, seed=run*100)
        all_results['sft'].append(results)
        print(f"Accuracy: {results['accuracy']:.2f}%, F1: {results['f1']:.2f}%")
    del model
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    
    # Evaluate DPO Model
    print("\n" + "="*60)
    print("EVALUATING DPO MODEL (5 runs)")
    print("="*60)
    model, tokenizer = load_dpo_model()
    for run in range(num_runs):
        print(f"\n--- Run {run+1}/5 (seed={run*100}) ---")
        results = evaluate_model(model, tokenizer, test_df, num_samples=num_samples_per_run, seed=run*100)
        all_results['dpo'].append(results)
        print(f"Accuracy: {results['accuracy']:.2f}%, F1: {results['f1']:.2f}%")
    del model
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    
    # Calculate statistics
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    final_stats = {}
    for model_name in ['base', 'sft', 'dpo']:
        print(f"\n{model_name.upper()} MODEL:")
        stats = calculate_statistics(all_results[model_name])
        final_stats[model_name] = stats
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            s = stats[metric]
            print(f"\n{metric.upper()}:")
            print(f"  Mean:     {s['mean']:.2f}%")
            print(f"  Std Dev:  {s['std']:.2f}%")
            print(f"  Variance: {s['variance']:.2f}")
            print(f"  Min:      {s['min']:.2f}%")
            print(f"  Max:      {s['max']:.2f}%")
            print(f"  Values:   {[f'{v:.2f}' for v in s['values']]}")
    
    # Save results
    output = {
        'config': {
            'num_runs': num_runs,
            'samples_per_run': num_samples_per_run,
            'total_test_samples': len(test_df),
            'seeds': [i*100 for i in range(num_runs)],
            'dataset': 'SARC train-balanced-sarcasm'
        },
        'raw_results': all_results,
        'statistics': {
            model_name: {
                metric: {
                    'mean': float(stats[metric]['mean']),
                    'std': float(stats[metric]['std']),
                    'variance': float(stats[metric]['variance']),
                    'min': float(stats[metric]['min']),
                    'max': float(stats[metric]['max']),
                    'values': [float(v) for v in stats[metric]['values']]
                }
                for metric in ['accuracy', 'precision', 'recall', 'f1']
            }
            for model_name, stats in final_stats.items()
        }
    }
    
    output_path = Path('results/variance_analysis_sarc.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nResults saved to {output_path}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE (Mean ± Std)")
    print("="*60)
    print(f"{'Model':<10} {'Accuracy':<20} {'Precision':<20} {'Recall':<20} {'F1':<20}")
    print("-" * 90)
    for model_name in ['base', 'sft', 'dpo']:
        stats = final_stats[model_name]
        print(f"{model_name.upper():<10} "
              f"{stats['accuracy']['mean']:.2f}±{stats['accuracy']['std']:.2f}%  "
              f"{stats['precision']['mean']:.2f}±{stats['precision']['std']:.2f}%  "
              f"{stats['recall']['mean']:.2f}±{stats['recall']['std']:.2f}%  "
              f"{stats['f1']['mean']:.2f}±{stats['f1']['std']:.2f}%")

if __name__ == "__main__":
    main()
