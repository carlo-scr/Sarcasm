"""
Evaluate DPO model on GEN test set.

Quick evaluation script to test DPO model performance without
running the full multi-stage evaluation.
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
import json
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_dpo_model(base_model_name, dpo_adapter_path, device):
    """Load DPO model with adapter."""
    print(f"\n{'='*80}")
    print(f"LOADING DPO MODEL")
    print(f"{'='*80}")
    print(f"Base model: {base_model_name}")
    print(f"DPO adapter: {dpo_adapter_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    
    # Load DPO adapter
    model = PeftModel.from_pretrained(base_model, dpo_adapter_path)
    model.eval()
    
    print(f"✓ DPO model loaded successfully")
    print(f"{'='*80}\n")
    
    return model, tokenizer


def evaluate_model(model, tokenizer, test_df, device, sample_size=None):
    """
    Evaluate DPO model on test set.
    
    Args:
        model: Loaded model
        tokenizer: Tokenizer
        test_df: Test DataFrame with text and label columns
        device: Device to use
        sample_size: Optional number of samples to evaluate (for quick testing)
    
    Returns:
        Dictionary with metrics
    """
    if sample_size:
        # Balanced sampling
        sarcastic = test_df[test_df['label'] == 1].sample(n=min(sample_size//2, len(test_df[test_df['label'] == 1])), random_state=42)
        non_sarcastic = test_df[test_df['label'] == 0].sample(n=min(sample_size//2, len(test_df[test_df['label'] == 0])), random_state=42)
        test_df = pd.concat([sarcastic, non_sarcastic]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Evaluating on {len(test_df)} samples (balanced)")
    else:
        print(f"Evaluating on {len(test_df)} samples (full test set)")
    
    predictions = []
    labels = []
    
    model.eval()
    
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
            text = row['text']
            label = row['label']
            
            # Format prompt
            prompt = f"""Is the following text sarcastic? Answer 'Yes' or 'No'.

Text: "{text}"

Answer:"""
            
            # Generate prediction
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode response
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Parse prediction
            response_lower = response.lower()
            if response_lower.startswith('yes') or ('sarcastic' in response_lower and 'not sarcastic' not in response_lower):
                pred = 1
            elif response_lower.startswith('no') or 'not sarcastic' in response_lower:
                pred = 0
            else:
                # Fallback: check for keywords
                if 'yes' in response_lower.split()[:5]:
                    pred = 1
                else:
                    pred = 0
            
            predictions.append(pred)
            labels.append(label)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'labels': labels
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate DPO model on GEN test set')
    parser.add_argument('--dpo_model', type=str, default='models/dpo_v3', 
                        help='Path to DPO model adapter')
    parser.add_argument('--test_data', type=str, default='data/splits/gen_test.csv',
                        help='Path to test dataset')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='Base model name')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of samples to evaluate (None for full dataset)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results (optional)')
    
    args = parser.parse_args()
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load test data
    print(f"\n{'='*80}")
    print("LOADING TEST DATA")
    print(f"{'='*80}")
    print(f"Source: {args.test_data}")
    
    test_df = pd.read_csv(args.test_data)
    
    # Handle different column names and formats
    if 'class' in test_df.columns:
        # GEN dataset format
        test_df['label'] = (test_df['class'] == 'sarc').astype(int)
    elif 'sarcastic' in test_df.columns:
        # iSarcasm format
        test_df['label'] = test_df['sarcastic']
    # else already has 'label' column
    
    print(f"Test samples: {len(test_df)}")
    print(f"  Sarcastic: {test_df['label'].sum()} ({test_df['label'].mean():.1%})")
    print(f"  Non-sarcastic: {len(test_df) - test_df['label'].sum()} ({1-test_df['label'].mean():.1%})")
    print(f"{'='*80}\n")
    
    # Load DPO model
    model, tokenizer = load_dpo_model(args.base_model, args.dpo_model, device)
    
    # Evaluate
    print(f"{'='*80}")
    print("EVALUATING DPO MODEL")
    print(f"{'='*80}\n")
    
    results = evaluate_model(model, tokenizer, test_df, device, args.sample_size)
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Model: {args.dpo_model}")
    print(f"Test set: {args.test_data}")
    print(f"Samples evaluated: {len(results['labels'])}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"  Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"  F1 Score:  {results['f1']:.4f} ({results['f1']*100:.2f}%)")
    
    print(f"\nPer-class Report:")
    print(classification_report(
        results['labels'], 
        results['predictions'], 
        target_names=['Not Sarcastic', 'Sarcastic'],
        digits=4
    ))
    print(f"{'='*80}\n")
    
    # Save results if output path specified
    if args.output:
        output_data = {
            'model': args.dpo_model,
            'test_data': args.test_data,
            'base_model': args.base_model,
            'samples_evaluated': len(results['labels']),
            'metrics': {
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1': results['f1']
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
