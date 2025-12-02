"""
Mine preference pairs from SFT model's predictions on iSarcasm dataset.

This creates DPO training data by:
1. Running the SFT model on iSarcasm examples
2. Creating preference pairs where:
   - Chosen: Correct classification
   - Rejected: Wrong classification
   
This teaches DPO to avoid the SFT model's actual failure patterns,
rather than using arbitrary/synthetic preference pairs.
"""

import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def load_sft_model(base_model_name, adapter_path, device):
    """Load the SFT model with LoRA adapter."""
    print(f"\n{'='*80}")
    print(f"LOADING SFT MODEL")
    print(f"{'='*80}")
    print(f"Base model: {base_model_name}")
    print(f"Adapter: {adapter_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()  # Merge for faster inference
    model.eval()
    
    print(f"✓ Model loaded successfully")
    return model, tokenizer


def get_model_prediction(model, tokenizer, text, device, max_new_tokens=50):
    """Get SFT model's prediction for a single text."""
    prompt = f"""Is the following text sarcastic? Answer 'Yes' or 'No'.

Text: "{text}"

Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temperature for more deterministic outputs
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (skip the prompt)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Parse prediction
    response_lower = response.lower()
    
    # Check for "yes" indicating sarcastic
    if response_lower.startswith('yes'):
        prediction = 1
    elif response_lower.startswith('no'):
        prediction = 0
    else:
        # More sophisticated parsing
        if 'not sarcastic' in response_lower or 'not being sarcastic' in response_lower:
            prediction = 0
        elif 'sarcastic' in response_lower or 'yes' in response_lower.split()[:5]:
            prediction = 1
        else:
            # Default to non-sarcastic if unclear
            prediction = 0
    
    return prediction, response


def mine_preference_pairs(model, tokenizer, isarcasm_df, device, output_path, adapter_path, 
                          filter_strategy='confident_mistakes'):
    """
    Mine preference pairs from SFT model predictions.
    
    Args:
        filter_strategy: 
            - 'all': Use all examples (original behavior)
            - 'confident_mistakes': Only use examples where SFT was confidently wrong
            - 'mistakes_only': Only use incorrect predictions (any confidence)
    
    Creates pairs where:
    - Chosen: Correct answer (ground truth)
    - Rejected: Wrong answer (what SFT predicted)
    
    This gives DPO clear signal about what the model should/shouldn't do.
    """
    print(f"\n{'='*80}")
    print(f"MINING PREFERENCE PAIRS FROM SFT PREDICTIONS")
    print(f"{'='*80}")
    print(f"Dataset size: {len(isarcasm_df)}")
    print(f"Filter strategy: {filter_strategy}")
    
    preference_data = []
    predictions = []
    
    model.eval()
    
    for idx, row in tqdm(isarcasm_df.iterrows(), total=len(isarcasm_df), desc="Processing"):
        text = row['tweet']
        true_label = int(row['sarcastic'])
        
        # Get SFT model's prediction
        try:
            pred, response = get_model_prediction(model, tokenizer, text, device)
            
            # Create the prompt (same format as training)
            prompt = f"""Is the following text sarcastic? Answer 'Yes' or 'No'.

Text: "{text}"

Answer:"""
            
            # Create preference pair
            if true_label == 1:
                chosen = "Yes, this is sarcastic."
                rejected = "No, this is not sarcastic."
            else:
                chosen = "No, this is not sarcastic."
                rejected = "Yes, this is sarcastic."
            
            is_correct = (pred == true_label)
            
            preference_data.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
                'true_label': true_label,
                'sft_prediction': pred,
                'sft_response': response,
                'is_correct': is_correct,
                'text': text
            })
            
            predictions.append(pred)
            
        except Exception as e:
            print(f"\nError processing example {idx}: {e}")
            continue
    
    # Calculate statistics BEFORE filtering
    correct_predictions = sum(1 for item in preference_data if item['is_correct'])
    total_before_filter = len(preference_data)
    accuracy = correct_predictions / total_before_filter if total_before_filter > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"MINING RESULTS (Before Filtering)")
    print(f"{'='*80}")
    print(f"Total examples processed: {total_before_filter}")
    print(f"SFT model accuracy: {accuracy:.1%} ({correct_predictions}/{total_before_filter})")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {total_before_filter - correct_predictions}")
    
    # Calculate per-class accuracy
    sarcastic_examples = [item for item in preference_data if item['true_label'] == 1]
    non_sarcastic_examples = [item for item in preference_data if item['true_label'] == 0]
    
    sarcastic_correct = sum(1 for item in sarcastic_examples if item['is_correct'])
    non_sarcastic_correct = sum(1 for item in non_sarcastic_examples if item['is_correct'])
    
    print(f"\nPer-class accuracy:")
    print(f"  Sarcastic: {sarcastic_correct/len(sarcastic_examples):.1%} ({sarcastic_correct}/{len(sarcastic_examples)})")
    print(f"  Non-sarcastic: {non_sarcastic_correct/len(non_sarcastic_examples):.1%} ({non_sarcastic_correct}/{len(non_sarcastic_examples)})")
    
    # Apply filtering strategy
    if filter_strategy == 'confident_mistakes':
        print(f"\n{'='*80}")
        print(f"APPLYING FILTER: Confident Mistakes Only")
        print(f"{'='*80}")
        print("Keeping only examples where SFT was wrong (confident errors to correct)")
        
        # Keep only incorrect predictions (mistakes)
        filtered_data = [item for item in preference_data if not item['is_correct']]
        
        print(f"✓ Filtered: {len(filtered_data)} mistakes out of {total_before_filter} examples")
        print(f"  Reduction: {(1 - len(filtered_data)/total_before_filter)*100:.1f}%")
        preference_data = filtered_data
        
    elif filter_strategy == 'mistakes_only':
        print(f"\n{'='*80}")
        print(f"APPLYING FILTER: All Mistakes")
        print(f"{'='*80}")
        filtered_data = [item for item in preference_data if not item['is_correct']]
        preference_data = filtered_data
        print(f"✓ Filtered: {len(filtered_data)} mistakes")
    
    # If 'all', no filtering
    
    total = len(preference_data)
    
    # Save preference pairs
    print(f"\n{'='*80}")
    print(f"FINAL PREFERENCE PAIRS")
    print(f"{'='*80}")
    print(f"Saving {len(preference_data)} preference pairs to: {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(preference_data, f, indent=2)
    
    print(f"✓ Saved {len(preference_data)} preference pairs")
    
    # Save summary statistics
    summary_path = output_path.replace('.json', '_summary.json')
    summary = {
        'filter_strategy': filter_strategy,
        'total_examples_processed': total_before_filter,
        'total_examples_kept': total,
        'sft_accuracy': accuracy,
        'correct_predictions': correct_predictions,
        'incorrect_predictions': total_before_filter - correct_predictions,
        'sarcastic_accuracy': sarcastic_correct / len(sarcastic_examples) if sarcastic_examples else 0,
        'non_sarcastic_accuracy': non_sarcastic_correct / len(non_sarcastic_examples) if non_sarcastic_examples else 0,
        'dataset': 'iSarcasm',
        'base_model': 'Qwen2.5-0.5B-Instruct',
        'sft_adapter': str(adapter_path)
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary to: {summary_path}")
    
    return preference_data, summary


def main():
    parser = argparse.ArgumentParser(description='Mine preference pairs from SFT model predictions')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='Base model name')
    parser.add_argument('--adapter_path', type=str, default='models/sft',
                        help='Path to SFT adapter')
    parser.add_argument('--isarcasm_path', type=str, default='data/isarcasm2022.csv',
                        help='Path to iSarcasm dataset')
    parser.add_argument('--output_path', type=str, default='data/sft_mined_preferences.json',
                        help='Output path for preference pairs')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Sample size for testing (None = use full dataset)')
    parser.add_argument('--filter_strategy', type=str, default='confident_mistakes',
                        choices=['all', 'confident_mistakes', 'mistakes_only'],
                        help='Filter strategy: all (no filter), confident_mistakes (only errors), mistakes_only (all errors)')
    
    args = parser.parse_args()
    
    # Setup paths
    if not os.path.isabs(args.adapter_path):
        args.adapter_path = os.path.join(parent_dir, args.adapter_path)
    if not os.path.isabs(args.isarcasm_path):
        args.isarcasm_path = os.path.join(parent_dir, args.isarcasm_path)
    if not os.path.isabs(args.output_path):
        args.output_path = os.path.join(parent_dir, args.output_path)
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load iSarcasm dataset
    print(f"\nLoading iSarcasm dataset from: {args.isarcasm_path}")
    isarcasm_df = pd.read_csv(args.isarcasm_path)
    
    # Sample if requested
    if args.sample_size:
        print(f"Sampling {args.sample_size} examples for testing...")
        isarcasm_df = isarcasm_df.sample(n=min(args.sample_size, len(isarcasm_df)), random_state=42)
    
    print(f"Dataset loaded: {len(isarcasm_df)} examples")
    print(f"Columns: {list(isarcasm_df.columns)}")
    
    # Load SFT model
    model, tokenizer = load_sft_model(args.base_model, args.adapter_path, device)
    
    # Mine preference pairs
    preference_data, summary = mine_preference_pairs(
        model, tokenizer, isarcasm_df, device, args.output_path, args.adapter_path,
        filter_strategy=args.filter_strategy
    )
    
    print(f"\n{'='*80}")
    print(f"DONE! Preference pairs ready for DPO training.")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"1. Review the mined preferences: {args.output_path}")
    print(f"2. Run DPO training with: python scripts/dpo_train_v2.py --preference_data {args.output_path}")
    

if __name__ == "__main__":
    main()
