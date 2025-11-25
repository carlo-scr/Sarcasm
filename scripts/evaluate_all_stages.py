"""
Comprehensive evaluation script to compare all training stages:
1. Base Model (zero-shot)
2. After SFT (Phase 1)
3. After DPO (Phase 2)

This script evaluates each model on the iSarcasm test set and saves comparative results.
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import json
from datetime import datetime
import os
from sklearn.model_selection import train_test_split

def load_model_and_tokenizer(model_path, is_adapter=False, base_model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """Load model and tokenizer. Handles both base models and LoRA adapters."""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name if is_adapter else model_path)
    
    if is_adapter and os.path.exists(model_path):
        # Load base model then adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        print(f"  Loaded adapter from {model_path}")
    else:
        # Load regular model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    
    return model, tokenizer

def get_prediction(model, tokenizer, text, max_new_tokens=50):
    """Get model prediction for a single text."""
    messages = [
        {"role": "user", "content": f"Is the following text sarcastic? Answer with only 'Yes' or 'No'.\n\nText: {text}\n\nAnswer:"}
    ]
    
    # Apply chat template
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
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    response = response.strip().lower()
    
    # Handle thinking tags if present
    if '</think>' in response:
        response = response.split('</think>')[-1].strip()
    
    # Parse response
    if 'yes' in response:
        return 1
    elif 'no' in response:
        return 0
    else:
        return -1

def evaluate_model(model, tokenizer, df, model_name, sample_size=None):
    """Evaluate a single model."""
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")
    print(f"Samples: {len(df)}")
    
    predictions = []
    correct = 0
    unclear = 0
    
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=model_name):
        text = row['tweet']
        true_label = row['sarcastic']
        
        pred = get_prediction(model, tokenizer, text)
        predictions.append(pred)
        
        if pred == -1:
            unclear += 1
        else:
            if pred == true_label:
                correct += 1
                if pred == 1:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if pred == 1:
                    false_positives += 1
                else:
                    false_negatives += 1
    
    # Calculate metrics
    valid_predictions = sum(1 for p in predictions if p != -1)
    accuracy = correct / valid_predictions if valid_predictions > 0 else 0
    
    # Precision, Recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        'model_name': model_name,
        'total_samples': len(df),
        'valid_predictions': valid_predictions,
        'unclear_responses': unclear,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'timestamp': datetime.now().isoformat()
    }
    
    # Print results
    print(f"\nResults for {model_name}:")
    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1 Score:  {f1:.2%}")
    print(f"  Unclear:   {unclear}")
    
    return results, predictions

def main():
    # Load test dataset
    print("="*70)
    print("COMPARATIVE MODEL EVALUATION")
    print("="*70)
    
    test_csv_path = 'data/splits/isarcasm_test.csv'
    print(f"\nLoading held-out test set from: {test_csv_path}")
    
    if not os.path.exists(test_csv_path):
        print(f"❌ Test split not found at {test_csv_path}")
        print("   Run 'python create_splits.py' first to create train/test splits.")
        return
    
    df_test = pd.read_csv(test_csv_path, index_col=0)
    print(f"✓ Test set: {len(df_test)} samples")
    print(f"  Sarcastic: {df_test['sarcastic'].sum()} ({df_test['sarcastic'].mean():.1%})")
    print(f"  Non-sarcastic: {len(df_test) - df_test['sarcastic'].sum()} ({1-df_test['sarcastic'].mean():.1%})")
    
    all_results = []
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Stage 1: Base Model (Zero-shot)
    print(f"\n{'='*70}")
    print("STAGE 1: Base Model (Zero-shot)")
    print("="*70)
    try:
        model, tokenizer = load_model_and_tokenizer(base_model_name, is_adapter=False)
        results_base, _ = evaluate_model(model, tokenizer, df_test, "Base Model (Zero-shot)")
        all_results.append(results_base)
        
        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"❌ Could not evaluate base model: {e}")
    
    # Stage 2: After SFT
    print(f"\n{'='*70}")
    print("STAGE 2: After SFT (Phase 1 - SARC training)")
    print("="*70)
    sft_path = "models/sft"
    if os.path.exists(sft_path):
        try:
            model, tokenizer = load_model_and_tokenizer(sft_path, is_adapter=True, base_model_name=base_model_name)
            results_sft, _ = evaluate_model(model, tokenizer, df_test, "SFT Model (SARC)")
            all_results.append(results_sft)
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"❌ Could not evaluate SFT model: {e}")
    else:
        print(f"⚠️  SFT model not found at {sft_path}")
        print("    Run 'python finetune_qwen.py' first")
    
    # Stage 3: After DPO
    print(f"\n{'='*70}")
    print("STAGE 3: After DPO (Phase 2 - iSarcasm refinement)")
    print("="*70)
    dpo_path = "models/dpo_enhanced"
    if os.path.exists(dpo_path):
        try:
            model, tokenizer = load_model_and_tokenizer(dpo_path, is_adapter=True, base_model_name=base_model_name)
            results_dpo, _ = evaluate_model(model, tokenizer, df_test, "DPO Model (iSarcasm Enhanced)")
            all_results.append(results_dpo)
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"❌ Could not evaluate DPO model: {e}")
    else:
        print(f"⚠️  DPO model not found at {dpo_path}")
        print("    Run 'python dpo_train.py' first")
    
    # Summary comparison
    print("\n" + "="*70)
    print("COMPARATIVE RESULTS SUMMARY")
    print("="*70)
    
    if len(all_results) > 0:
        print(f"\n{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 78)
        for result in all_results:
            print(f"{result['model_name']:<30} "
                  f"{result['accuracy']:<12.2%} "
                  f"{result['precision']:<12.2%} "
                  f"{result['recall']:<12.2%} "
                  f"{result['f1_score']:<12.2%}")
        
        # Calculate improvements
        if len(all_results) >= 2:
            print("\n" + "="*70)
            print("IMPROVEMENTS")
            print("="*70)
            base_acc = all_results[0]['accuracy']
            
            for i in range(1, len(all_results)):
                improvement = (all_results[i]['accuracy'] - base_acc) * 100
                print(f"{all_results[i]['model_name']}: +{improvement:.1f} percentage points from base")
        
        # Save all results
        comparison = {
            'evaluation_date': datetime.now().isoformat(),
            'test_set_size': len(df_test),
            'models': all_results
        }
        
        with open('comparative_results.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n✓ Results saved to comparative_results.json")
    else:
        print("No models were evaluated. Please train models first.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
