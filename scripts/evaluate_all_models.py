"""
Unified evaluation script for BERT and LLM models (Qwen, MobileLLM)
Evaluates on 500 samples from GEN test set
"""

import json
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from tqdm import tqdm

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
NUM_SAMPLES = 500
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Model configurations
MODELS = {
    'bert_sarcasm': {
        'type': 'bert',
        'path': PROJECT_ROOT / "models" / "bert_sarcasm"
    },
    'qwen_dpo_mistakes': {
        'type': 'llm',
        'base': 'Qwen/Qwen2.5-0.5B-Instruct',
        'path': PROJECT_ROOT / "models" / "qwen_dpo_mistakes"
    }
}

def load_test_data(num_samples=500):
    """Load and sample test data"""
    print(f"\nLoading test data from {DATA_DIR}...")
    test_df = pd.read_csv(DATA_DIR / "gen_test.csv")
    
    if num_samples and num_samples < len(test_df):
        test_df = test_df.sample(n=num_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled {num_samples} examples from test set")
    
    # Convert labels
    label_map = {"notsarc": 0, "sarc": 1}
    test_df['label'] = test_df['class'].map(label_map)
    
    print(f"Test samples: {len(test_df)}")
    print(f"  Sarcastic: {(test_df['label'] == 1).sum()}")
    print(f"  Non-sarcastic: {(test_df['label'] == 0).sum()}")
    
    return test_df

def evaluate_bert_model(model_path, test_df):
    """Evaluate BERT-based model"""
    print(f"\nLoading BERT model from {model_path}...")
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    
    predictions = []
    labels = test_df['label'].values
    
    print("Evaluating...")
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
            text = str(row['text'])
            
            inputs = tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred)
    
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return labels, predictions

def evaluate_llm_model(base_model, adapter_path, test_df):
    """Evaluate LLM-based model (Qwen, MobileLLM)"""
    print(f"\nLoading LLM model...")
    print(f"  Base: {base_model}")
    print(f"  Adapter: {adapter_path}")
    
    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load adapter if path exists
    if adapter_path.exists():
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        print(f"⚠️  Adapter not found at {adapter_path}, using base model only")
        model = base
    
    model.eval()
    
    predictions = []
    labels = test_df['label'].values
    
    print("Evaluating...")
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
            text = str(row['text'])
            
            prompt = f"""Is the following text sarcastic? Answer with only 'Yes' or 'No'.

Text: {text}

Answer:"""
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip().lower()
            else:
                answer = response.strip().lower()
            
            # Parse prediction
            first_word = answer.split()[0] if answer.split() else ""
            if first_word.startswith("yes"):
                pred = 1
            elif first_word.startswith("no"):
                pred = 0
            else:
                # Fallback
                pred = 1 if "yes" in answer[:20] else 0
            
            predictions.append(pred)
    
    del model, base, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return labels, predictions

def calculate_metrics(labels, predictions):
    """Calculate metrics"""
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

def main():
    print("=" * 80)
    print("Multi-Model Evaluation: BERT + LLM Models")
    print("=" * 80)
    
    # Load test data once
    test_df = load_test_data(num_samples=NUM_SAMPLES)
    
    all_results = {}
    
    # Evaluate each model
    for model_name, config in MODELS.items():
        print("\n" + "=" * 80)
        print(f"Evaluating: {model_name}")
        print("=" * 80)
        
        try:
            if config['type'] == 'bert':
                if not config['path'].exists():
                    print(f"⚠️  Model not found at {config['path']}, skipping...")
                    continue
                
                labels, predictions = evaluate_bert_model(config['path'], test_df)
            
            elif config['type'] == 'llm':
                if not config['path'].exists():
                    print(f"⚠️  Adapter not found at {config['path']}, skipping...")
                    continue
                
                labels, predictions = evaluate_llm_model(config['base'], config['path'], test_df)
            
            else:
                print(f"Unknown model type: {config['type']}")
                continue
            
            # Calculate metrics
            metrics = calculate_metrics(labels, predictions)
            
            print(f"\n{'='*60}")
            print(f"RESULTS: {model_name.upper()}")
            print(f"{'='*60}")
            print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            
            # Save results
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            results_file = RESULTS_DIR / f"{model_name}_eval.json"
            
            output_data = {
                'model_name': model_name,
                'evaluation_date': datetime.now().isoformat(),
                'test_samples': len(labels),
                'metrics': metrics
            }
            
            with open(results_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"  ✓ Results saved to: {results_file}")
            
            all_results[model_name] = metrics
        
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Comparison summary
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        
        print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 80)
        
        for model_name, metrics in all_results.items():
            print(f"{model_name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
        
        # Find best model
        best_model = max(all_results.items(), key=lambda x: x[1]['f1'])
        print(f"\n🏆 Best Model (by F1): {best_model[0]} (F1: {best_model[1]['f1']:.4f})")
    
    elif all_results:
        model_name = list(all_results.keys())[0]
        metrics = all_results[model_name]
        print(f"\n✅ Evaluation complete!")
        print(f"  {model_name}: F1 = {metrics['f1']:.4f}")
    
    else:
        print("\n❌ No models were evaluated.")

if __name__ == "__main__":
    main()
