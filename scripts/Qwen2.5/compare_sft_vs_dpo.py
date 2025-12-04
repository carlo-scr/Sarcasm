"""
Compare SFT vs DPO Model Performance
=====================================
Analyze specific cases where DPO classifies better (or worse) than SFT.
Stops after finding one example where DPO is correct and SFT is wrong.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROJECT_ROOT = Path(__file__).parent.parent.parent
SFT_ADAPTER = str(PROJECT_ROOT / "models" / "sft")
DPO_ADAPTER = str(PROJECT_ROOT / "models" / "qwen_dpo_mistakes")
TEST_DATA = str(PROJECT_ROOT / "data" / "splits" / "gen_test.csv")
OUTPUT_FILE = str(PROJECT_ROOT / "results" / "sft_vs_dpo_comparison.json")
OUTPUT_CSV = str(PROJECT_ROOT / "results" / "sft_vs_dpo_first_50.csv")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print("="*70)

# Load test data
print("\n📂 Loading test data...")
test_df = pd.read_csv(TEST_DATA)
print(f"   Test set size: {len(test_df)}")
print(f"   Class distribution: {test_df['class'].value_counts().to_dict()}")

# Load tokenizer
print("\n🔧 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Function to load model with adapter
def load_model_with_adapter(adapter_path, model_name="model"):
    """Load base model and apply LoRA adapter."""
    print(f"\n🤖 Loading {model_name}...")
    print(f"   Base model: {BASE_MODEL}")
    print(f"   Adapter: {adapter_path}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map=DEVICE
    )
    
    # Load adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    print(f"   ✓ {model_name} loaded successfully")
    return model

# Load both models
sft_model = load_model_with_adapter(SFT_ADAPTER, "SFT Model")
dpo_model = load_model_with_adapter(DPO_ADAPTER, "DPO Model")

# Function to classify with a model (using chat template like in evaluate script)
def get_prediction(model, tokenizer, text):
    """Get model prediction for a single text using chat template with confidence score."""
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
        # Get logits for confidence calculation
        outputs_with_scores = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )
        
        outputs = outputs_with_scores.sequences
        scores = outputs_with_scores.scores
    
    # Get only the generated response (remove prompt)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    response_clean = response.strip().lower()
    
    # Calculate confidence from first token (Yes/No decision)
    if len(scores) > 0:
        first_token_logits = scores[0][0]  # Logits for first generated token
        first_token_probs = torch.softmax(first_token_logits, dim=-1)
        
        # Get tokens for "Yes" and "No"
        yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
        
        yes_prob = first_token_probs[yes_token_id].item()
        no_prob = first_token_probs[no_token_id].item()
        
        # Normalize between Yes/No only
        total_prob = yes_prob + no_prob
        if total_prob > 0:
            yes_confidence = (yes_prob / total_prob) * 100
            no_confidence = (no_prob / total_prob) * 100
        else:
            yes_confidence = 0.0
            no_confidence = 0.0
    else:
        yes_confidence = 0.0
        no_confidence = 0.0
    
    # Handle thinking tags if present
    if '</think>' in response_clean:
        response_clean = response_clean.split('</think>')[-1].strip()
    
    # Parse response - Yes=sarcastic(1), No=not sarcastic(0)
    if 'yes' in response_clean:
        return 1, response, yes_confidence
    elif 'no' in response_clean:
        return 0, response, no_confidence
    else:
        return -1, response, max(yes_confidence, no_confidence)  # unclear

# Run evaluation with early stopping
print("\n🔍 Running comparative evaluation...")
print("   Goal: Find ONE case where DPO is correct and SFT is wrong")
print("="*70)

found_example = False
example_case = None
first_50_results = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Searching"):
    text = row['text']
    true_label = 1 if row['class'] == 'sarc' else 0
    
    # Get predictions with confidence
    sft_pred, sft_response, sft_confidence = get_prediction(sft_model, tokenizer, text)
    dpo_pred, dpo_response, dpo_confidence = get_prediction(dpo_model, tokenizer, text)
    
    # Store first 30 results
    if idx < 50:
        first_50_results.append({
            'text': text,
            'true_label': row['class'],
            'sft_prediction': 'Yes' if sft_pred == 1 else 'No' if sft_pred == 0 else 'Unclear',
            'sft_confidence': round(sft_confidence, 2),
            'sft_response': sft_response,
            'dpo_prediction': 'Yes' if dpo_pred == 1 else 'No' if dpo_pred == 0 else 'Unclear',
            'dpo_confidence': round(dpo_confidence, 2),
            'dpo_response': dpo_response
        })
    
    # Check if this is the case we're looking for
    sft_correct = (sft_pred == true_label)
    dpo_correct = (dpo_pred == true_label)
    
    # Print first few for debugging
    if idx < 3:
        print(f"\n[Sample {idx+1}]")
        print(f"  Text: {text[:100]}...")
        print(f"  True: {'sarcastic' if true_label == 1 else 'not sarcastic'}")
        print(f"  SFT pred: {sft_pred} ({'Yes' if sft_pred == 1 else 'No' if sft_pred == 0 else 'Unclear'}) [{sft_confidence:.1f}%] - {'✓' if sft_correct else '✗'}")
        print(f"  DPO pred: {dpo_pred} ({'Yes' if dpo_pred == 1 else 'No' if dpo_pred == 0 else 'Unclear'}) [{dpo_confidence:.1f}%] - {'✓' if dpo_correct else '✗'}")
    
    # Found it! DPO correct, SFT wrong
    if dpo_correct and not sft_correct:
        example_case = {
            'index': idx,
            'text': text,
            'true_label': row['class'],
            'true_label_binary': true_label,
            'sft_prediction': sft_pred,
            'sft_response': sft_response,
            'sft_confidence': round(sft_confidence, 2),
            'sft_correct': sft_correct,
            'dpo_prediction': dpo_pred,
            'dpo_response': dpo_response,
            'dpo_confidence': round(dpo_confidence, 2),
            'dpo_correct': dpo_correct
        }
        found_example = True
        print(f"\n✅ FOUND EXAMPLE at sample {idx+1}!")
        # break
    if idx > 50:
        break

print("\n" + "="*70)
if found_example:
    print("🎯 SUCCESS: Found case where DPO outperforms SFT")
    print("="*70)
    
    case = example_case
    print(f"\nSample Index: {case['index']}")
    print(f"\nText:")
    print(f"  {case['text']}")
    print(f"\nTrue Label: {case['true_label'].upper()}")
    print(f"\nSFT:")
    print(f"  Prediction: {'Sarcastic (Yes)' if case['sft_prediction'] == 1 else 'Not Sarcastic (No)'} ❌")
    print(f"  Confidence: {case['sft_confidence']:.2f}%")
    print(f"  Raw response: {case['sft_response'][:150]}...")
    print(f"\nDPO:")
    print(f"  Prediction: {'Sarcastic (Yes)' if case['dpo_prediction'] == 1 else 'Not Sarcastic (No)'} ✓")
    print(f"  Confidence: {case['dpo_confidence']:.2f}%")
    print(f"  Raw response: {case['dpo_response'][:150]}...")
    
    # Save result
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'evaluation_date': pd.Timestamp.now().isoformat(),
        'test_set': TEST_DATA,
        'found_dpo_better_example': True,
        'example': case,
        'samples_checked': case['index'] + 1
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
else:
    print("⚠️  No example found where DPO outperforms SFT")
    print("="*70)
    print(f"Checked all {len(test_df)} samples")
    
    # Save result
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'evaluation_date': pd.Timestamp.now().isoformat(),
        'test_set': TEST_DATA,
        'found_dpo_better_example': False,
        'samples_checked': len(test_df)
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")

# Save first 30 results to CSV
csv_output_path = Path(OUTPUT_CSV)
csv_output_path.parent.mkdir(parents=True, exist_ok=True)
first_50_df = pd.DataFrame(first_50_results)
first_50_df.to_csv(csv_output_path, index=False)
print(f"✓ First 30 results saved to: {csv_output_path}")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE")
print("="*70)
