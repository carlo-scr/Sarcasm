"""
Mine hard negatives from SFT model predictions.
This script finds examples where the SFT model is confidently wrong,
and adds them as high-value training pairs for DPO.
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import json

def load_sft_model(base_model_name="Qwen/Qwen2.5-0.5B-Instruct", adapter_path="models/sft"):
    """Load the SFT model."""
    print(f"Loading SFT model from: {adapter_path}")
    
    # Auto-detect device and set optimal dtype
    if torch.cuda.is_available():
        dtype = torch.float16
        device_info = "CUDA"
    elif torch.backends.mps.is_available():
        dtype = torch.float32
        device_info = "MPS"
    else:
        dtype = torch.float32
        device_info = "CPU"
    print(f"  Device: {device_info}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model, tokenizer

def get_prediction_with_confidence(model, tokenizer, text):
    """Get model prediction and confidence score."""
    messages = [
        {"role": "user", "content": f"Is the following text sarcastic? Answer with only 'Yes' or 'No'.\n\nText: {text}\n\nAnswer:"}
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
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        # Get logits for confidence scoring
        outputs_logits = model(**inputs)
        
        # Generate prediction
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    response = tokenizer.decode(outputs.sequences[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    response_clean = response.strip().lower()
    
    # Handle thinking tags
    if '</think>' in response_clean:
        response_clean = response_clean.split('</think>')[-1].strip()
    
    # Parse response
    if 'yes' in response_clean:
        prediction = 1
    elif 'no' in response_clean:
        prediction = 0
    else:
        prediction = -1
    
    # Calculate confidence from first token probability
    if len(outputs.scores) > 0:
        first_token_scores = outputs.scores[0][0]  # [vocab_size]
        probs = torch.softmax(first_token_scores, dim=0)
        confidence = probs.max().item()
    else:
        confidence = 0.5
    
    return prediction, confidence, response

def mine_hard_negatives(train_csv_path, output_path="data/hard_negatives.json", confidence_threshold=0.6):
    """
    Find examples where SFT model is confidently wrong.
    
    Args:
        train_csv_path: Path to training data
        output_path: Where to save hard negatives
        confidence_threshold: Minimum confidence to consider (0-1)
    """
    print("="*70)
    print("HARD NEGATIVE MINING FROM SFT MODEL")
    print("="*70)
    
    # Load model
    model, tokenizer = load_sft_model()
    
    # Load training data
    df = pd.read_csv(train_csv_path, index_col=0)
    print(f"\nAnalyzing {len(df)} training examples...")
    
    hard_negatives = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Mining"):
        text = row['tweet']
        true_label = row['sarcastic']
        
        pred, confidence, response = get_prediction_with_confidence(model, tokenizer, text)
        
        # Skip unclear predictions
        if pred == -1:
            continue
        
        # Find high-confidence mistakes
        if pred != true_label and confidence >= confidence_threshold:
            # Handle NaN values in sarcasm type columns
            def safe_int(val):
                """Convert to int, handling NaN values."""
                try:
                    if pd.isna(val):
                        return 0
                    return int(val)
                except (ValueError, TypeError):
                    return 0
            
            hard_negatives.append({
                'text': text,
                'true_label': int(true_label),
                'predicted_label': int(pred),
                'confidence': float(confidence),
                'model_response': response,
                'sarcasm_type': {
                    'irony': safe_int(row.get('irony', 0)),
                    'satire': safe_int(row.get('satire', 0)),
                    'overstatement': safe_int(row.get('overstatement', 0)),
                    'understatement': safe_int(row.get('understatement', 0)),
                    'rhetorical_question': safe_int(row.get('rhetorical_question', 0))
                }
            })
    
    print(f"\n✓ Found {len(hard_negatives)} hard negatives (confidence >= {confidence_threshold})")
    
    # Sort by confidence (highest mistakes first)
    hard_negatives.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(hard_negatives, f, indent=2)
    
    print(f"✓ Saved to: {output_path}")
    
    # Print statistics
    false_positives = sum(1 for hn in hard_negatives if hn['predicted_label'] == 1)
    false_negatives = sum(1 for hn in hard_negatives if hn['predicted_label'] == 0)
    
    print(f"\nBreakdown:")
    print(f"  False Positives (said 'sarcastic' but wasn't): {false_positives}")
    print(f"  False Negatives (said 'not sarcastic' but was): {false_negatives}")
    print(f"  Avg Confidence: {sum(hn['confidence'] for hn in hard_negatives) / len(hard_negatives):.2%}")
    
    # Show examples
    print(f"\nTop 3 Most Confident Mistakes:")
    for i, hn in enumerate(hard_negatives[:3], 1):
        print(f"\n  {i}. [{hn['confidence']:.1%} confident]")
        print(f"     Text: {hn['text'][:80]}...")
        print(f"     Predicted: {'Sarcastic' if hn['predicted_label'] == 1 else 'Not Sarcastic'}")
        print(f"     Actually: {'Sarcastic' if hn['true_label'] == 1 else 'Not Sarcastic'}")
    
    return hard_negatives

def main():
    train_csv_path = "data/splits/isarcasm_train.csv"
    output_path = "data/hard_negatives.json"
    
    hard_negatives = mine_hard_negatives(
        train_csv_path,
        output_path,
        confidence_threshold=0.6  # Only high-confidence mistakes
    )
    
    print("\n" + "="*70)
    print("Mining Complete!")
    print("="*70)
    print(f"Use these {len(hard_negatives)} examples for targeted DPO training")

if __name__ == "__main__":
    main()
