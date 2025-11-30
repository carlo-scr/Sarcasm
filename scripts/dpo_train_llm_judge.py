import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from huggingface_hub import InferenceClient
import re
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_MODEL_PATH = "models/sft"  # The model we'll generate TWO responses from
JUDGE_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
SAMPLE_SIZE = 50

# ============================================================================
# STEP 1: INITIALIZE JUDGE MODEL
# ============================================================================

def load_judge_model(model_name=JUDGE_MODEL_NAME, hf_token=None):
    """Initialize the judge model via Hugging Face Inference API."""
    print(f"🔧 Initializing judge model: {model_name}")
    
    if hf_token is None:
        raise ValueError("Please provide your HuggingFace API token!")
    
    client = InferenceClient(token=hf_token)
    print("✓ Judge model ready\n")
    
    return client, model_name


# ============================================================================
# STEP 2: LOAD THE MODEL THAT WILL GENERATE PREFERENCE PAIRS
# ============================================================================

def load_candidate_model(model_path, is_adapter=False, base_model_name=BASE_MODEL_NAME):
    """Load the model that will generate multiple responses."""
    print(f"  Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name if is_adapter else model_path
    )
    
    if is_adapter and os.path.exists(model_path):
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    model.eval()
    return model, tokenizer


# ============================================================================
# STEP 3: GENERATE TWO DIFFERENT RESPONSES FROM THE SAME MODEL
# ============================================================================

def generate_response(model, tokenizer, text, max_new_tokens=100, temperature=0.7, seed=None):
    """
    Generate a single response from the model.
    
    Args:
        model: The language model
        tokenizer: Model's tokenizer
        text: Input text to analyze
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more diverse)
        seed: Random seed for reproducibility
    
    Returns:
        str: Model's response
    """
    messages = [
        {"role": "user", "content": f"""Is the following text sarcastic? Sarcasm often involves irony, exaggeration, or saying the opposite of what is meant. 
Answer with 'Yes' or 'No' in the first line. Then, briefly explain your reasoning.

Text: {text}

Answer:"""}
    ]
    
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except:
        prompt = messages[0]['content']
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,  # IMPORTANT: Must be True to get different outputs
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    return response.strip()


def generate_response_pair(model, tokenizer, text, max_new_tokens=100):
    """
    Generate TWO different responses from the SAME model for DPO.
    
    THIS IS THE KEY FOR DPO:
    - Same model, same prompt
    - Different sampling = different outputs
    - Judge picks which one is better
    
    Args:
        model: The language model
        tokenizer: Model's tokenizer
        text: Input text
        max_new_tokens: Max tokens per response
    
    Returns:
        tuple: (response_1, response_2)
    """
    # Generate response 1 with temperature and seed
    response_1 = generate_response(
        model, tokenizer, text, 
        max_new_tokens=max_new_tokens,
        temperature=0.8,  # Higher temp for more diversity
        seed=42
    )
    
    # Generate response 2 with different seed for different output
    response_2 = generate_response(
        model, tokenizer, text,
        max_new_tokens=max_new_tokens, 
        temperature=0.8,
        seed=123  # Different seed = different response
    )
    
    return response_1, response_2


def collect_preference_pairs(df_sample, model, tokenizer):
    """
    Collect preference pairs for DPO training.
    
    For each sample:
    1. Generate response A from the model
    2. Generate response B from the SAME model (different sampling)
    3. Later, judge will pick which is better (preferred vs rejected)
    
    Args:
        df_sample: DataFrame with test samples
        model: The model to generate responses
        tokenizer: Model's tokenizer
    
    Returns:
        list: List of dicts with paired responses
    """
    preference_pairs = []
    
    print(f"\n{'='*70}")
    print("🎲 Generating response pairs for DPO")
    print(f"{'='*70}")
    
    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Generating Pairs"):
        # Generate TWO different responses from the same model
        response_a, response_b = generate_response_pair(model, tokenizer, row['tweet'])
        
        # Store both responses - judge will decide which is better
        preference_pairs.append({
            'index': idx,
            'prompt': row['tweet'],
            'true_label': row['sarcastic'],
            'response_a': response_a,
            'response_b': response_b,
            # These will be filled in by the judge
            'preferred': None,
            'rejected': None,
            'winner': None,
            'judge_reasoning': None
        })
    
    return preference_pairs


# ============================================================================
# STEP 4: CREATE JUDGE PROMPT FOR PREFERENCE SELECTION
# ============================================================================

def create_judge_prompt_for_dpo(text, true_label, response_a, response_b):
    """
    Create a prompt for the judge to pick the better response.
    Label text is provided so the judge can prioritize correctness.
    """
    label_text = "sarcastic" if true_label == 1 else "not sarcastic"
    
    prompt = f"""You are an expert judge evaluating sarcasm detection responses.

Text to analyze: "{text}"

Ground truth: The text is {label_text}.

Response A:
{response_a}

Response B:
{response_b}

Evaluation criteria (in order of importance):
1. Correctness: The response's Yes/No decision must match the ground truth.
2. Explanation quality: Clear, specific, and insightful reasoning.
3. Confidence calibration and clarity.

Instructions:
- Do NOT choose a response that contradicts the ground truth label unless both are wrong.
- If both are wrong, choose the one closer to correct and explain why.
- First line MUST be exactly: "Winner: A", "Winner: B", or "Winner: Tie". Do not answer anything else in the first line.
- On the next line, briefly explain your reasoning.

Now provide your judgment.
"""
    return prompt



# ============================================================================
# STEP 5: GET JUDGE VERDICTS
# ============================================================================

def get_judge_verdict(judge_client, judge_model_name, prompt, max_tokens=300):
    """Get the judge's verdict via Inference API."""
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = judge_client.chat_completion(
            model=judge_model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"⚠️ Error getting judge verdict: {e}")
        return None


def parse_judge_verdict(verdict_text):
    """
    Parse the judge's verdict to extract winner.
    
    Returns:
        str: 'A', 'B', 'Tie', or 'Error'
    """
    if verdict_text is None:
        return 'Error'
    
    verdict_lower = verdict_text.lower()
    
    # Method 1: Regex
    winner_match = re.search(r'winner:\s*(a|b|tie)', verdict_lower)
    if winner_match:
        result = winner_match.group(1).upper()
        return 'Tie' if result == 'TIE' else result
    
    # Method 2: Explicit patterns
    if 'winner: a' in verdict_lower or 'winner:a' in verdict_lower:
        return 'A'
    elif 'winner: b' in verdict_lower or 'winner:b' in verdict_lower:
        return 'B'
    elif 'winner: tie' in verdict_lower or 'winner:tie' in verdict_lower:
        return 'Tie'
    
    # Method 3: Fallback
    first_part = verdict_lower[:150]
    has_a = 'response a' in first_part
    has_b = 'response b' in first_part
    
    if has_a and not has_b:
        return 'A'
    elif has_b and not has_a:
        return 'B'
    
    print(f"⚠️ Unclear verdict: {verdict_text[:100]}")
    return 'Tie'


# ============================================================================
# STEP 6: JUDGE ALL PREFERENCE PAIRS
# ============================================================================

import random

def judge_preference_pairs(preference_pairs, judge_client, judge_model_name):
    """
    Have the judge evaluate all preference pairs.

    This creates the labeled data for DPO:
    - preferred: The better response (chosen by judge)
    - rejected: The worse response (not chosen by judge)
    """
    print(f"\n{'='*70}")
    print("⚖️  Judge evaluating preference pairs")
    print(f"{'='*70}")
    
    judged_pairs = []
    
    for pair in tqdm(preference_pairs, desc="Judge Evaluating"):
        # Randomize which response is shown as A vs B to reduce position bias
        if random.random() < 0.5:
            shown_a = pair['response_a']
            shown_b = pair['response_b']
            swapped = False
        else:
            shown_a = pair['response_b']
            shown_b = pair['response_a']
            swapped = True
        
        # Create comparison prompt with possibly swapped A/B
        prompt = create_judge_prompt_for_dpo(
            pair['prompt'],
            pair['true_label'],
            shown_a,
            shown_b
        )
        
        # Get judge's verdict
        verdict = get_judge_verdict(judge_client, judge_model_name, prompt)
        winner = parse_judge_verdict(verdict)
        
        # Map winner back to original responses, taking swapping into account
        if winner == 'A':
            preferred = pair['response_b'] if swapped else pair['response_a']
            rejected  = pair['response_a'] if swapped else pair['response_b']
        elif winner == 'B':
            preferred = pair['response_a'] if swapped else pair['response_b']
            rejected  = pair['response_b'] if swapped else pair['response_a']
        else:
            # Winner == 'Tie' or 'Error' -> keep but mark explicitly
            preferred = None
            rejected = None
        
        judged_pairs.append({
            'index': pair['index'],
            'prompt': pair['prompt'],
            'true_label': pair['true_label'],
            'response_a': pair['response_a'],
            'response_b': pair['response_b'],
            'winner': winner,
            'preferred': preferred,
            'rejected': rejected,
            'swapped': swapped,
            'judge_reasoning': verdict
        })
    
    return judged_pairs


# ============================================================================
# STEP 7: CREATE DPO TRAINING DATASET
# ============================================================================

def extract_binary_label_from_response(response_text):
    """
    Extract a binary prediction (1 = sarcastic, 0 = not sarcastic) from a model response.
    Very simple heuristic: look for leading 'yes'/'no'.
    """
    if response_text is None:
        return None
    text = response_text.strip().lower()
    # look only at the first few characters / words
    first = text[:30]
    if "yes" in first:
        return 1
    if "no" in first:
        return 0
    return None


def create_dpo_dataset(judged_pairs, filter_ties=True):
    """
    Convert judged pairs into DPO training format.
    Also log whether preferred prediction matches the ground truth.
    """
    dpo_data = []
    
    for pair in judged_pairs:
        # Skip ties or errors if requested
        if filter_ties and pair['winner'] in ['Tie', 'Error']:
            continue
        
        if pair['winner'] not in ['A', 'B']:
            continue
        
        preferred = pair['preferred']
        rejected = pair['rejected']
        if preferred is None or rejected is None:
            continue
        
        # Optional: compute whether preferred response label matches ground truth
        y_pref = extract_binary_label_from_response(preferred)
        label_match = (y_pref == pair['true_label']) if y_pref is not None else None
        
        dpo_data.append({
            'prompt': pair['prompt'],
            'chosen': preferred,
            'rejected': rejected,
            'true_label': pair['true_label'],
            'preferred_matches_label': label_match
        })
    
    df_dpo = pd.DataFrame(dpo_data)
    
    print(f"\n{'='*70}")
    print("📊 DPO Dataset Statistics")
    print(f"{'='*70}")
    print(f"Total judged pairs: {len(judged_pairs)}")
    print(f"Used pairs (clear A/B winner): {len(df_dpo)}")
    print(f"Ties: {len([p for p in judged_pairs if p['winner'] == 'Tie'])}")
    print(f"Errors: {len([p for p in judged_pairs if p['winner'] == 'Error'])}")
    if 'preferred_matches_label' in df_dpo.columns:
        n_known = df_dpo['preferred_matches_label'].notnull().sum()
        if n_known > 0:
            acc = df_dpo['preferred_matches_label'].dropna().mean()
            print(f"Judge-chosen responses matching label (on parsed cases): {acc:.3f}")
    
    return df_dpo



# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_dpo_preference_data(hf_token, test_csv_path='data/splits/isarcasm_test.csv', 
                                  model_path=SFT_MODEL_PATH, is_adapter=True):
    """
    Complete pipeline to generate DPO preference data.
    
    Steps:
    1. Load your SFT model
    2. Generate TWO responses per prompt
    3. Have judge pick which is better
    4. Create DPO training dataset
    
    Args:
        hf_token: Your HuggingFace API token
        test_csv_path: Path to test data
        model_path: Path to your SFT model
        is_adapter: Whether model is a LoRA adapter
    
    Returns:
        tuple: (dpo_dataset, all_judged_pairs)
    """
    # Load test data
    print("📁 Loading test data...")
    df_test = pd.read_csv(test_csv_path, index_col=0)
    
    df_sample = df_test.sample(n=min(SAMPLE_SIZE, len(df_test)), random_state=42)
    print(f"🎯 Generating preference pairs for {len(df_sample)} samples\n")
    
    # Initialize judge
    judge_client, judge_model_name = load_judge_model(hf_token=hf_token)
    
    # Load the model that will generate pairs
    print("🤖 Loading candidate model for response generation...")
    model, tokenizer = load_candidate_model(model_path, is_adapter)
    
    # Generate response pairs
    preference_pairs = collect_preference_pairs(df_sample, model, tokenizer)
    
    # Clean up model memory
    del model, tokenizer
    torch.cuda.empty_cache()
    
    # Have judge evaluate all pairs
    judged_pairs = judge_preference_pairs(preference_pairs, judge_client, judge_model_name)
    
    # Create DPO dataset
    df_dpo = create_dpo_dataset(judged_pairs, filter_ties=True)
    
    return df_dpo, judged_pairs


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Set your HuggingFace token
    HF_TOKEN = os.environ.get("HF_TOKEN")
    
    # Generate DPO preference data
    dpo_dataset, all_pairs = generate_dpo_preference_data(HF_TOKEN)
    
    # Save the DPO training dataset
    dpo_dataset.to_csv('data/dpo_preferences_dt.csv', index=False)
    print(f"\n💾 Saved DPO dataset: {len(dpo_dataset)} preference pairs")
    
    # Save all judged pairs for analysis
    pd.DataFrame(all_pairs).to_csv('data/all_judged_pairs_dt.csv', index=False)
    print(f"💾 Saved all judged pairs for analysis")
    
    # Display sample
    print("\n📋 Sample DPO training data:")
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    print(dpo_dataset.head(3))
    
    print("\n✅ DPO preference data generation complete!")
    print("\nNext steps:")
    print("1. Use this data to train with DPO")
    print("2. The 'chosen' responses are what the model should learn to prefer")
    print("3. The 'rejected' responses are what it should learn to avoid")