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
Answer with 'Yes' or 'No' and briefly explain your reasoning.

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
    
    For DPO, we need to know:
    - Which response is PREFERRED (better quality)
    - Which response is REJECTED (worse quality)
    
    Args:
        text: Original input text
        true_label: Ground truth (0 or 1)
        response_a: First response from model
        response_b: Second response from model
    
    Returns:
        str: Formatted judge prompt
    """
    label_text = "sarcastic" if true_label == 1 else "not sarcastic"
    
    prompt = f"""You are an expert judge evaluating sarcasm detection responses. Your task is to compare two responses from the SAME model and determine which one is better.

**Text to analyze:** "{text}"

**Ground Truth:** This text is {label_text}.

**Response A:**
{response_a}

**Response B:**
{response_b}

**Evaluation Criteria:**
1. **Correctness**: Does the response match the ground truth?
2. **Reasoning Quality**: Is the explanation clear, specific, and insightful?
3. **Confidence Calibration**: Is the model appropriately confident?
4. **Explanation Depth**: Does it identify specific sarcasm indicators (irony, exaggeration, context clues)?
5. **Clarity**: Is it easy to understand?

**Your Task:**
- Compare both responses carefully
- Determine which response is BETTER overall
- If both are equally good/bad, choose "Tie"
- Explain your reasoning in 2-3 sentences
- Format: "Winner: [A/B/Tie]" followed by reasoning

**Your Judgment:**
Winner:"""
    
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

def judge_preference_pairs(preference_pairs, judge_client, judge_model_name):
    """
    Have the judge evaluate all preference pairs.
    
    This creates the labeled data for DPO:
    - preferred: The better response (chosen by judge)
    - rejected: The worse response (not chosen by judge)
    
    Args:
        preference_pairs: List of response pairs
        judge_client: HuggingFace client
        judge_model_name: Judge model name
    
    Returns:
        list: Preference pairs with judge labels
    """
    print(f"\n{'='*70}")
    print("⚖️  Judge evaluating preference pairs")
    print(f"{'='*70}")
    
    judged_pairs = []
    
    for pair in tqdm(preference_pairs, desc="Judge Evaluating"):
        # Create comparison prompt
        prompt = create_judge_prompt_for_dpo(
            pair['prompt'],
            pair['true_label'],
            pair['response_a'],
            pair['response_b']
        )
        
        # Get judge's verdict
        verdict = get_judge_verdict(judge_client, judge_model_name, prompt)
        winner = parse_judge_verdict(verdict)
        
        # Assign preferred and rejected based on winner
        if winner == 'A':
            preferred = pair['response_a']
            rejected = pair['response_b']
        elif winner == 'B':
            preferred = pair['response_b']
            rejected = pair['response_a']
        else:  # Tie or Error
            # For ties, randomly assign or skip
            # For DPO, you typically want clear preferences
            preferred = pair['response_a']
            rejected = pair['response_b']
        
        judged_pairs.append({
            'index': pair['index'],
            'prompt': pair['prompt'],
            'true_label': pair['true_label'],
            'response_a': pair['response_a'],
            'response_b': pair['response_b'],
            'winner': winner,
            'preferred': preferred,
            'rejected': rejected,
            'judge_reasoning': verdict
        })
    
    return judged_pairs


# ============================================================================
# STEP 7: CREATE DPO TRAINING DATASET
# ============================================================================

def create_dpo_dataset(judged_pairs, filter_ties=True):
    """
    Convert judged pairs into DPO training format.
    
    DPO expects:
    - prompt: The input
    - chosen: The preferred response
    - rejected: The worse response
    
    Args:
        judged_pairs: List of judged preference pairs
        filter_ties: Whether to exclude ties from training data
    
    Returns:
        DataFrame: Ready for DPO training
    """
    dpo_data = []
    
    for pair in judged_pairs:
        # Skip ties if filtering
        if filter_ties and pair['winner'] == 'Tie':
            continue
        
        # Skip errors
        if pair['winner'] == 'Error':
            continue
        
        # Format for DPO training
        dpo_data.append({
            'prompt': pair['prompt'],
            'chosen': pair['preferred'],
            'rejected': pair['rejected'],
            'true_label': pair['true_label']
        })
    
    df_dpo = pd.DataFrame(dpo_data)
    
    print(f"\n{'='*70}")
    print("📊 DPO Dataset Statistics")
    print(f"{'='*70}")
    print(f"Total pairs: {len(dpo_data)}")
    print(f"Pairs with clear preference: {len([p for p in judged_pairs if p['winner'] in ['A', 'B']])}")
    print(f"Ties: {len([p for p in judged_pairs if p['winner'] == 'Tie'])}")
    print(f"Errors: {len([p for p in judged_pairs if p['winner'] == 'Error'])}")
    
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
    HF_TOKEN = "hf_YOUR_TOKEN_HERE"
    
    # Generate DPO preference data
    dpo_dataset, all_pairs = generate_dpo_preference_data(HF_TOKEN)
    
    # Save the DPO training dataset
    dpo_dataset.to_csv('data/dpo_preferences.csv', index=False)
    print(f"\n💾 Saved DPO dataset: {len(dpo_dataset)} preference pairs")
    
    # Save all judged pairs for analysis
    pd.DataFrame(all_pairs).to_csv('data/all_judged_pairs.csv', index=False)
    print(f"💾 Saved all judged pairs for analysis")
    
    # Display sample
    print("\n📋 Sample DPO training data:")
    print(dpo_dataset.head(3))
    
    print("\n✅ DPO preference data generation complete!")
    print("\nNext steps:")
    print("1. Use this data to train with DPO")
    print("2. The 'chosen' responses are what the model should learn to prefer")
    print("3. The 'rejected' responses are what it should learn to avoid")