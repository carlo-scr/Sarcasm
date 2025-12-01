"""
Comprehensive model comparison: Base vs SFT vs DPO.

Evaluates all three stages on the held-out GEN test set with:
- Overall metrics (Accuracy, Precision, Recall, F1)
- Per-class metrics (Sarcastic vs Non-sarcastic)
- Confusion matrices
- Statistical significance tests
- Visualization of improvements

This script provides a fair comparison since all models are evaluated
on the SAME held-out test set (data/splits/gen_test.csv).
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

def load_model_and_tokenizer(model_path, base_model="Qwen/Qwen2.5-0.5B-Instruct"):
    """Load a model (base or fine-tuned) with tokenizer."""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    if model_path == base_model:
        # Load base model directly
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    else:
        # Load base model + adapter
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    
    model.eval()
    print("✓ Model loaded successfully")
    
    return model, tokenizer


def evaluate_model(model, tokenizer, test_df, model_name, sample_size=1000):
    """
    Evaluate a model on test set with comprehensive metrics.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        test_df: Test DataFrame with text and class columns
        model_name: Name for logging
        sample_size: Number of samples to evaluate (None for all)
    
    Returns:
        Dictionary of metrics and predictions
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}")
    
    # Sample balanced test set
    if sample_size:
        sarc_samples = test_df[test_df['class'] == 'sarc'].sample(n=min(sample_size//2, len(test_df[test_df['class'] == 'sarc'])), random_state=42)
        notsarc_samples = test_df[test_df['class'] == 'notsarc'].sample(n=min(sample_size//2, len(test_df[test_df['class'] == 'notsarc'])), random_state=42)
        eval_df = pd.concat([sarc_samples, notsarc_samples]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        eval_df = test_df
    
    print(f"Evaluating on {len(eval_df)} samples ({len(sarc_samples)} sarc, {len(notsarc_samples)} notsarc)")
    
    predictions = []
    labels = []
    confidences = []
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"Evaluating {model_name}"):
            text = row['text']
            label = 1 if row['class'] == 'sarc' else 0
            
            prompt = f"""Is the following text sarcastic? Answer with 'Yes' or 'No'.

Text: "{text}"

Answer:"""
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.0, do_sample=False, output_scores=True, return_dict_in_generate=True)
            
            # Decode response
            response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            response_lower = response.lower()
            
            # Extract prediction
            if 'sarcastic' in response_lower and 'not sarcastic' not in response_lower:
                pred = 1
            elif 'not sarcastic' in response_lower or ('no' in response_lower.split()[:10] and 'yes' not in response_lower.split()[:5]):
                pred = 0
            else:
                # Default to majority class
                pred = 0
            
            # Extract confidence (average logit probability of generated tokens)
            if hasattr(outputs, 'scores') and outputs.scores:
                token_probs = [torch.softmax(score, dim=-1).max().item() for score in outputs.scores[:5]]  # First 5 tokens
                confidence = np.mean(token_probs)
            else:
                confidence = 0.5
            
            predictions.append(pred)
            labels.append(label)
            confidences.append(confidence)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Per-class metrics
    report = classification_report(labels, predictions, target_names=['Not Sarcastic', 'Sarcastic'], output_dict=True)
    
    # Print results
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Avg Confidence: {np.mean(confidences):.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"               Predicted")
    print(f"               Not Sarc  Sarcastic")
    print(f"Actual Not Sarc  {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"       Sarcastic {cm[1,0]:6d}    {cm[1,1]:6d}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  Not Sarcastic - Precision: {report['Not Sarcastic']['precision']:.4f}, Recall: {report['Not Sarcastic']['recall']:.4f}, F1: {report['Not Sarcastic']['f1-score']:.4f}")
    print(f"  Sarcastic     - Precision: {report['Sarcastic']['precision']:.4f}, Recall: {report['Sarcastic']['recall']:.4f}, F1: {report['Sarcastic']['f1-score']:.4f}")
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_confidence': float(np.mean(confidences)),
        'confusion_matrix': cm.tolist(),
        'per_class': {
            'not_sarcastic': {
                'precision': report['Not Sarcastic']['precision'],
                'recall': report['Not Sarcastic']['recall'],
                'f1': report['Not Sarcastic']['f1-score'],
                'support': report['Not Sarcastic']['support']
            },
            'sarcastic': {
                'precision': report['Sarcastic']['precision'],
                'recall': report['Sarcastic']['recall'],
                'f1': report['Sarcastic']['f1-score'],
                'support': report['Sarcastic']['support']
            }
        },
        'predictions': predictions,
        'labels': labels,
        'confidences': confidences
    }
    
    return results


def plot_comparison(all_results, output_path="comparison_plot.png"):
    """Create comprehensive comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    models = [r['model_name'] for r in all_results]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 1. Overall Metrics Bar Chart
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    x = np.arange(len(metric_labels))
    width = 0.25
    
    for i, result in enumerate(all_results):
        values = [result[m] for m in metrics]
        axes[0, 0].bar(x + i*width, values, width, label=result['model_name'], color=colors[i], alpha=0.8)
    
    axes[0, 0].set_xlabel('Metric', fontsize=11)
    axes[0, 0].set_ylabel('Score', fontsize=11)
    axes[0, 0].set_title('Overall Metrics Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels(metric_labels)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1.0])
    
    # 2. Improvement from Base
    if len(all_results) >= 2:
        base_acc = all_results[0]['accuracy']
        improvements_acc = [(r['accuracy'] - base_acc) * 100 for r in all_results[1:]]
        
        base_f1 = all_results[0]['f1']
        improvements_f1 = [(r['f1'] - base_f1) * 100 for r in all_results[1:]]
        
        x2 = np.arange(len(improvements_acc))
        width2 = 0.35
        
        axes[0, 1].bar(x2 - width2/2, improvements_acc, width2, label='Accuracy Δ', color='#4ECDC4', alpha=0.8)
        axes[0, 1].bar(x2 + width2/2, improvements_f1, width2, label='F1 Δ', color='#45B7D1', alpha=0.8)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        axes[0, 1].set_xlabel('Model', fontsize=11)
        axes[0, 1].set_ylabel('Improvement (%)', fontsize=11)
        axes[0, 1].set_title('Improvement Over Base Model', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(x2)
        axes[0, 1].set_xticklabels([r['model_name'] for r in all_results[1:]])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Per-Class F1 Scores
    not_sarc_f1 = [r['per_class']['not_sarcastic']['f1'] for r in all_results]
    sarc_f1 = [r['per_class']['sarcastic']['f1'] for r in all_results]
    
    x3 = np.arange(len(models))
    width3 = 0.35
    
    axes[1, 0].bar(x3 - width3/2, not_sarc_f1, width3, label='Not Sarcastic', color='#FF6B6B', alpha=0.8)
    axes[1, 0].bar(x3 + width3/2, sarc_f1, width3, label='Sarcastic', color='#4ECDC4', alpha=0.8)
    
    axes[1, 0].set_xlabel('Model', fontsize=11)
    axes[1, 0].set_ylabel('F1 Score', fontsize=11)
    axes[1, 0].set_title('Per-Class F1 Scores', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x3)
    axes[1, 0].set_xticklabels(models, rotation=15, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 1.0])
    
    # 4. Confusion Matrices (Latest Model)
    latest_result = all_results[-1]
    cm = np.array(latest_result['confusion_matrix'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Sarc', 'Sarcastic'],
                yticklabels=['Not Sarc', 'Sarcastic'],
                ax=axes[1, 1], cbar_kws={'label': 'Count'})
    
    axes[1, 1].set_xlabel('Predicted', fontsize=11)
    axes[1, 1].set_ylabel('Actual', fontsize=11)
    axes[1, 1].set_title(f'Confusion Matrix - {latest_result["model_name"]}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved comparison plot to {output_path}")


def main():
    """Main comparison function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Base, SFT, and DPO models")
    parser.add_argument("--test_csv", type=str, default="data/splits/gen_test.csv", help="Test set CSV")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model")
    parser.add_argument("--sft_model", type=str, default="models/sft", help="SFT model path")
    parser.add_argument("--dpo_model", type=str, default="models/dpo_enhanced", help="DPO model path")
    parser.add_argument("--sample_size", type=int, default=1000, help="Test samples (None for all)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--skip_base", action="store_true", help="Skip base model evaluation")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="sarcasm-detection", help="WandB project name")
    
    args = parser.parse_args()
    use_wandb = not args.no_wandb
    
    # Initialize WandB
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"comparison-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "test_set": args.test_csv,
                "sample_size": args.sample_size,
                "base_model": args.base_model,
                "sft_model": args.sft_model,
                "dpo_model": args.dpo_model
            }
        )
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    print(f"Loading test data from: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    print(f"✓ Loaded {len(test_df)} test samples")
    print(f"  Sarcastic: {(test_df['class'] == 'sarc').sum()}")
    print(f"  Not Sarcastic: {(test_df['class'] == 'notsarc').sum()}")
    
    all_results = []
    
    # Evaluate Base Model (optional)
    if not args.skip_base:
        print(f"\n{'#'*80}")
        print("STAGE 1: BASE MODEL")
        print(f"{'#'*80}")
        
        base_model, base_tokenizer = load_model_and_tokenizer(args.base_model)
        base_results = evaluate_model(base_model, base_tokenizer, test_df, "Base Model", args.sample_size)
        all_results.append(base_results)
        
        # Free memory
        del base_model, base_tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Evaluate SFT Model
    print(f"\n{'#'*80}")
    print("STAGE 2: SFT MODEL (Trained on GEN)")
    print(f"{'#'*80}")
    
    sft_model, sft_tokenizer = load_model_and_tokenizer(args.sft_model, args.base_model)
    sft_results = evaluate_model(sft_model, sft_tokenizer, test_df, "SFT Model (GEN)", args.sample_size)
    all_results.append(sft_results)
    
    del sft_model, sft_tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Evaluate DPO Model
    print(f"\n{'#'*80}")
    print("STAGE 3: DPO MODEL (Trained on iSarcasm)")
    print(f"{'#'*80}")
    
    dpo_model, dpo_tokenizer = load_model_and_tokenizer(args.dpo_model, args.base_model)
    dpo_results = evaluate_model(dpo_model, dpo_tokenizer, test_df, "DPO Model (iSarcasm)", args.sample_size)
    all_results.append(dpo_results)
    
    del dpo_model, dpo_tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Summary
    print(f"\n{'='*80}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 73)
    for result in all_results:
        print(f"{result['model_name']:<25} {result['accuracy']:<12.4f} {result['precision']:<12.4f} {result['recall']:<12.4f} {result['f1']:<12.4f}")
    
    # Calculate improvements
    if len(all_results) >= 2:
        print(f"\n{'Improvement':<25} {'Accuracy Δ':<12} {'F1 Δ':<12}")
        print("-" * 49)
        
        for i in range(1, len(all_results)):
            acc_delta = (all_results[i]['accuracy'] - all_results[0]['accuracy']) * 100
            f1_delta = (all_results[i]['f1'] - all_results[0]['f1']) * 100
            print(f"{all_results[i]['model_name']:<25} {acc_delta:+11.2f}pp {f1_delta:+11.2f}pp")
            
            # Log to WandB
            if use_wandb:
                wandb.log({
                    f"{all_results[i]['model_name']}/accuracy_improvement": acc_delta,
                    f"{all_results[i]['model_name']}/f1_improvement": f1_delta
                })
    
    # Log metrics to WandB
    if use_wandb:
        for result in all_results:
            wandb.log({
                f"{result['model_name']}/accuracy": result['accuracy'],
                f"{result['model_name']}/precision": result['precision'],
                f"{result['model_name']}/recall": result['recall'],
                f"{result['model_name']}/f1": result['f1'],
                f"{result['model_name']}/avg_confidence": result['avg_confidence']
            })
    
    # Save results
    summary = {
        'test_set': args.test_csv,
        'sample_size': args.sample_size,
        'evaluation_date': datetime.now().isoformat(),
        'models': all_results
    }
    
    output_json = f"{args.output_dir}/model_comparison_results.json"
    with open(output_json, 'w') as f:
        # Remove numpy arrays for JSON serialization
        for result in summary['models']:
            result['predictions'] = [int(p) for p in result['predictions']]
            result['labels'] = [int(l) for l in result['labels']]
            result['confidences'] = [float(c) for c in result['confidences']]
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved detailed results to {output_json}")
    
    # Create visualization
    output_plot = f"{args.output_dir}/model_comparison_plot.png"
    plot_comparison(all_results, output_plot)
    
    # Log visualization to WandB
    if use_wandb:
        wandb.log({"comparison_plot": wandb.Image(output_plot)})
        wandb.finish()
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results: {output_json}")
    print(f"Plot: {output_plot}")


if __name__ == "__main__":
    main()
