"""
Data preparation utilities for the two-phase training pipeline:
Phase 1 (SFT): SARC dataset
Phase 2 (DPO): iSarcasm dataset
"""

import pandas as pd
import os

def check_datasets():
    """Check availability and statistics of datasets."""
    print("="*70)
    print("DATASET OVERVIEW")
    print("="*70)
    
    # Check SARC
    sarc_path = "data/SARC/train-balanced-sarcasm.csv"
    if os.path.exists(sarc_path):
        # Get file size
        size_mb = os.path.getsize(sarc_path) / (1024 * 1024)
        print(f"\n✓ SARC Dataset Found: {sarc_path}")
        print(f"  Size: {size_mb:.1f} MB")
        
        # Quick stats (reading first chunk to avoid memory issues)
        try:
            df_sample = pd.read_csv(sarc_path, nrows=1000)
            print(f"  Columns: {list(df_sample.columns)}")
            print(f"  Sample sarcasm rate: {df_sample['label'].mean():.2%}")
        except Exception as e:
            print(f"  Note: Large file, use sampling during training")
    else:
        print(f"\n✗ SARC Dataset Not Found: {sarc_path}")
        print("  Please ensure SARC data is in data/SARC/ folder")
    
    # Check iSarcasm
    isarcasm_path = "data/isarcasm2022.csv"
    if os.path.exists(isarcasm_path):
        df = pd.read_csv(isarcasm_path, index_col=0)
        print(f"\n✓ iSarcasm Dataset Found: {isarcasm_path}")
        print(f"  Total samples: {len(df)}")
        print(f"  Sarcastic: {df['sarcastic'].sum()} ({df['sarcastic'].mean():.1%})")
        print(f"  Non-sarcastic: {len(df) - df['sarcastic'].sum()}")
        print(f"  Features: {list(df.columns)}")
        
        # Sarcasm type breakdown
        if 'irony' in df.columns:
            print("\n  Sarcasm Type Distribution:")
            print(f"    Irony: {df['irony'].sum()}")
            print(f"    Satire: {df['satire'].sum()}")
            print(f"    Overstatement: {df['overstatement'].sum()}")
            print(f"    Understatement: {df['understatement'].sum()}")
            print(f"    Rhetorical Question: {df['rhetorical_question'].sum()}")
    else:
        print(f"\n✗ iSarcasm Dataset Not Found: {isarcasm_path}")
    
    print("\n" + "="*70)
    print("TRAINING STRATEGY")
    print("="*70)
    print("Phase 1 (SFT):")
    print("  → Train on SARC (large volume)")
    print("  → Learn general sarcasm patterns")
    print("  → Output: ./qwen_sarc_sft/")
    print("\nPhase 2 (DPO):")
    print("  → Start from Phase 1 checkpoint")
    print("  → Refine with iSarcasm preferences")
    print("  → Utilize rich annotations (irony, satire, etc.)")
    print("  → Output: ./qwen_sarcasm_dpo/")
    print("="*70)

def create_train_val_split(csv_path, output_dir="data/splits", train_ratio=0.8):
    """Create train/validation splits for evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path, index_col=0)
    
    # Stratified split to maintain class balance
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, 
        train_size=train_ratio, 
        random_state=42,
        stratify=df['sarcastic']
    )
    
    train_path = os.path.join(output_dir, 'isarcasm_train.csv')
    val_path = os.path.join(output_dir, 'isarcasm_val.csv')
    
    train_df.to_csv(train_path)
    val_df.to_csv(val_path)
    
    print(f"Created splits:")
    print(f"  Train: {train_path} ({len(train_df)} samples)")
    print(f"  Val: {val_path} ({len(val_df)} samples)")
    
    return train_path, val_path

def sample_sarc_dataset(input_path, output_path, sample_size=50000):
    """Create a smaller sample of SARC for faster experimentation."""
    print(f"Sampling {sample_size} examples from SARC...")
    
    # Read in chunks and sample
    df = pd.read_csv(input_path, nrows=sample_size * 3)  # Read more to ensure we have enough
    
    # Balance classes
    sarc = df[df['label'] == 1].sample(n=min(sample_size//2, len(df[df['label'] == 1])), random_state=42)
    non_sarc = df[df['label'] == 0].sample(n=min(sample_size//2, len(df[df['label'] == 0])), random_state=42)
    
    sampled = pd.concat([sarc, non_sarc]).sample(frac=1, random_state=42)  # Shuffle
    
    sampled.to_csv(output_path, index=False)
    print(f"Saved {len(sampled)} samples to {output_path}")
    print(f"  Sarcastic: {sampled['label'].sum()}")
    print(f"  Non-sarcastic: {len(sampled) - sampled['label'].sum()}")
    
    return output_path

if __name__ == "__main__":
    # Check what datasets are available
    check_datasets()
    
    # Optionally create train/val splits for iSarcasm
    print("\n" + "="*70)
    print("Creating iSarcasm train/val splits...")
    print("="*70)
    try:
        create_train_val_split("data/isarcasm2022.csv")
    except Exception as e:
        print(f"Could not create splits: {e}")
