"""
Split GEN-sarc-notsarc dataset into train/test splits.
Train: 80% (used for SFT and DPO training)
Test: 20% (held-out for evaluation only)
"""

import pandas as pd
import os

def split_gen_dataset():
    """Split GEN dataset into train and test sets."""
    print("="*70)
    print("SPLITTING GEN-SARC-NOTSARC DATASET")
    print("="*70)
    
    # Load dataset
    input_path = "data/GEN-sarc-notsarc.csv"
    print(f"\nLoading dataset from: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ Dataset not found at {input_path}")
        return
    
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df)} samples")
    
    # Count classes
    sarc_count = (df['class'] == 'sarc').sum()
    notsarc_count = (df['class'] == 'notsarc').sum()
    
    print(f"  Sarcastic: {sarc_count} ({sarc_count/len(df):.1%})")
    print(f"  Non-sarcastic: {notsarc_count} ({notsarc_count/len(df):.1%})")
    
    # Split each class separately to maintain balance
    sarc_df = df[df['class'] == 'sarc'].reset_index(drop=True)
    notsarc_df = df[df['class'] == 'notsarc'].reset_index(drop=True)
    
    # 80/20 split for each class
    sarc_train = sarc_df.sample(frac=0.8, random_state=42)
    sarc_test = sarc_df.drop(sarc_train.index)
    
    notsarc_train = notsarc_df.sample(frac=0.8, random_state=42)
    notsarc_test = notsarc_df.drop(notsarc_train.index)
    
    # Combine and shuffle
    train_df = pd.concat([sarc_train, notsarc_train]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat([sarc_test, notsarc_test]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✓ Created train/test splits:")
    print(f"  Training set: {len(train_df)} samples")
    print(f"    Sarcastic: {(train_df['class'] == 'sarc').sum()} ({(train_df['class'] == 'sarc').mean():.1%})")
    print(f"    Non-sarcastic: {(train_df['class'] == 'notsarc').sum()} ({(train_df['class'] == 'notsarc').mean():.1%})")
    
    print(f"  Test set: {len(test_df)} samples")
    print(f"    Sarcastic: {(test_df['class'] == 'sarc').sum()} ({(test_df['class'] == 'sarc').mean():.1%})")
    print(f"    Non-sarcastic: {(test_df['class'] == 'notsarc').sum()} ({(test_df['class'] == 'notsarc').mean():.1%})")
    
    # Create output directory
    os.makedirs("data/splits", exist_ok=True)
    
    # Save splits
    train_path = "data/splits/gen_train.csv"
    test_path = "data/splits/gen_test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✓ Saved splits:")
    print(f"  Training: {train_path}")
    print(f"  Test: {test_path}")
    print("\n" + "="*70)

if __name__ == "__main__":
    split_gen_dataset()
