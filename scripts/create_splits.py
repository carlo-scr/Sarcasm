"""
Create train/test splits for iSarcasm dataset.
Train split (80%) will be used for DPO training.
Test split (20%) will be held out for evaluation only.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import json

def create_isarcasm_splits():
    """Create stratified train/test splits for iSarcasm."""
    print("Creating iSarcasm train/test splits...")
    
    # Load full dataset
    df = pd.read_csv('data/isarcasm2022.csv', index_col=0)
    print(f"\nTotal samples: {len(df)}")
    print(f"Sarcastic: {df['sarcastic'].sum()} ({df['sarcastic'].mean():.1%})")
    print(f"Non-sarcastic: {len(df) - df['sarcastic'].sum()} ({1-df['sarcastic'].mean():.1%})")
    
    # Create stratified split
    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['sarcastic']
    )
    
    # Save splits
    df_train.to_csv('data/splits/isarcasm_train.csv')
    df_test.to_csv('data/splits/isarcasm_test.csv')
    
    print(f"\n✓ Train split: {len(df_train)} samples")
    print(f"  Sarcastic: {df_train['sarcastic'].sum()} ({df_train['sarcastic'].mean():.1%})")
    print(f"  Non-sarcastic: {len(df_train) - df_train['sarcastic'].sum()}")
    
    print(f"\n✓ Test split: {len(df_test)} samples")
    print(f"  Sarcastic: {df_test['sarcastic'].sum()} ({df_test['sarcastic'].mean():.1%})")
    print(f"  Non-sarcastic: {len(df_test) - df_test['sarcastic'].sum()}")
    
    print(f"\n✓ Saved to data/splits/")
    print("  - isarcasm_train.csv (for DPO training)")
    print("  - isarcasm_test.csv (for evaluation only)")

if __name__ == "__main__":
    create_isarcasm_splits()
