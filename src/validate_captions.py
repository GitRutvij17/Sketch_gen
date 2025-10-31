"""
STEP 2 (Optional): Validate Captions
Quick quality check before training

Save this file in: SKETCHGEN/src/
Run: python validate_captions.py
Time: 5 minutes
"""

import pandas as pd
import os

# ===== CONFIGURATION =====
CSV_FILE = "../data/captions/final_captions.csv"
# =========================

def main():
    print("="*80)
    print("CAPTION QUALITY VALIDATION")
    print("="*80)
    
    # Check if file exists
    if not os.path.exists(CSV_FILE):
        print(f"\n ERROR: Caption file not found: {CSV_FILE}")
        print("\nRun process_all_captions.py first!")
        return
    
    # Load captions
    print(f"\n Loading captions from: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    print(f"    Loaded {len(df)} captions")
    
    # Detect column name
    if 'cleaned_caption' in df.columns:
        caption_col = 'cleaned_caption'
    elif 'caption' in df.columns:
        caption_col = 'caption'
    else:
        print("\n ERROR: No valid caption column found in CSV!")
        print(" Expected one of: 'caption' or 'cleaned_caption'")
        print(f" Columns found: {list(df.columns)}")
        return

    # Sample 15 random captions
    print("\n Sampling 15 random captions for review...")
    samples = df.sample(n=min(15, len(df)), random_state=42)
    
    print("\n" + "="*80)
    print("CAPTION SAMPLES")
    print("="*80)
    
    for i, (idx, row) in enumerate(samples.iterrows(), 1):
        word_count = len(str(row[caption_col]).split())
        print(f"\n{i}. Image: {row['image_id']}")
        print(f"   Caption: {row[caption_col]}")
        print(f"   [{word_count} words]")
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    word_counts = df[caption_col].astype(str).str.split().str.len()
    
    print(f"\nTotal captions: {len(df)}")
    print(f"\nCaption length (characters):")
    print(f"  Avg: {df[caption_col].str.len().mean():.1f}")
    print(f"  Min: {df[caption_col].str.len().min()}")
    print(f"  Max: {df[caption_col].str.len().max()}")
    print(f"\nWord count:")
    print(f"  Avg: {word_counts.mean():.1f}")
    print(f"  Min: {word_counts.min()}")
    print(f"  Max: {word_counts.max()}")
    print(f"  Ideal (10–30 words): {((word_counts >= 10) & (word_counts <= 30)).sum()} "
          f"({((word_counts >= 10) & (word_counts <= 30)).sum()/len(df)*100:.1f}%)")
    
    # Check duplicates
    duplicates = df[caption_col].duplicated().sum()
    print(f"\nDuplicate captions: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    if duplicates < 10:
        print("   Good - very few duplicates!")
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("\n Captions look ready for training!")
    print("\nYou can now:")
    print("  1. Start LoRA fine-tuning with Stable Diffusion")
    print("  2. Data is in: data/processed/train/")
    print("="*80)


if __name__ == '__main__':
    main()
