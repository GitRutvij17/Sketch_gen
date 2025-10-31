"""
STEP 1: Clean CelebA Captions & Prepare Training Data
This is the ONLY file you need to run!

Save this file in: SKETCHGEN/src/
Run: python clean_and_prepare_captions.py
Time: 10-15 minutes
"""

import os
import re
from pathlib import Path
import shutil
from tqdm import tqdm
import pandas as pd

# ===== CONFIGURATION =====
TEXT_DIR = "../data/text"  # Your caption folder
IMAGE_DIR = "../data/images"
OUTPUT_TRAIN_DIR = "../data/processed/train"
OUTPUT_CSV = "../data/captions/final_captions.csv"
MAX_CAPTION_WORDS = 30  # Max words per caption
# =========================

def clean_caption(caption):
    """Clean and shorten verbose CelebA caption"""
    
    # Remove redundant phrases
    caption = re.sub(r'This (person|woman|man|individual|girl|boy) (is|has)', '', caption, flags=re.IGNORECASE)
    caption = re.sub(r'The (person|woman|man|individual|girl|boy) (is|has)', '', caption, flags=re.IGNORECASE)
    caption = re.sub(r'She (is|has|wears)', '', caption, flags=re.IGNORECASE)
    caption = re.sub(r'He (is|has|wears)', '', caption, flags=re.IGNORECASE)
    
    # Combine multiple sentences into one
    caption = caption.replace('. ', ', ')
    caption = caption.replace('..', '.')
    
    # Remove multiple spaces
    caption = re.sub(r'\s+', ' ', caption)
    
    # Remove multiple commas
    caption = re.sub(r',\s*,+', ',', caption)
    
    # Remove leading/trailing punctuation
    caption = caption.strip(' .,')
    
    # Limit to MAX_CAPTION_WORDS words
    words = caption.split()
    if len(words) > MAX_CAPTION_WORDS:
        caption = ' '.join(words[:MAX_CAPTION_WORDS])
    
    # Ensure starts with capital
    if caption:
        caption = caption[0].upper() + caption[1:]
    
    # Clean up and add period
    caption = caption.rstrip(',').strip() + '.'
    
    return caption


def main():
    print("="*80)
    print("CLEANING CELEBA CAPTIONS & PREPARING TRAINING DATA")
    print("="*80)
    
    # Check directories exist
    if not Path(TEXT_DIR).exists():
        print(f"\n ERROR: Caption directory not found: {TEXT_DIR}")
        print("\nCheck your folder structure:")
        print("  data/text/celeba-caption/ should contain .txt files")
        return
    
    if not Path(IMAGE_DIR).exists():
        print(f"\n ERROR: Image directory not found: {IMAGE_DIR}")
        return
    
    # Get all caption files
    caption_files = list(Path(TEXT_DIR).glob("*.txt"))
    
    print(f"\n Found {len(caption_files)} caption files")
    
    if len(caption_files) == 0:
        print("\n No .txt files found in caption directory!")
        return
    
    results = []
    matched_count = 0
    skipped_count = 0
    
    print(f"\n Processing captions...")
    
    for caption_file in tqdm(caption_files, desc="Processing"):
        # Get image name
        image_name = caption_file.stem
        
        # Find image file
        image_found = False
        image_path = None
        
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            potential_path = Path(IMAGE_DIR) / f"{image_name}{ext}"
            if potential_path.exists():
                image_found = True
                image_path = potential_path
                image_filename = f"{image_name}{ext}"
                break
        
        if not image_found:
            skipped_count += 1
            continue
        
        # Read caption
        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                original_caption = f.read().strip()
            
            # Skip if empty
            if len(original_caption) < 5:
                skipped_count += 1
                continue
            
            # Clean caption
            cleaned_caption = clean_caption(original_caption)
            
            results.append({
                'image_id': image_filename,
                'original_caption': original_caption,
                'cleaned_caption': cleaned_caption
            })
            
            matched_count += 1
        
        except Exception as e:
            skipped_count += 1
            continue
    
    print(f"\n Matched {matched_count} image-caption pairs")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} files (no matching image)")
    
    if matched_count == 0:
        print("\n ERROR: No image-caption pairs found!")
        print("Check that image filenames match caption filenames")
        return
    
    # Save to CSV
    df = pd.DataFrame(results)
    
    # Show samples
    print("\n" + "="*80)
    print("SAMPLE BEFORE & AFTER")
    print("="*80)
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"\n{i+1}. Image: {row['image_id']}")
        print(f"   ORIGINAL: {row['original_caption'][:70]}...")
        print(f"   CLEANED:  {row['cleaned_caption']}")
    
    # Statistics
    print("\n" + "="*80)
    print("CAPTION STATISTICS")
    print("="*80)
    print(f"Total captions: {len(df)}")
    print(f"\nOriginal captions:")
    print(f"  Avg length: {df['original_caption'].str.len().mean():.1f} chars")
    print(f"  Avg words: {df['original_caption'].str.split().str.len().mean():.1f} words")
    print(f"\nCleaned captions:")
    print(f"  Avg length: {df['cleaned_caption'].str.len().mean():.1f} chars")
    print(f"  Avg words: {df['cleaned_caption'].str.split().str.len().mean():.1f} words")
    
    # Save CSV
    print(f"\n Saving captions to CSV...")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"    Saved to: {OUTPUT_CSV}")
    
    # Prepare training data
    print(f"\n Preparing training data...")
    os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
    
    saved_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Copying images and captions"):
        image_id = row['image_id']
        caption = row['cleaned_caption']
        
        # Find source image
        image_found = False
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            potential_path = Path(IMAGE_DIR) / f"{row['image_id'].split(chr(46))[0]}{ext}"
            if potential_path.exists():
                src_image = potential_path
                image_found = True
                break
        
        if not image_found:
            src_image = Path(IMAGE_DIR) / image_id
        
        if src_image.exists():
            # Copy/link image
            dst_image = Path(OUTPUT_TRAIN_DIR) / image_id
            
            if not dst_image.exists():
                try:
                    os.symlink(src_image, dst_image)
                except:
                    shutil.copy2(src_image, dst_image)
            
            # Save caption
            txt_file = dst_image.with_suffix('.txt')
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            saved_count += 1
    
    print(f"\n Saved {saved_count} image-caption pairs to training directory!")
    print(f"   Location: {OUTPUT_TRAIN_DIR}")
    print(f"   Format: image.jpg + image.txt")
    
    print("\n" + "="*80)
    print(" COMPLETE!")
    print("="*80)
    print(f"\nYour training data is ready in:")
    print(f"  {OUTPUT_TRAIN_DIR}/")
    print(f"\nNext step: Run validate_captions.py (optional)")
    print(f"Then: Start LoRA training!")
    print("="*80)


if __name__ == '__main__':
    main()
