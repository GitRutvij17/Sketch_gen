#!/usr/bin/env python3
"""
PROCESS ALL CAPTIONS - Universal Caption Processor
Handles: subfolders, encoding, naming mismatches, etc.

Save as: SKETCHGEN/src/process_all_captions.py
Run: python process_all_captions.py
"""

import os
import shutil
from pathlib import Path
import re
import sys

try:
    from tqdm import tqdm
    HAS_TQDM = True
except:
    HAS_TQDM = False
    print("Note: tqdm not installed, progress bar disabled")

try:
    import pandas as pd
    HAS_PANDAS = True
except:
    HAS_PANDAS = False
    print("Note: pandas not installed, CSV export disabled")


def clean_caption(text):
    """Clean caption text"""
    if not text:
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text.strip())
    if len(text) > 300:
        text = text[:300]
    return text.strip()


def find_caption_files(start_path):
    """Recursively find all .txt files"""
    path = Path(start_path)
    if not path.exists():
        return []
    return list(path.rglob("*.txt"))


def find_image_files(img_dir):
    """Find all image files"""
    path = Path(img_dir)
    if not path.exists():
        return []
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG', '*.JPEG']:
        images.extend(list(path.glob(ext)))
    return images


def match_images_captions(captions, images):
    """Match caption files with image files by stem"""
    matches = []
    
    for cap_file in captions:
        cap_stem = cap_file.stem
        
        for img_file in images:
            if img_file.stem == cap_stem:
                matches.append((img_file, cap_file))
                break
    
    return matches


def read_caption_file(filepath):
    """Try to read caption file with multiple encodings"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return f.read()
        except:
            continue
    
    return None


def main():
    print("=" * 80)
    print("UNIVERSAL CAPTION PROCESSOR")
    print("=" * 80)
    
    # Step 1: Get caption directory
    print("\nSTEP 1: Caption Directory")
    print("-" * 80)
    print("Where are your caption files?")
    print("Examples: ../data/text/celeba-caption  or  ../data/text  or  ../data")
    print()
    
    caption_path = input("Enter path (or press Enter for auto-search): ").strip()
    
    if not caption_path:
        print("\nAuto-searching...")
        for p in [Path("../data/text/celeba-caption"), 
                  Path("../data/text"), 
                  Path("../data")]:
            if p.exists():
                caption_path = str(p)
                print(f"Found: {caption_path}")
                break
    
    if not caption_path or not Path(caption_path).exists():
        print("Path doesn't exist!")
        return False
    
    # Step 2: Find caption files
    print("\nSTEP 2: Finding Caption Files")
    print("-" * 80)
    caption_files = find_caption_files(caption_path)
    print(f"Found {len(caption_files)} .txt files")
    
    if len(caption_files) == 0:
        print("No .txt files found!")
        return False
    
    print(f"   Examples: {[f.name for f in caption_files[:3]]}")
    
    # Step 3: Find image directory
    print("\nSTEP 3: Image Directory")
    print("-" * 80)
    img_path = "../data/images"
    if not Path(img_path).exists():
        print(f"Default path not found: {img_path}")
        img_path = input("Enter image directory path: ").strip()
    
    image_files = find_image_files(img_path)
    print(f"Found {len(image_files)} image files")
    print(f"   Examples: {[f.name for f in image_files[:3]]}")
    
    if len(image_files) == 0:
        print("No images found!")
        return False
    
    # Step 4: Match files
    print("\nSTEP 4: Matching Captions with Images")
    print("-" * 80)
    matches = match_images_captions(caption_files, image_files)
    print(f"Matched {len(matches)} pairs")
    
    if len(matches) == 0:
        print("No matches found!")
        print("Image and caption filenames may not match")
        return False
    
    # Step 5: Process and copy
    print("\nSTEP 5: Processing")
    print("-" * 80)
    
    output_dir = Path("../data/processed/train")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    success = 0
    failed = 0
    
    iterator = tqdm(matches, desc="Processing") if HAS_TQDM else matches
    
    for img_file, cap_file in iterator:
        try:
            # Read caption
            caption_text = read_caption_file(cap_file)
            if not caption_text:
                failed += 1
                continue
            
            # Clean caption
            caption_text = clean_caption(caption_text)
            
            # Copy image
            dst_img = output_dir / img_file.name
            if not dst_img.exists():
                try:
                    os.symlink(img_file, dst_img)
                except:
                    shutil.copy2(img_file, dst_img)
            
            # Write caption
            txt_out = dst_img.with_suffix('.txt')
            with open(txt_out, 'w', encoding='utf-8') as f:
                f.write(caption_text)
            
            results.append({
                'image_id': img_file.name,
                'caption': caption_text
            })
            
            success += 1
        
        except Exception as e:
            failed += 1
    
    # Step 6: Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Successfully processed: {success}")
    print(f"Failed: {failed}")
    print(f"Total: {len(results)} image-caption pairs")
    
    if len(results) > 0:
        print(f"\nOutput directory: {output_dir.absolute()}")
        print(f"   - {success} image files (.jpg, .png)")
        print(f"   - {success} caption files (.txt)")
        
        # Save CSV
        if HAS_PANDAS:
            csv_file = Path("../data/captions/final_captions.csv")
            csv_file.parent.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame(results)
            df.to_csv(csv_file, index=False)
            print(f"\nCSV saved: {csv_file.absolute()}")
        
        # Show samples
        print(f"\nSample captions:")
        for i, result in enumerate(results[:3], 1):
            print(f"   {i}. {result['image_id']}")
            print(f"      {result['caption'][:70]}...")
        
        print("\n" + "=" * 80)
        print("SUCCESS! Training data ready!")
        print("=" * 80)
        print("\nNext step: Run validate_captions.py")
        return True
    else:
        print("\nNo successful matches")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
