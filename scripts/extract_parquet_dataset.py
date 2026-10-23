"""
Parquet Dataset Extraction Script
Parses HuggingFace Parquet datasets containing pomegranate disease images and extracts
them into data/raw/leaf/ and data/raw/fruit/ with appropriate class labels.
"""

import os
import glob
import io
from pathlib import Path
import pandas as pd
from PIL import Image

def extract_parquet_files():
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    raw_dir = Path("data/raw")
    leaf_dir = raw_dir / "leaf"
    fruit_dir = raw_dir / "fruit"
    
    leaf_dir.mkdir(parents=True, exist_ok=True)
    fruit_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Parquet Image Extraction Engine")
    print("="*60)
    
    parquet_files = list(cache_dir.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet dataset files.")
    
    extracted_leaf_count = 0
    extracted_fruit_count = 0
    
    for p_file in parquet_files:
        print(f"Reading parquet: {p_file.name} from {p_file.parent.parent.name}...")
        try:
            df = pd.read_parquet(p_file)
            print(f" -> Columns: {list(df.columns)}, Total rows: {len(df)}")
            
            # Identify image column and label column
            img_col = None
            label_col = None
            
            for col in df.columns:
                if col in ['image', 'img', 'file', 'data']:
                    img_col = col
                elif col in ['label', 'labels', 'class', 'target', 'disease']:
                    label_col = col
                    
            if not img_col:
                img_col = df.columns[0]
            if not label_col and len(df.columns) > 1:
                label_col = df.columns[1]
                
            repo_name = p_file.parent.parent.name.lower()
            
            for idx, row in df.iterrows():
                try:
                    # Get image data
                    img_data = row[img_col]
                    img_obj = None
                    
                    if isinstance(img_data, dict):
                        if 'bytes' in img_data and img_data['bytes'] is not None:
                            img_obj = Image.open(io.BytesIO(img_data['bytes']))
                        elif 'path' in img_data and img_data['path']:
                            img_obj = Image.open(img_data['path'])
                    elif isinstance(img_data, bytes):
                        img_obj = Image.open(io.BytesIO(img_data))
                    elif isinstance(img_data, str) and os.path.exists(img_data):
                        img_obj = Image.open(img_data)
                        
                    if img_obj is None:
                        continue
                        
                    # Get label name
                    raw_label = str(row[label_col]) if label_col in row else "Unknown"
                    
                    # Convert integer labels if metadata exists or format string
                    label_clean = raw_label.replace('_', ' ').replace('-', ' ').title()
                    
                    # Determine modality (fruit vs leaf)
                    is_fruit = 'fruit' in repo_name or 'fruit' in label_clean.lower()
                    
                    if is_fruit:
                        dest_folder = fruit_dir / label_clean
                        extracted_fruit_count += 1
                    else:
                        dest_folder = leaf_dir / label_clean
                        extracted_leaf_count += 1
                        
                    dest_folder.mkdir(parents=True, exist_ok=True)
                    dest_filename = dest_folder / f"parquet_{p_file.stem}_{idx:04d}.jpg"
                    
                    img_obj.convert("RGB").save(dest_filename, "JPEG", quality=95)
                    
                except Exception as ex_row:
                    continue
                    
        except Exception as ex_file:
            print(f" -> Error reading {p_file.name}: {ex_file}")
            
    print(f"Extraction completed: {extracted_leaf_count} Leaf images, {extracted_fruit_count} Fruit images extracted.")

if __name__ == "__main__":
    extract_parquet_files()
