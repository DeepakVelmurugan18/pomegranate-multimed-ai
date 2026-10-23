"""
Extract and Normalize All Pomegranate Multimodal Datasets
Reads parquet and archive files from HuggingFace cache and organizes leaf & fruit images
into clean target class subdirectories under data/raw/leaf/ and data/raw/fruit/.
"""

import os
import io
import shutil
import pandas as pd
from PIL import Image
from pathlib import Path

# Explicit class label mapping dictionaries
VEDIC_SEEKER_MAP = {
    0: "Bacterial Blight",
    1: "Alternaria",
    2: "Anthracnose",
    3: "Cercospora"
}

AGML_MAP = {
    0: "Anthracnose",
    1: "Bacterial Blight",
    2: "Healthy",
    3: "Cercospora"
}

def clean_raw_dirs():
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    (raw_dir / "leaf").mkdir(parents=True, exist_ok=True)
    (raw_dir / "fruit").mkdir(parents=True, exist_ok=True)

def extract_all():
    clean_raw_dirs()
    
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    raw_dir = Path("data/raw")
    leaf_dir = raw_dir / "leaf"
    fruit_dir = raw_dir / "fruit"
    
    print("="*60)
    print("Extracting & Normalizing All Pomegranate Dataset Repositories")
    print("="*60)
    
    leaf_count = 0
    fruit_count = 0
    
    # 1. Process Parquet files
    parquet_files = list(cache_dir.rglob("*.parquet"))
    print(f"Processing {len(parquet_files)} parquet dataset files...")
    
    for p_file in parquet_files:
        repo_name = p_file.parent.parent.name.lower()
        print(f" -> Processing {p_file.name} from repo: {repo_name}...")
        
        try:
            df = pd.read_parquet(p_file)
            
            for idx, row in df.iterrows():
                try:
                    img_data = row['image']
                    if isinstance(img_data, dict) and 'bytes' in img_data:
                        img_obj = Image.open(io.BytesIO(img_data['bytes']))
                    elif isinstance(img_data, bytes):
                        img_obj = Image.open(io.BytesIO(img_data))
                    else:
                        continue
                        
                    raw_label = row['label']
                    
                    if 'vedicseeker' in repo_name:
                        class_name = VEDIC_SEEKER_MAP.get(int(raw_label), "Healthy")
                    elif 'project-agml' in repo_name:
                        class_name = AGML_MAP.get(int(raw_label), "Healthy")
                    elif 'techmahadev' in repo_name:
                        class_name = "Healthy"
                    else:
                        class_name = str(raw_label).replace('_', ' ').title()
                        
                    target_dir = leaf_dir / class_name
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    img_filename = target_dir / f"{p_file.stem}_{idx:05d}.jpg"
                    img_obj.convert("RGB").save(img_filename, "JPEG", quality=95)
                    leaf_count += 1
                    
                except Exception as ex_idx:
                    continue
                    
        except Exception as ex_p:
            print(f" -> Error reading {p_file.name}: {ex_p}")
            
    # 2. Process RAR archive for Fruit images (Ibrahim125 dataset)
    rar_files = list(cache_dir.rglob("*.rar"))
    if rar_files:
        import patoolib
        temp_extract = Path("data/raw/temp_rar_extract")
        temp_extract.mkdir(parents=True, exist_ok=True)
        for rf in rar_files:
            try:
                patoolib.extract_archive(str(rf), outdir=str(temp_extract))
            except Exception as e:
                print(f"Rar extraction note: {e}")
                
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        for img_file in temp_extract.rglob('*'):
            if img_file.is_file() and img_file.suffix.lower() in image_exts:
                class_dir = img_file.parent.name.replace('_', ' ').replace('-', ' ').title()
                dest_folder = fruit_dir / class_dir
                dest_folder.mkdir(parents=True, exist_ok=True)
                dest_path = dest_folder / img_file.name
                if not dest_path.exists():
                    shutil.copy2(img_file, dest_path)
                    fruit_count += 1
                    
        if temp_extract.exists():
            shutil.rmtree(temp_extract, ignore_errors=True)
            
    print("="*60)
    print(f"Dataset Extraction Complete!")
    print(f"  - Total Raw Leaf Images: {leaf_count}")
    print(f"  - Total Raw Fruit Images: {fruit_count}")
    print("="*60)

if __name__ == "__main__":
    extract_all()
