"""
Dataset Acquisition Script for Pomegranate-MultiMed AI
Acquires pomegranate leaf and fruit disease datasets via HuggingFace Hub public repositories,
extracts compressed archives if present, and organizes images into data/raw/leaf and data/raw/fruit.
"""

import os
import shutil
import zipfile
import tarfile
from pathlib import Path
from huggingface_hub import snapshot_download

def extract_if_archive(file_path, dest_dir):
    p = Path(file_path)
    if p.suffix.lower() == '.zip':
        print(f" -> Extracting zip archive: {p.name}...")
        try:
            with zipfile.ZipFile(p, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            return True
        except Exception as e:
            print(f" -> Error extracting zip {p.name}: {e}")
    elif p.suffix.lower() in ['.tar', '.gz', '.tgz']:
        print(f" -> Extracting tar archive: {p.name}...")
        try:
            with tarfile.open(p, 'r:*') as tar_ref:
                tar_ref.extractall(dest_dir)
            return True
        except Exception as e:
            print(f" -> Error extracting tar {p.name}: {e}")
    return False

def acquire_pomegranate_datasets():
    raw_dir = Path("data/raw")
    leaf_dir = raw_dir / "leaf"
    fruit_dir = raw_dir / "fruit"
    
    leaf_dir.mkdir(parents=True, exist_ok=True)
    fruit_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Pomegranate Multimodal Dataset Ingestion Engine")
    print("="*60)
    
    hf_repos = [
        "Project-AgML/pomegranate_disease_classification",
        "Ibrahim125/Pomegranate-Fruit-Diseases-Dataset-for-Deep-Learning-Models",
        "TechMahadev/pomegranatePlantHealth",
        "AmitTheGreat/pomegranateFruitDiseas",
        "VedicSeeker/pomegranateDiseaseClassification"
    ]
    
    acquired_paths = []
    
    for repo_id in hf_repos:
        try:
            print(f"[+] Fetching dataset repository: {repo_id}...")
            download_path = snapshot_download(repo_id=repo_id, repo_type="dataset")
            print(f" -> Successfully fetched at: {download_path}")
            acquired_paths.append(download_path)
        except Exception as err:
            print(f" -> Repository {repo_id} failed: {err}")
            
    print("[2/3] Extracting archives and sorting into Leaf and Fruit modalities...")
    
    # Process each downloaded path
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    leaf_count = 0
    fruit_count = 0
    
    for repo_path in acquired_paths:
        p_path = Path(repo_path)
        
        # Check for zip/tar archives inside download directory and extract
        for archive_file in list(p_path.rglob('*.zip')) + list(p_path.rglob('*.tar')) + list(p_path.rglob('*.tar.gz')):
            extract_dest = p_path / f"extracted_{archive_file.stem}"
            extract_if_archive(archive_file, extract_dest)
            
        # Iterate over all images
        for img_file in p_path.rglob('*'):
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                rel_parts = [part.lower() for part in img_file.parts]
                
                # Identify class from directory structure
                class_dir = img_file.parent.name
                if class_dir.lower() in ['data', 'images', 'train', 'test', 'val', 'valid', 'extracted']:
                    class_dir = img_file.parent.parent.name
                    
                # Standardize class directory name
                class_dir_clean = class_dir.replace('-', ' ').replace('_', ' ').title()
                
                # Classify modality based on path keywords
                is_fruit = any(kw in rel_parts for kw in ['fruit', 'fruits', 'epicarp', 'pomegranatefruit'])
                is_leaf = any(kw in rel_parts for kw in ['leaf', 'leaves', 'foliar', 'pomegranateplant'])
                
                if is_fruit and not is_leaf:
                    target_root = fruit_dir / class_dir_clean
                    fruit_count += 1
                elif is_leaf and not is_fruit:
                    target_root = leaf_dir / class_dir_clean
                    leaf_count += 1
                else:
                    # Contextual default based on repository name
                    if 'fruit' in p_path.name.lower():
                        target_root = fruit_dir / class_dir_clean
                        fruit_count += 1
                    else:
                        target_root = leaf_dir / class_dir_clean
                        leaf_count += 1
                        
                target_root.mkdir(parents=True, exist_ok=True)
                dest_path = target_root / img_file.name
                if not dest_path.exists():
                    shutil.copy2(img_file, dest_path)
                    
    print(f"[3/3] Acquisition completed: {leaf_count} Leaf images, {fruit_count} Fruit images organized into data/raw/.")
    return leaf_count, fruit_count

if __name__ == "__main__":
    acquire_pomegranate_datasets()
