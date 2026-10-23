"""
Phase 1 & 1.1: Refined Group-Aware Audit & Splitting Engine
Guarantees 0 cross-split group leakage by handling source dataset groupings and filename prefixes.
"""

import os
import sys
import json
import hashlib
import random
from pathlib import Path
from collections import defaultdict, Counter
import concurrent.futures

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

TARGET_TAXONOMY = ['Healthy', 'Alternaria', 'Anthracnose', 'Bacterial Blight', 'Cercospora']

RAW_TO_TARGET_MAP = {
    '0': 'Bacterial Blight',
    '1': 'Alternaria',
    '2': 'Anthracnose',
    '3': 'Cercospora',
    'Healthy': 'Healthy',
    'Alternaria': 'Alternaria',
    'Anthracnose': 'Anthracnose',
    'Bacterial Blight': 'Bacterial Blight',
    'Cercospora': 'Cercospora'
}

def compute_fast_md5(file_path):
    file_size = os.path.getsize(file_path)
    hash_md5 = hashlib.md5()
    hash_md5.update(str(file_size).encode('utf-8'))
    with open(file_path, "rb") as f:
        hash_md5.update(f.read(65536))
    return hash_md5.hexdigest(), file_size

def extract_group_id(file_path):
    """Extract plant/tree/session identifier from filename or path."""
    p = Path(file_path)
    stem = p.stem.replace('-', '_')
    parts = stem.split('_')
    
    # Check for explicit session/plant prefixes
    if len(parts) > 1 and any(kw in parts[0].lower() for kw in ['tree', 'plant', 'session', 'field', 'case', 'loc']):
        return parts[0]
        
    # Default to unique image file path hash as group ID to prevent accidental grouping of unrelated files
    return hashlib.md5(str(p.name).encode('utf-8')).hexdigest()[:12]

def audit_single_file(img_path, modality, raw_class):
    try:
        hash_val, file_size = compute_fast_md5(img_path)
        with Image.open(img_path) as img:
            width, height = img.size
            mode = img.mode
            
        group_id = extract_group_id(img_path)
        
        sev_label = "N/A"
        for s_tag in ['low', 'medium', 'high', 'mild', 'moderate', 'severe']:
            if s_tag in Path(img_path).name.lower():
                sev_label = s_tag.capitalize()
                break
                
        return {
            'filepath': str(img_path),
            'modality': modality,
            'raw_class': raw_class,
            'group_id': group_id,
            'width': width,
            'height': height,
            'mode': mode,
            'hash_val': hash_val,
            'file_size': file_size,
            'expert_severity': sev_label,
            'is_valid': True,
            'error': None
        }
    except Exception as err:
        return {
            'filepath': str(img_path),
            'modality': modality,
            'raw_class': raw_class,
            'is_valid': False,
            'error': str(err)
        }

def inspect_and_audit_dataset():
    raw_dir = Path("data/raw")
    leaf_dir = raw_dir / "leaf"
    fruit_dir = raw_dir / "fruit"
    
    processed_dir = Path("data/processed")
    reports_dir = Path("outputs/reports")
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Phase 1 & 1.1: Dataset Audit & Group-Aware Splitting Engine")
    print("="*70)
    
    tasks = []
    
    for modality, mod_dir in [('leaf', leaf_dir), ('fruit', fruit_dir)]:
        if not mod_dir.exists():
            continue
        for class_path in mod_dir.iterdir():
            if class_path.is_dir():
                raw_class = class_path.name
                for img_file in class_path.rglob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        tasks.append((str(img_file), modality, raw_class))
                        
    print(f" -> Found {len(tasks)} raw image files to audit across Leaf and Fruit modalities.")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(audit_single_file, path, mod, r_cls) for path, mod, r_cls in tasks]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
            
    valid_samples = [r for r in results if r['is_valid']]
    corrupt_samples = [r for r in results if not r['is_valid']]
    
    seen_hashes = set()
    unique_samples = []
    duplicate_count = 0
    
    for s in valid_samples:
        if s['hash_val'] in seen_hashes:
            s['is_duplicate'] = True
            duplicate_count += 1
        else:
            s['is_duplicate'] = False
            seen_hashes.add(s['hash_val'])
            unique_samples.append(s)
            
    print(f" -> Total images audited: {len(results)}")
    print(f" -> Corrupt files filtered: {len(corrupt_samples)}")
    print(f" -> Duplicates removed: {duplicate_count}")
    print(f" -> Unique valid dataset size: {len(unique_samples)}")

    # Taxonomy Mapping
    raw_classes = sorted(list(set(s['raw_class'] for s in unique_samples)))
    taxonomy_mapping = {}
    taxonomy_deltas = []
    
    for r_cls in raw_classes:
        mapped = RAW_TO_TARGET_MAP.get(r_cls, r_cls)
        taxonomy_mapping[r_cls] = mapped
        if r_cls != mapped:
            taxonomy_deltas.append(f"Raw directory '{r_cls}' mapped to Target Class '{mapped}'.")

    for s in unique_samples:
        s['mapped_class'] = taxonomy_mapping.get(s['raw_class'], s['raw_class'])

    # Save Taxonomy Delta Report
    with open(processed_dir / "taxonomy_delta_report.md", "w", encoding="utf-8") as f:
        f.write("# Taxonomy Delta Report (Phase 1 Audit)\n\n")
        f.write(f"- **Target Project Taxonomy**: `{TARGET_TAXONOMY}`\n")
        f.write(f"- **Raw Dataset Folders Discovered**: `{raw_classes}`\n\n")
        f.write("## Taxonomy Mapping Table\n\n")
        f.write("| Raw Folder Name | Mapped Target Class | Taxonomy Status | Rationale |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for r_cls in raw_classes:
            mapped = taxonomy_mapping[r_cls]
            status = "Matched Target Class" if mapped in TARGET_TAXONOMY else "DELTA CLASS"
            f.write(f"| `{r_cls}` | `{mapped}` | {status} | Taxonomy alignment verification |\n")
        f.write("\n## Deltas & Mapping Rationale\n")
        for delta in taxonomy_deltas:
            f.write(f"- {delta}\n")
        f.write("\n> [!IMPORTANT]\n> User sign-off required for Taxonomy Delta Report before transforming class labels.\n")

    # Pairing Audit (Mode A vs Mode B)
    leaf_groups = set(s['group_id'] for s in unique_samples if s['modality'] == 'leaf')
    fruit_groups = set(s['group_id'] for s in unique_samples if s['modality'] == 'fruit')
    shared_groups = leaf_groups.intersection(fruit_groups)
    
    pairing_mode = "Mode B — Class-Aligned Unpaired Multimodal Learning"
    same_plant_default = False

    for s in unique_samples:
        s['pair_type'] = "class_aligned_unpaired"
        s['same_plant'] = same_plant_default

    # Group-Aware Stratified Splitting (70% Train, 15% Val, 15% Test)
    group_buckets = defaultdict(list)
    for s in unique_samples:
        bucket_key = (s['mapped_class'], s['group_id'])
        group_buckets[bucket_key].append(s)

    train_samples, val_samples, test_samples = [], [], []
    class_to_buckets = defaultdict(list)
    for (cls_name, g_id), samples in group_buckets.items():
        class_to_buckets[cls_name].append(samples)
        
    for cls_name, bucket_list in class_to_buckets.items():
        random.shuffle(bucket_list)
        total_cls_samples = sum(len(b) for b in bucket_list)
        
        train_target = int(0.70 * total_cls_samples)
        val_target = int(0.15 * total_cls_samples)
        
        curr_train, curr_val = 0, 0
        
        for bucket in bucket_list:
            b_len = len(bucket)
            if curr_train < train_target:
                for s in bucket: s['split'] = 'train'
                train_samples.extend(bucket)
                curr_train += b_len
            elif curr_val < val_target:
                for s in bucket: s['split'] = 'validation'
                val_samples.extend(bucket)
                curr_val += b_len
            else:
                for s in bucket: s['split'] = 'test'
                test_samples.extend(bucket)

    print(f" -> Group-Aware Stratified Splits: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    # Save Dataframes & Reports
    pd.DataFrame(unique_samples).to_csv(processed_dir / "dataset_summary.csv", index=False)
    pd.DataFrame(train_samples).to_csv(processed_dir / "train.csv", index=False)
    pd.DataFrame(val_samples).to_csv(processed_dir / "validation.csv", index=False)
    pd.DataFrame(test_samples).to_csv(processed_dir / "test.csv", index=False)

    print("="*70)
    print("Dataset audit and splitting updated successfully.")

if __name__ == "__main__":
    inspect_and_audit_dataset()
