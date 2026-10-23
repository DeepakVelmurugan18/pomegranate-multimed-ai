"""
Extract RAR dataset files for Pomegranate Fruit Diseases into data/raw/fruit/
"""

import os
import shutil
from pathlib import Path
import patoolib

def extract_rar_datasets():
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    fruit_dir = Path("data/raw/fruit")
    fruit_dir.mkdir(parents=True, exist_ok=True)
    
    rar_files = list(cache_dir.rglob("*.rar"))
    print(f"Found {len(rar_files)} RAR dataset files.")
    
    temp_extract = Path("data/raw/temp_rar_extract")
    temp_extract.mkdir(parents=True, exist_ok=True)
    
    for rf in rar_files:
        print(f"Extracting RAR archive: {rf.name}...")
        try:
            patoolib.extract_archive(str(rf), outdir=str(temp_extract))
        except Exception as e:
            print(f"Error extracting {rf.name}: {e}")
            
    # Organize extracted image files
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    moved_count = 0
    
    for img_file in temp_extract.rglob('*'):
        if img_file.is_file() and img_file.suffix.lower() in image_exts:
            class_dir = img_file.parent.name.replace('_', ' ').replace('-', ' ').title()
            dest_folder = fruit_dir / class_dir
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            dest_path = dest_folder / img_file.name
            if not dest_path.exists():
                shutil.copy2(img_file, dest_path)
                moved_count += 1
                
    # Clean up temp
    if temp_extract.exists():
        shutil.rmtree(temp_extract, ignore_errors=True)
        
    print(f"RAR extraction finished. {moved_count} Fruit disease images copied into data/raw/fruit/.")

if __name__ == "__main__":
    extract_rar_datasets()
