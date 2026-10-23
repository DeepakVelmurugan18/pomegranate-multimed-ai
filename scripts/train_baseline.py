"""
Phase 2: Single-Modality ResNet50 Baselines (E1-L and E1-F)
Trains ResNet50 baseline models for Leaf (E1-L) and Fruit (E1-F) modalities independently.
Uses real training data, ImageNet normalization, AdamW optimizer, Cosine Annealing scheduler,
early stopping based on Validation Macro-F1, and evaluates strictly once on frozen test sets.
"""

import os
import sys
import time
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import ResNet50_Weights

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from concurrent.futures import ThreadPoolExecutor

# Optimize CPU multithreading
num_cpus = os.cpu_count() or 4
torch.set_num_threads(num_cpus)

# Fixed reproducibility seed
SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# Define 5 Target Disease Classes & Index Mapping
CLASS_NAMES = ['Healthy', 'Alternaria', 'Anthracnose', 'Bacterial Blight', 'Cercospora']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASS_NAMES)}

class PomegranateDataset(Dataset):
    def __init__(self, df, transform=None, cache_in_memory=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.cache_in_memory = cache_in_memory
        self.cached_tensors = []
        self.labels = []
        
        for cls_name in self.df['mapped_class']:
            self.labels.append(CLASS_TO_IDX.get(cls_name, 0))
            
        if self.cache_in_memory:
            print(f" -> Pre-loading {len(self.df)} images into RAM ByteTensor cache (224x224 RGB)...", flush=True)
            to_tensor = T.PILToTensor()
            def load_single_tensor(path):
                try:
                    img = Image.open(path).convert('RGB').resize((224, 224))
                    return to_tensor(img)
                except Exception:
                    return torch.zeros((3, 224, 224), dtype=torch.uint8)

            filepaths = self.df['filepath'].tolist()
            with ThreadPoolExecutor(max_workers=16) as executor:
                self.cached_tensors = list(executor.map(load_single_tensor, filepaths))
            print(f" -> Successfully cached {len(self.cached_tensors)} image tensors in memory!", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.cache_in_memory:
            img_tensor = self.cached_tensors[idx]
            image = T.ToPILImage()(img_tensor)
        else:
            img_path = self.df.iloc[idx]['filepath']
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception:
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms():
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, eval_transform

def build_resnet50_model(num_classes=5, freeze_backbone=True):
    """ResNet50 feature extractor with pretrained ImageNet weights."""
    model = None
    cache_ckpt = Path(os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth"))
    
    for attempt in range(3):
        try:
            weights = ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            print(" -> Successfully loaded pretrained ResNet50 ImageNet weights.", flush=True)
            break
        except Exception as err:
            print(f" -> Retry [{attempt+1}/3] loading ResNet50 weights due to: {err}", flush=True)
            if cache_ckpt.exists():
                try:
                    cache_ckpt.unlink()
                except Exception:
                    pass
            time.sleep(2)
            
    if model is None:
        print(" -> [!] Download failed; initializing ResNet50 without pretrained weights fallback.", flush=True)
        model = models.resnet50(weights=None)
        
    if freeze_backbone:
        # Freeze backbone parameters for fast, stable CPU feature extraction
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
                
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    total_batches = len(dataloader)
    for batch_idx, (images, labels) in enumerate(dataloader, 1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 5 == 0 or batch_idx == total_batches:
            print(f"    Batch [{batch_idx:03d}/{total_batches:03d}] - Loss: {loss.item():.4f}", flush=True)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, acc, macro_f1

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    eval_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return eval_loss, acc, macro_f1, np.array(all_preds), np.array(all_labels)

def train_and_evaluate_baseline(modality_name, exp_id):
    print("="*75)
    print(f"Phase 2 Baseline Training: {exp_id} ({modality_name.upper()} Modality ResNet50)")
    print("="*75)
    
    output_dir = Path(f"outputs/baseline/{modality_name}_resnet50")
    metrics_file = output_dir / "metrics.json"
    if metrics_file.exists():
        print(f" -> [SKIP] Experiment {exp_id} ({modality_name.upper()}) is already completed!", flush=True)
        with open(metrics_file, "r") as f:
            existing_metrics = json.load(f)
        print(f" -> Loaded existing metrics (Test Macro-F1: {existing_metrics['test_performance']['macro_f1']})", flush=True)
        return existing_metrics
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" -> Execution Device: {device} (threads: {torch.get_num_threads()})", flush=True)
    
    processed_dir = Path("data/processed")
    df_train = pd.read_csv(processed_dir / "train.csv", low_memory=False)
    df_val = pd.read_csv(processed_dir / "validation.csv", low_memory=False)
    df_test = pd.read_csv(processed_dir / "test.csv", low_memory=False)
    
    df_train_mod = df_train[df_train['modality'] == modality_name].copy()
    df_val_mod = df_val[df_val['modality'] == modality_name].copy()
    df_test_mod = df_test[df_test['modality'] == modality_name].copy()
    
    print(f" -> {modality_name.upper()} Dataset Size: Train={len(df_train_mod)}, Val={len(df_val_mod)}, Test={len(df_test_mod)}", flush=True)
    
    train_tf, eval_tf = get_transforms()
    train_dataset = PomegranateDataset(df_train_mod, transform=train_tf)
    val_dataset = PomegranateDataset(df_val_mod, transform=eval_tf)
    test_dataset = PomegranateDataset(df_test_mod, transform=eval_tf)
    
    batch_size = 256
    num_workers = 0
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    model = build_resnet50_model(num_classes=5, freeze_backbone=True).to(device)
    total_p, trainable_p = count_parameters(model)
    print(f" -> Total Parameters: {total_p:,} ({total_p/1e6:.2f}M) | Trainable: {trainable_p:,} ({trainable_p/1e6:.2f}M)", flush=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    output_dir = Path(f"outputs/baseline/{modality_name}_resnet50")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_macro_f1 = -1.0
    patience = 6
    patience_counter = 0
    best_model_path = output_dir / "best_model.pth"
    
    history = []
    start_train_time = time.time()
    
    print("\n -> Starting Training Loop (Model Selection Metric: Validation Macro-F1)...", flush=True)
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate_model(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_macro_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_macro_f1': val_f1,
            'lr': current_lr,
            'epoch_time_sec': round(epoch_time, 2)
        })
        
        print(f"Epoch [{epoch:02d}/{epochs}] - Train Loss: {train_loss:.4f} | Train Macro-F1: {train_f1:.4f} || Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f} | Time: {epoch_time:.1f}s", flush=True)
        
        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"    >>> Best Checkpoint Saved! (Val Macro-F1: {best_val_macro_f1:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f" -> Early stopping triggered at Epoch {epoch} (No Val Macro-F1 improvement for {patience} epochs).", flush=True)
                break

    total_train_time = time.time() - start_train_time
    print(f"\n -> Training Completed in {total_train_time/60:.2f} minutes.", flush=True)
    
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(output_dir / "training_history.csv", index=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df_hist['epoch'], df_hist['train_loss'], label='Train Loss', color='blue', linewidth=2)
    axes[0].plot(df_hist['epoch'], df_hist['val_loss'], label='Val Loss', color='orange', linewidth=2)
    axes[0].set_title(f"{exp_id} - Loss Curves ({modality_name.capitalize()})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    axes[1].plot(df_hist['epoch'], df_hist['train_macro_f1'], label='Train Macro-F1', color='green', linewidth=2)
    axes[1].plot(df_hist['epoch'], df_hist['val_macro_f1'], label='Val Macro-F1', color='red', linewidth=2)
    axes[1].set_title(f"{exp_id} - Macro-F1 Score Curves ({modality_name.capitalize()})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1 Score")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300)
    plt.close()
    
    print(f"\n -> Evaluating Frozen Best Checkpoint on Pure Untouched Test Set...", flush=True)
    best_model = build_resnet50_model(num_classes=5, freeze_backbone=False).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    
    test_start = time.time()
    test_loss, test_acc, test_macro_f1, test_preds, test_labels = evaluate_model(best_model, test_loader, criterion, device)
    test_latency_ms = ((time.time() - test_start) / len(df_test_mod)) * 1000.0
    
    test_precision = precision_score(test_labels, test_preds, average='macro', zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average='macro', zero_division=0)
    
    per_class_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)
    per_class_f1_dict = {CLASS_NAMES[i]: round(float(per_class_f1[i]), 4) for i in range(len(CLASS_NAMES))}
    
    print("="*60)
    print(f"FINAL TEST RESULTS: {exp_id} ({modality_name.upper()} ResNet50)")
    print("="*60)
    print(f"  - Test Accuracy:          {test_acc*100:.2f}%")
    print(f"  - Test Macro-F1:          {test_macro_f1:.4f}")
    print(f"  - Test Macro-Precision:   {test_precision:.4f}")
    print(f"  - Test Macro-Recall:      {test_recall:.4f}")
    print(f"  - Per-Class F1 Scores:    {per_class_f1_dict}")
    print(f"  - Inference Latency:      {test_latency_ms:.2f} ms/sample")
    print("="*60, flush=True)
    
    cls_report = classification_report(test_labels, test_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(cls_report, f, indent=4)
        
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="left")
    ax.set_yticklabels(CLASS_NAMES)
    
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black' if cm[i, j] < cm.max()/2 else 'white', fontweight='bold')
            
    plt.title(f"Confusion Matrix: {exp_id} ({modality_name.capitalize()} ResNet50 Baseline)", pad=20)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()
    
    metrics_summary = {
        "experiment_id": exp_id,
        "modality": modality_name,
        "model_architecture": "ResNet50",
        "dataset_size": {
            "train": len(df_train_mod),
            "validation": len(df_val_mod),
            "test": len(df_test_mod)
        },
        "training_time_minutes": round(total_train_time / 60, 2),
        "inference_latency_ms_per_sample": round(test_latency_ms, 2),
        "parameter_count_millions": round(total_p / 1e6, 2),
        "trainable_parameter_count_millions": round(trainable_p / 1e6, 4),
        "test_performance": {
            "accuracy": round(float(test_acc), 4),
            "macro_precision": round(float(test_precision), 4),
            "macro_recall": round(float(test_recall), 4),
            "macro_f1": round(float(test_macro_f1), 4),
            "per_class_f1": per_class_f1_dict
        },
        "best_val_macro_f1": round(float(best_val_macro_f1), 4),
        "device_used": str(device)
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print(f" -> Artifacts saved cleanly at: {output_dir}/", flush=True)
    return metrics_summary

if __name__ == "__main__":
    print("="*75)
    print("Phase 2 Execution: Single-Modality ResNet50 Baselines")
    print("="*75)
    
    # Run E1-F (Fruit ResNet50 Baseline)
    fruit_metrics = train_and_evaluate_baseline('fruit', 'E1-F')
    
    # Run E1-L (Leaf ResNet50 Baseline)
    leaf_metrics = train_and_evaluate_baseline('leaf', 'E1-L')
    
    print("\n" + "="*75)
    print("Phase 2 Baselines Execution Successfully Completed!")
    print("="*75)
