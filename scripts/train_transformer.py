"""
Phase 3: Single-Modality Swin Transformer Feature Extractors (E2-L and E3-F)
Trains Swin-B baseline models for Fruit (E3-F) and Leaf (E2-L) modalities independently.
Uses real training data, ImageNet normalization, AdamW optimizer, Cosine Annealing scheduler,
early stopping based on Validation Macro-F1, and evaluates strictly once on frozen test sets.
"""

import os
import sys
import time
import json
import random
import argparse
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
from torchvision.models import Swin_B_Weights

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

# Optimize CPU multithreading
num_cpus = os.cpu_count() or 4
torch.set_num_threads(num_cpus)
print(f"Using {torch.get_num_threads()} CPU threads.")

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

class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.filepaths = self.df['filepath'].tolist()
        self.labels = [CLASS_TO_IDX.get(c, 0) for c in self.df['mapped_class']]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def get_transforms():
    eval_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return eval_tf

def build_swin_b_model(num_classes=5, freeze_backbone=True):
    """
    Swin-B hierarchical vision transformer with pretrained ImageNet weights.
    Freeze backbone parameters (patch partition + 4 Swin stages + norm)
    Trainable parameters: linear head (nn.Linear(1024, 5))
    """
    print(" -> Loading pretrained Swin-B ImageNet weights...", flush=True)
    weights = Swin_B_Weights.DEFAULT
    model = models.swin_b(weights=weights)
    
    in_features = model.head.in_features  # 1024
    
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
                
    model.head = nn.Linear(in_features, num_classes)
    model.head.weight.requires_grad = True
    model.head.bias.requires_grad = True
    
    return model

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen

def extract_swin_embeddings(model, dataloader, device, cache_file=None):
    """
    Extracts 1024-dimensional feature embeddings through the frozen Swin-B backbone.
    Caches to disk for computational reproducibility and zero redundant CPU passes.
    """
    if cache_file and Path(cache_file).exists():
        print(f" -> Loading pre-extracted embeddings from cache: {cache_file}...", flush=True)
        cached_data = torch.load(cache_file, map_location='cpu')
        return cached_data['embeddings'], cached_data['labels']

    print(f" -> Extracting 1024-dim features using frozen Swin-B backbone ({len(dataloader.dataset)} images)...", flush=True)
    model.eval()
    
    # Define backbone feature extractor (everything up to head)
    def extract_features(x):
        x = model.features(x)
        x = model.norm(x)
        x = model.permute(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    all_embeddings = []
    all_labels = []
    total_batches = len(dataloader)
    start_t = time.time()
    
    with torch.no_grad():
        for b_idx, (images, labels) in enumerate(dataloader, 1):
            images = images.to(device)
            emb = extract_features(images)
            all_embeddings.append(emb.cpu())
            all_labels.append(labels)
            if b_idx % 10 == 0 or b_idx == total_batches:
                elapsed = time.time() - start_t
                img_done = b_idx * dataloader.batch_size
                rate = min(img_done, len(dataloader.dataset)) / max(elapsed, 0.001)
                print(f"    Feature Batch [{b_idx:03d}/{total_batches:03d}] - Processed {min(img_done, len(dataloader.dataset))}/{len(dataloader.dataset)} ({rate:.1f} img/s)", flush=True)
                
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    if cache_file:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'embeddings': embeddings, 'labels': labels}, cache_file)
        print(f" -> Cached embeddings saved to: {cache_file}", flush=True)
        
    return embeddings, labels

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def train_and_evaluate_swin_b(modality_name, exp_id):
    print("="*75)
    print(f"Phase 3 Transformer Training: {exp_id} ({modality_name.upper()} Modality Swin-B)")
    print("="*75)
    
    output_dir = Path(f"outputs/transformer/{modality_name}_swin_b")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(f"outputs/transformer/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_dir / "metrics.json"
    if metrics_file.exists():
        print(f" -> [SKIP] Experiment {exp_id} ({modality_name.upper()}) is already completed!", flush=True)
        with open(metrics_file, "r") as f:
            return json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" -> Execution Device: {device} (threads: {torch.get_num_threads()})", flush=True)

    # 1. Load Data Splits
    processed_dir = Path("data/processed")
    df_train = pd.read_csv(processed_dir / "train.csv", low_memory=False)
    df_val = pd.read_csv(processed_dir / "validation.csv", low_memory=False)
    df_test = pd.read_csv(processed_dir / "test.csv", low_memory=False)

    df_train_mod = df_train[df_train['modality'] == modality_name].copy().reset_index(drop=True)
    df_val_mod = df_val[df_val['modality'] == modality_name].copy().reset_index(drop=True)
    df_test_mod = df_test[df_test['modality'] == modality_name].copy().reset_index(drop=True)

    print(f" -> {modality_name.upper()} Dataset Size: Train={len(df_train_mod)}, Val={len(df_val_mod)}, Test={len(df_test_mod)}", flush=True)

    # 2. Build Model & Count Parameters
    model = build_swin_b_model(num_classes=5, freeze_backbone=True).to(device)
    total_p, trainable_p, frozen_p = count_parameters(model)
    print(f" -> Total Parameters:     {total_p:,} ({total_p/1e6:.2f}M)")
    print(f" -> Trainable Parameters: {trainable_p:,} ({trainable_p/1e6:.4f}M) [Linear Head: 1024 -> 5]")
    print(f" -> Frozen Parameters:    {frozen_p:,} ({frozen_p/1e6:.2f}M) [Swin-B Backbone]")

    # Save Configuration
    config = {
        "experiment_id": exp_id,
        "modality": modality_name,
        "model_architecture": "Swin-B (Swin Transformer Base)",
        "weights": "Swin_B_Weights.DEFAULT (ImageNet-1K)",
        "input_resolution": [224, 224, 3],
        "feature_dimension": 1024,
        "freeze_backbone": True,
        "parameters": {
            "total": total_p,
            "trainable": trainable_p,
            "frozen": frozen_p,
            "trainable_components": ["head.weight", "head.bias"],
            "frozen_components": ["features.*", "norm.*"]
        },
        "dataset_splits": {
            "train": len(df_train_mod),
            "validation": len(df_val_mod),
            "test": len(df_test_mod)
        },
        "class_mapping": CLASS_TO_IDX,
        "training_hyperparameters": {
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "epochs": 20,
            "scheduler": "CosineAnnealingLR",
            "batch_size": 256,
            "model_selection_metric": "Validation Macro-F1",
            "early_stopping_patience": 6
        }
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    print(f" -> Saved configuration to: {output_dir / 'config.json'}")

    eval_tf = get_transforms()
    batch_size = 128
    
    train_raw_loader = DataLoader(ImageDataset(df_train_mod, transform=eval_tf), batch_size=batch_size, shuffle=False, num_workers=0)
    val_raw_loader = DataLoader(ImageDataset(df_val_mod, transform=eval_tf), batch_size=batch_size, shuffle=False, num_workers=0)
    test_raw_loader = DataLoader(ImageDataset(df_test_mod, transform=eval_tf), batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Extract / Load Features
    train_emb, train_lbl = extract_swin_embeddings(model, train_raw_loader, device, cache_dir / f"{modality_name}_train.pt")
    val_emb, val_lbl = extract_swin_embeddings(model, val_raw_loader, device, cache_dir / f"{modality_name}_val.pt")
    test_emb, test_lbl = extract_swin_embeddings(model, test_raw_loader, device, cache_dir / f"{modality_name}_test.pt")

    train_emb_loader = DataLoader(EmbeddingDataset(train_emb, train_lbl), batch_size=256, shuffle=True)
    val_emb_loader = DataLoader(EmbeddingDataset(val_emb, val_lbl), batch_size=256, shuffle=False)
    test_emb_loader = DataLoader(EmbeddingDataset(test_emb, test_lbl), batch_size=256, shuffle=False)

    # 4. Train Classification Head with Validation Macro-F1 Model Selection
    head = model.head.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_macro_f1 = -1.0
    patience = 6
    patience_counter = 0
    best_head_state = None

    history = []
    print("\n -> Training Swin-B Classification Head (Selection Metric: Validation Macro-F1)...", flush=True)
    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        head.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for x, y in train_emb_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = head(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            train_targets.extend(y.cpu().numpy())

        scheduler.step()
        train_loss /= len(train_emb_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='macro', zero_division=0)

        # Validation
        head.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_emb_loader:
                x, y = x.to(device), y.to(device)
                out = head(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                val_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        val_loss /= len(val_emb_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        epoch_time = time.time() - epoch_start

        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 4),
            'train_macro_f1': round(train_f1, 4),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 4),
            'val_macro_f1': round(val_f1, 4),
            'lr': scheduler.get_last_lr()[0],
            'epoch_time_sec': round(epoch_time, 2)
        })

        print(f"Epoch [{epoch:02d}/{epochs}] - Train Loss: {train_loss:.4f} | Train Macro-F1: {train_f1:.4f} || Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f} | Time: {epoch_time:.2f}s", flush=True)

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_head_state = head.state_dict().copy()
            patience_counter = 0
            print(f"    >>> Best Checkpoint Saved! (Val Macro-F1: {best_val_macro_f1:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f" -> Early stopping triggered at Epoch {epoch}.", flush=True)
                break

    total_train_time = time.time() - start_train_time
    print(f"\n -> Head Training Completed in {total_train_time:.1f} seconds.", flush=True)

    # 5. Assemble Full Best Model and Save Checkpoint
    model.head.load_state_dict(best_head_state)
    best_model_path = output_dir / "best_model.pth"
    torch.save(model.state_dict(), best_model_path)
    print(f" -> Saved complete best Swin-B model to: {best_model_path} ({best_model_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    # Save Training History CSV & Curves
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(output_dir / "training_history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df_hist['epoch'], df_hist['train_loss'], label='Train Loss', color='blue', linewidth=2)
    axes[0].plot(df_hist['epoch'], df_hist['val_loss'], label='Val Loss', color='orange', linewidth=2)
    axes[0].set_title(f"{exp_id} - Loss Curves ({modality_name.capitalize()} Swin-B)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(df_hist['epoch'], df_hist['train_macro_f1'], label='Train Macro-F1', color='green', linewidth=2)
    axes[1].plot(df_hist['epoch'], df_hist['val_macro_f1'], label='Val Macro-F1', color='red', linewidth=2)
    axes[1].set_title(f"{exp_id} - Macro-F1 Score Curves ({modality_name.capitalize()} Swin-B)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1 Score")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300)
    plt.close()

    # 6. Evaluate strictly once on frozen test set
    print(f"\n -> Evaluating Frozen Best Checkpoint on Pure Untouched Test Set ({len(df_test_mod)} samples)...", flush=True)
    head.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for x, y in test_emb_loader:
            x = x.to(device)
            out = head(x)
            test_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            test_targets.extend(y.numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)

    test_acc = accuracy_score(test_targets, test_preds)
    test_macro_f1 = f1_score(test_targets, test_preds, average='macro', zero_division=0)
    test_precision = precision_score(test_targets, test_preds, average='macro', zero_division=0)
    test_recall = recall_score(test_targets, test_preds, average='macro', zero_division=0)

    per_class_f1 = f1_score(test_targets, test_preds, average=None, zero_division=0)
    per_class_f1_dict = {CLASS_NAMES[i]: round(float(per_class_f1[i]), 4) for i in range(len(CLASS_NAMES))}

    # Measure Standardized End-to-End Inference Latency on 100 test samples
    print(" -> Measuring standardized end-to-end inference latency...", flush=True)
    model.eval()
    sample_loader = DataLoader(ImageDataset(df_test_mod.head(100), transform=eval_tf), batch_size=1, shuffle=False)
    latencies = []
    with torch.no_grad():
        for img, _ in sample_loader:
            t0 = time.time()
            _ = model(img.to(device))
            latencies.append((time.time() - t0) * 1000.0)
    avg_latency_ms = float(np.median(latencies[5:]))  # discard warmup

    print("="*60)
    print(f"FINAL TEST RESULTS: {exp_id} ({modality_name.upper()} Swin-B)")
    print("="*60)
    print(f"  - Test Accuracy:          {test_acc*100:.2f}%")
    print(f"  - Test Macro-F1:          {test_macro_f1:.4f}")
    print(f"  - Test Macro-Precision:   {test_precision:.4f}")
    print(f"  - Test Macro-Recall:      {test_recall:.4f}")
    print(f"  - Per-Class F1 Scores:    {per_class_f1_dict}")
    print(f"  - End-to-End Latency:     {avg_latency_ms:.2f} ms/sample")
    print("="*60, flush=True)

    # Save Classification Report
    cls_report = classification_report(test_targets, test_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(cls_report, f, indent=4)

    # Save Confusion Matrix
    cm = confusion_matrix(test_targets, test_preds)
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

    plt.title(f"Confusion Matrix: {exp_id} ({modality_name.capitalize()} Swin-B)", pad=20)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # Save Metrics JSON
    metrics_summary = {
        "experiment_id": exp_id,
        "modality": modality_name,
        "model_architecture": "Swin-B",
        "dataset_size": {
            "train": len(df_train_mod),
            "validation": len(df_val_mod),
            "test": len(df_test_mod)
        },
        "training_time_seconds": round(total_train_time, 2),
        "inference_latency_ms_per_sample": round(avg_latency_ms, 2),
        "parameter_count_millions": round(total_p / 1e6, 2),
        "trainable_parameter_count_millions": round(trainable_p / 1e6, 4),
        "frozen_parameter_count_millions": round(frozen_p / 1e6, 2),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["fruit", "leaf", "all"], default="all")
    args = parser.parse_args()

    if args.modality in ["fruit", "all"]:
        fruit_res = train_and_evaluate_swin_b("fruit", "E3-F")
    if args.modality in ["leaf", "all"]:
        leaf_res = train_and_evaluate_swin_b("leaf", "E2-L")

    print("\nPhase 3 Execution Completed Successfully!")
