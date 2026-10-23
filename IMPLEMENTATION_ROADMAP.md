# Implementation Roadmap & Phase Breakdown

## 1. Project Directory Structure

```
pomegranate-disease-ai/
│
├── data/
│   ├── raw/
│   │   ├── leaf/               # Raw leaf images organized by class
│   │   └── fruit/              # Raw fruit images organized by class
│   ├── processed/              # Cleaned, audited, and split datasets
│   ├── train/                  # Group-aware stratified training split (70%)
│   ├── validation/             # Group-aware validation split (15%)
│   └── test/                   # Pure real unseen test split (15%)
│
├── gan/
│   ├── models/                 # Class-conditional GAN networks
│   ├── training/               # GAN training loop scripts
│   ├── generation/             # Minority class image generator
│   └── evaluation/             # Classifier consistency & FID quality gates
│
├── models/
│   ├── cnn_baseline/           # ResNet50 baseline model
│   ├── leaf_transformer/       # Swin-B leaf feature extractor
│   ├── fruit_transformer/      # Swin-B fruit feature extractor
│   └── multimodal/             # Feature fusion, classifier & severity heads
│
├── severity/                   # Lesion area thresholding & severity proxy estimator
│
├── xai/                        # Grad-CAM++ & Swin Attention Rollout engines
│
├── evaluation/
│   ├── metrics/                # Accuracy, Precision, Recall, Macro-F1 scripts
│   ├── confusion_matrix/       # Confusion matrix generators & plotters
│   ├── ablation/               # E1-E8 ablation study runner & aggregator
│   └── reports/                # Generated JSON/CSV evaluation metrics
│
├── notebooks/                  # Interactive research notebooks
├── configs/                    # YAML configuration files
├── scripts/                    # CLI scripts for pipeline stages
├── outputs/                    # Model checkpoints, plots, XAI maps
├── app/                        # Interactive Web Application UI & API backends
├── tests/                      # PyTest unit tests for pipeline verification
├── docs/                       # Research documentation & 7-chapter paper draft
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview & usage guide
```

---

## 2. Phase-by-Phase Execution Roadmap

### Phase 0: Project Understanding & Design (COMPLETED & REVISED)
- **Goal**: Formulate project specifications, technical risk analysis, dataset strategy, system architecture, experiment plan, and implementation roadmap.
- **Outputs**: `PROJECT_SPEC.md`, `ARCHITECTURE.md`, `DATASET_PLAN.md`, `EXPERIMENT_PLAN.md`, `IMPLEMENTATION_ROADMAP.md`.
- **Status**: 🟢 **Phase 0 Revisions Completed & Approved.**

---

### Phase 1: Dataset Investigation, Ingestion & Audit Pipeline (COMPLETED)
- 46,974 valid images verified across 5 classes (`Healthy`, `Alternaria`, `Anthracnose`, `Bacterial Blight`, `Cercospora`).
- Mode B (*Class-Aligned Unpaired Multimodal Learning*) confirmed; group-aware stratified 70/15/15 split executed.
- Modality-specific imbalance diagnostic confirmed selective GAN candidates for Fruit (`Alternaria`, `Cercospora`).
- **Status**: 🟢 **Phase 1 & Phase 1.1 Completed & Approved.**

---

### Phase 2: Preprocessing & Baseline Models (COMPLETED)
- Trained ResNet50 single-modality baselines on pure real splits:
  - **E1-F (Fruit ResNet50)**: **95.92% Test Accuracy**, **0.9540 Test Macro-F1**.
  - **E1-L (Leaf ResNet50)**: **70.03% Test Accuracy**, **0.7184 Test Macro-F1**.
- Model checkpoints, training curves, confusion matrices, classification reports, and `metrics.json` saved in `outputs/baseline/`.
- **Status**: 🟢 **Phase 2 Completed. Ready for Review.**

---

### Phase 3: Single-Modality Transformer Models (NEXT STEP)
- Train Swin-B Leaf Transformer (E2) and Swin-B Fruit Transformer (E3). Compare against CNN baseline.
- 🛑 **STOP -> Confirm Transformer feature extraction performance.**

### Phase 4: Selective GAN Augmentation & Quality Control
- Train Class-Conditional GAN on training set minority classes ($IR_c > 1.5$). Apply classifier consistency gate ($P(y=c) \ge 0.80$) and FID evaluation. (Deploy Mixup/CutMix fallback if GAN sample size is insufficient).
- 🛑 **STOP -> Review synthetic image quality report.**

### Phase 5: Multimodal Feature Fusion & Classifier
- Train Multimodal Transformer with Concatenation (E4, E5, E6) and Cross-Attention (E7).
- 🛑 **STOP -> Review multimodal fusion accuracy and Macro-F1.**

### Phase 6: Severity Estimation Framework
- Implement joint multi-task severity estimation head (E8). Predict `Low`, `Medium`, `High` for diseased samples (`N/A` for `Healthy`).
- 🛑 **STOP -> Review severity accuracy and MAE.**

### Phase 7: Explainable AI Engine
- Integrate Grad-CAM++ for ResNet50 and Attention Rollout for Swin Transformer. Generate visual explanation overlays on leaf and fruit lesion regions.
- 🛑 **STOP -> Review XAI heatmaps.**

### Phase 8: Comprehensive Ablation Study & Final Benchmarking
- Aggregate results across E1 through E8. Generate publication-ready comparative tables, Macro-F1 plots, confusion matrices, and latency reports.
- 🛑 **STOP -> Review final research thesis metrics.**

### Phase 9: Interactive Web Application
- Build user interface allowing dual image upload, disease classification, severity score, confidence, and XAI heatmap overlays.

### Phase 10: 7-Chapter Research Documentation
- Author 7-chapter research thesis/paper draft citing verified reference literature.

### Phase 11: Final Validation Audit
- Execute final 15-point verification checklist and sign off repository.
