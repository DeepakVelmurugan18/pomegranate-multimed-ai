# Dataset Strategy & Anti-Leakage Protocol

## 1. Dataset Ingestion & Phase 1 Audit Protocol

### Target Datasets
1. **Pomegranate Disease Dataset (Kaggle / Mendeley Data)**: Primary source for pomegranate leaf and fruit disease images.
2. **Field Pomegranate Multimodal Collection**: Secondary literature datasets.

### Phase 1 Audit Protocol
During Phase 1 (Dataset Ingestion & Audit), a automated inspection script will analyze raw dataset files and generate `dataset_report.json` containing:
- Total image count ($N_{\text{total}}$)
- Modality breakdown ($N_{\text{leaf}}$, $N_{\text{fruit}}$)
- Pairing audit: Determine whether leaf and fruit images are **genuinely paired** (Mode A: same physical plant case) or **unpaired** (Mode B: separate leaf and fruit collections).
- Image dimensions & RGB channel verification.
- Corrupt image filtering & duplicate hash detection (MD5 & perceptual pHash).
- Exact class count matrix across modalities.

---

## 2. Provisional Taxonomy & Delta Report Protocol

Target Study Taxonomy:
1. `Healthy`
2. `Alternaria`
3. `Anthracnose`
4. `Bacterial Blight`
5. `Cercospora`

> [!IMPORTANT]  
> **Taxonomy Delta Protocol**: The 5 classes above represent the target study taxonomy. If Phase 1 reveals differences in raw folder names, extra classes (e.g. *Canker*), or missing classes, Antigravity will **not** silently delete or re-map classes. It will produce a **Taxonomy Delta Report** detailing:
> - Raw dataset class distribution
> - Proposed mapping / filtering rationale
> - Impact on sample counts
> 
> Phase 1 execution will STOP and wait for explicit user approval of the Taxonomy Delta Report before transforming dataset directories.

---

## 3. Unpaired vs Paired Modality Handling

- **Mode A — Genuine Paired Multimodal Data**: If Phase 1 identifies true 1-to-1 leaf-fruit image pairs from identical plant cases, joint multimodal samples $(I_{\text{leaf}}^{(i)}, I_{\text{fruit}}^{(i)}, y^{(i)})$ will be constructed.
- **Mode B — Class-Aligned Unpaired Multimodal Learning**: If leaf and fruit datasets are collected independently, samples will be grouped by shared disease class $y$. Batches will sample class-matched leaf and fruit images $(I_{\text{leaf}}, I_{\text{fruit}}, y)$ and explicitly document the pipeline as *class-aligned unpaired multimodal learning* rather than claiming plant-level biological pairing.

---

## 4. Group-Aware Stratified Splitting & Anti-Leakage Rules

### Stratified Ratios
- **Training Set**: 70%
- **Validation Set**: 15%
- **Test Set**: 15%

```
               RAW ACQUIRED DATASET
                         |
                         v
             DUPLICATE & CORRUPT REMOVAL
                         |
                         v
             GROUP-AWARE STRATIFIED SPLIT
            (Same Plant/Tree/Capture Session
              Locked to a Single Split)
                         |
       +-----------------+-----------------+
       |                 |                 |
       v                 v                 v
 TRAINING SET (70%)  VALIDATION (15%)  TEST SET (15%)
       |                 |                 |
       v                 v                 v
SELECTIVE GAN AUG    REAL ONLY        REAL ONLY
  (Train Only)       UNTOUCHED        UNTOUCHED & UNSEEN
       |                 |                 |
       v                 v                 v
AUGMENTED TRAIN       VAL EVAL         TEST EVAL
    DATASET
```

### Anti-Leakage Safeguards:
1. **Group-Aware Splitting**: Photos originating from the same physical plant, tree, or capture session will be assigned together to either Train, Validation, or Test to prevent biological leakage.
2. **Split-Before-Augment**: Splitting is executed **strictly before** any data augmentation or GAN generation.
3. **Training-Only GAN Execution**: GAN training occurs **exclusively on the 70% training split**.
4. **Pure Test Set**: Validation and Test sets remain 100% real, unseen, and unaugmented.

---

## 5. Selective GAN Strategy & QC Gates

### Unified Imbalance Ratio Formula
$$\text{Imbalance Ratio } (IR_c) = \frac{\max_k N_k}{N_c}$$
- Candidate minority class if $IR_c > 1.5$.
- Phase 1 will compute exact class counts $N_c$ and $IR_c$ values for all classes. GAN augmentation will target classes confirmed as minority after dataset inspection.

### GAN Quality Control (QC) Protocol
Synthetic images must pass automated QC checks before inclusion in the augmented training set:
1. **Validity Check**: Non-black image, standard deviation $std > 15.0$, no severe grid/checkerboard artifacts.
2. **Classifier Consistency Gate**: Pre-trained ResNet50 baseline prediction on synthetic image must match target class $c$ with confidence $P(y=c) \ge 0.80$.
3. **FID Score Metric**: Fréchet Inception Distance will be calculated and reported as a comparative quality distribution metric across ablation experiments.

### Non-GAN Fallback Protocol
If Phase 1 sample counts for minority classes are insufficient to stably train a class-conditional GAN, a documented fallback strategy using **Mixup, CutMix, and AutoAugment** will be executed and reported transparently in the thesis.

---

## 6. Grounded Severity Estimation Hierarchy

### Hierarchy for Severity Ground Truth:
1. **Level 1 — Real Expert Severity Labels**: Use existing expert severity labels (`Low`, `Medium`, `High`) if present in the raw dataset metadata.
2. **Level 2 — External Severity Metadata**: Use external expert percentage annotations if available.
3. **Level 3 — Lesion-Burden Proxy (Fallback)**: If explicit annotations are absent, compute visual lesion area coverage ($S = \frac{A_{\text{lesion}}}{A_{\text{total}}} \times 100\%$) via HSV color segmentation/Otsu thresholding.
   - Severity Tier Mapping for diseased cases:
     - `Low Severity`: $S \le 15\%$
     - `Medium Severity`: $15\% < S \le 35\%$
     - `High Severity`: $S > 35\%$
   - For `Healthy` samples: Severity is assigned as `N/A`.
   - This proxy will be validated for segmentation accuracy on a manual sample subset prior to use, and explicitly designated as a **lesion-burden proxy** in research reports.
