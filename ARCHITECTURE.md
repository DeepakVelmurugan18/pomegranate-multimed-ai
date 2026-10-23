# System Architecture: Pomegranate-MultiMed AI

## 1. End-to-End System Architecture Diagram

```
                              +--------------------------+
                              |    RAW POMEGRANATE DATA  |
                              +------------+-------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
        +-----------------------+                     +-----------------------+
        |  LEAF IMAGE DATASET   |                     |  FRUIT IMAGE DATASET  |
        +-----------+-----------+                     +-----------+-----------+
                    |                                             |
                    v                                             v
        +-----------------------+                     +-----------------------+
        |   DATA CLEANING &     |                     |   DATA CLEANING &     |
        |  DUPLICATE CHECK      |                     |  DUPLICATE CHECK      |
        +-----------+-----------+                     +-----------+-----------+
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
                             +--------------------------+
                             |   GROUP-AWARE SPLITTING  |
                             |  (Same Plant/Tree Locked |
                             |   to Single Train/Val/Test|
                             +-------------+------------+
                                           |
                                           v
                             +--------------------------+
                             |    CLASS DISTRIBUTION    |
                             |  ANALYSIS (IR_c Calculation)
                             +-------------+------------+
                                           |
                                           v
                             +--------------------------+
                             |  SELECTIVE GAN ENGINE    |
                             | (Class-Conditional GAN   |
                             |   on Minority Train Only)|
                             +-------------+------------+
                                           |
                                           v
                             +--------------------------+
                             | AUGMENTED TRAINING DATA  |
                             +-------------+------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
        +-----------------------+                     +-----------------------+
        |  LEAF TRANSFORMER     |                     |  FRUIT TRANSFORMER    |
        | (Swin-B Backbone)     |                     | (Swin-B Backbone)     |
        +-----------+-----------+                     +-----------+-----------+
                    |                                             |
                    v                                             v
        +-----------------------+                     +-----------------------+
        | LEAF EMBEDDING (F_L)  |                     | FRUIT EMBEDDING (F_F) |
        |      [1 x 768]        |                     |      [1 x 768]        |
        +-----------+-----------+                     +-----------+-----------+
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
                             +--------------------------+
                             |     FUSION MODULE        |
                             |  - Concatenation (Base)  |
                             |  - Cross-Attention (Exp) |
                             +-------------+------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
        +-----------------------+                     +-----------------------+
        | DISEASE CLASSIFIER    |                     | SEVERITY ESTIMATOR    |
        | (5 Classes + Softmax) |                     | (Low / Med / High)    |
        | [Healthy, Alternaria, |                     | (For diseased cases;  |
        |  Blight, Anthracnose, |                     |  Healthy = N/A)       |
        |  Cercospora]          |                     |                       |
        +-----------+-----------+                     +-----------+-----------+
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
                             +--------------------------+
                             |  EXPLAINABLE AI ENGINE   |
                             | (Grad-CAM++ / Attention) |
                             +-------------+------------+
                                           |
                                           v
                             +--------------------------+
                             |     FINAL DIAGNOSIS      |
                             | (Disease + Severity + XAI|
                             +--------------------------+
```

---

## 2. Component Specifications

### A. Preprocessing & Augmentation Pipeline
- **Input Dimensions**: $224 \times 224 \times 3$ RGB.
- **Normalization**: ImageNet mean ($\mu = [0.485, 0.456, 0.406]$) and std ($\sigma = [0.229, 0.224, 0.225]$).
- **Standard Augmentations**: Random horizontal flip, random rotation ($\pm 15^\circ$), color jitter.

### B. Baseline & Transformer Architectures
- **Primary CNN Baseline**: **ResNet50** (evaluated on single modalities in Exp 1).
- **Primary Transformer Backbone**: **Swin-B (Swin Transformer Base)** for both Leaf and Fruit branches to maintain architectural consistency across modalities.

### C. Class Imbalance & Selective GAN Engine
- **Imbalance Ratio Formula**:
  $$IR_c = \frac{\max_k N_k}{N_c}$$
- **Minority Selection Rule**: Candidate minority class if $IR_c > 1.5$. Phase 1 will compute actual distribution before locking target classes.
- **Primary Architecture Choice**: **Class-Conditional GAN** (e.g. AC-GAN / SAGAN architecture for class-conditioned synthesis).
- **Non-GAN Fallback Protocol**: If Phase 1 reveals that minority class sample sizes are too small to stably converge a GAN without severe mode collapse, a documented fallback strategy utilizing Mixup, CutMix, and AutoAugment will be deployed and reported transparently.
- **Quality Control Filter**:
  1. Classifier consistency gate: Synthetic image must be predicted as target class $c$ by pre-trained baseline classifier with confidence $P(y=c) \ge 0.80$.
  2. FID score reported as a comparative quality distribution metric across experiments.

### D. Multimodal Feature Fusion Strategy
- **Baseline Fusion (Exp 4-6)**: Concatenation + Projection MLP ($[F_{\text{leaf}} \parallel F_{\text{fruit}}] \to \text{ReLU} \to \text{MLP}$).
- **Experimental Fusion (Exp 7-8)**: **Cross-Attention Multimodal Fusion** (Leaf Queries attending to Fruit Keys/Values, and vice versa). Cross-Attention will be evaluated as an experimental hypothesis rather than assumed to be superior upfront.

---

## 3. Multi-Task Output Heads

### Disease Classifier Head
- **Classes**: 5 classes (`Healthy`, `Alternaria`, `Anthracnose`, `Bacterial Blight`, `Cercospora`).
- **Layers**: Linear($512 \to 128$) $\to$ Dropout(0.3) $\to$ Linear($128 \to 5$) $\to$ Softmax.

### Severity Estimator Head
- **Target Tiers**: 3 tiers (`Low`, `Medium`, `High`) evaluated **only on diseased samples**. For `Healthy` samples, Severity output is designated as `N/A`.
- **Layers**: Linear($512 \to 64$) $\to$ ReLU $\to$ Linear($64 \to 3$) $\to$ Softmax.

---

## 4. Explainable AI (XAI) Framework

1. **Grad-CAM++**: Applied to the final convolutional block of ResNet50 baseline.
2. **Transformer Attention Rollout**: Aggregates multi-head self-attention maps across Swin Transformer blocks.
3. **Evaluation Strategy**:
   - Visual inspection of heatmap alignment over lesion regions.
   - Qualitative comparison across correct and misclassified samples.
   - Explanation consistency across leaf and fruit modalities.
