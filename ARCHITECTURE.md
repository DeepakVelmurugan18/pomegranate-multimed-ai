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
- **Baseline Fusion (Exp 4-5)**: Feature Concatenation + Projection MLP ($[F_{\text{leaf}} \parallel F_{\text{fruit}}] \to \text{LayerNorm} \to \text{GELU} \to \text{MLP}$).
- **Non-Degenerate Token-Level Cross-Attention (Exp 6-7)**:
  Rather than performing cross-attention over global pooled vectors (which reduces degenerate scalar attention to an additive projection), the framework projects 1024-d embeddings into $K=8$ subspace tokens ($X \in \mathbb{R}^{B \times 8 \times 128}$):
  $$Q_L = X_L W_Q, \quad K_F = X_F W_K, \quad V_F = X_F W_V$$
  $$\text{Attn}(Q_L, K_F, V_F) = \text{softmax}\left(\frac{Q_L K_F^T}{\sqrt{d_k}}\right) V_F \in \mathbb{R}^{B \times 8 \times 128}$$
  The $8 \times 8$ cross-attention matrix is computed across 8 distinct keys, generating true entropy-rich attention distributions where foliar lesion tokens attend to pericarp rind tokens.
- **Decoupled Evaluation Protocol**:
  To eliminate label-leakage artifacts in unpaired datasets:
  1. *Coupled Oracle*: Class-matched pairing (evaluates theoretical upper bound).
  2. *Label-Agnostic Pairing*: Fruit partner assigned with zero label knowledge.
  3. *Dual-Organ Late Fusion*: Independent inference on available organs combined via calibrated confidence weighting:
     $$\hat{y} = \text{argmax}\left(w_{\text{leaf}} \cdot P(y \mid X_{\text{leaf}}) + w_{\text{fruit}} \cdot P(y \mid X_{\text{fruit}})\right)$$

---

## 3. Multi-Task Output Heads

### Disease Classifier Head
- **Classes**: 5 classes (`Healthy`, `Alternaria`, `Anthracnose`, `Bacterial Blight`, `Cercospora`).
- **Layers**: Linear($2048 \to 512$) $\to$ LayerNorm $\to$ GELU $\to$ Dropout(0.2) $\to$ Linear($512 \to 5$) $\to$ Softmax.

### Objective Lesion-Burden Severity Proxy Head
- **Target Tiers**: 3 tiers (`Low <10%`, `Medium 10–25%`, `High >25%`) evaluated strictly for diseased samples. For `Healthy` samples, Severity output is designated as `N/A`.
- **Methodological Status**: Formulated explicitly as an objective surface lesion-burden proxy ($S = A_{\text{lesion}} / A_{\text{total}} \times 100\%$) calibrated to phytosanitary standard area diagrams, rather than unvalidated clinical ground truth.

---

## 4. Pathology Attribution & Diagnostics Framework

1. **Layer-4 Grad-CAM**: Applied to the final convolutional block of ResNet50 baseline backbones to generate authentic gradient-weighted spatial heatmaps over necrotic perimeters.
2. **Token Cross-Attention Heatmaps**: Inspects the $(8 \times 8)$ inter-token weight matrix to trace cross-modal feature dependencies.
3. **Multi-Seed Statistical Reporting**: 3–5 random seed evaluations reporting $\text{Mean} \pm \text{Std}$, 95% Confidence Intervals, and McNemar's significance tests ($p < 0.05$).
4. **Diagnostic Deliverables**: Normalized confusion matrices, Expected Calibration Error (ECE), and inference latency benchmarks.
