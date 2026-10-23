# Project Specification: Pomegranate-MultiMed AI

## Title
**An Explainable Multimodal Transformer Framework with Selective GAN Augmentation for Pomegranate Disease Detection and Severity Assessment**

**Short Project Name**: `Pomegranate-MultiMed AI`

---

## 1. Project Vision & Research Objectives
Plant disease identification is a critical domain in agricultural computer vision. Conventional AI systems rely on a single visual modality (either leaf images or fruit images), which often misses complementary physiological symptoms present across different plant organs. 

**Pomegranate-MultiMed AI** introduces a multimodal deep learning framework combining:
1. Dual visual inputs: **Leaf Images** + **Fruit Images**.
2. **Selective GAN Augmentation** strictly targeted at minority disease classes in the training set (with documented non-GAN fallback if dataset size is insufficient).
3. **Dual Vision Transformer Branches** (Primary backbone: **Swin-B** for both leaf and fruit branches to ensure controlled cross-modality evaluation).
4. **Multimodal Feature Fusion** evaluating Feature Concatenation vs Cross-Attention fusion.
5. **Multi-Task Output**: 5-class disease recognition and 3-tier disease severity estimation for diseased cases.
6. **Explainable AI (XAI)** visualizations (Grad-CAM++ for CNN baseline, Attention Rollout for Swin Transformer) grounding model predictions in visual pathology.

### Primary Research Contribution
The contribution is **not** claiming GANs or Transformers as novel individual modules. Rather, it lies in the **rigorous integration, controlled unconfounded ablation, and empirical evaluation** of:
`Selective GAN Augmentation + Dual Swin Transformer Branches + Multimodal Feature Fusion + Disease Classification + Severity Proxy Estimation + Explainable Pathology Mapping`.

---

## 2. Target Disease Taxonomy (Provisional)

| Class ID | Disease Class Name | Description / Visual Characteristics |
| :--- | :--- | :--- |
| **1** | **Healthy** | Normal leaf/fruit tissue with no visible lesion spots or discoloration. |
| **2** | **Alternaria** | Alternaria fruit spot / leaf spot causing small black necrotic spots. |
| **3** | **Anthracnose** | Circular brownish-black lesions with dark margins on leaves and fruits. |
| **4** | **Bacterial Blight** | *Xanthomonas axonopodis* pv. *punicae*, brown/black water-soaked lesions. |
| **5** | **Cercospora** | Cercospora leaf spot, small brown circular or irregular spots. |

> [!IMPORTANT]  
> **Provisional Taxonomy Protocol**: The 5 classes listed above are proposed as the primary target study taxonomy. In Phase 1 (Dataset Audit), the actual acquired dataset folder structure and class labels will be verified. If discrepancies are found (e.g. extra classes like *Canker*, or alternative class names), Antigravity will produce a **Taxonomy Delta Report** and obtain explicit user approval before performing any taxonomy transformation or dataset filtering.

---

## 3. Visual Modalities & Multimodal Strategy

### Visual Modalities
- **Modality A (Leaf)**: Captures foliar lesion structure, vein necrosis, spot density, and micro-texture.
- **Modality B (Fruit)**: Captures epicarp spots, rind cracking, necrotic lesions, and fruit rots.

### Multimodal Data Strategy (Determined in Phase 1)
Rather than assuming biological pairing upfront, Phase 1 will audit the dataset to establish the true modality setup:
- **Mode A — True Paired Multimodal Data**: If Phase 1 verifies genuine paired observations ($Leaf_i + Fruit_i$ from the exact same physical plant case), the model will perform joint paired multimodal fusion.
- **Mode B — Class-Aligned Unpaired Multimodal Learning**: If leaf and fruit datasets are unpaired ($Leaf_i$ and $Fruit_j$ from different plants), samples will be batched by shared disease label. The framework will be explicitly described as *class-aligned unpaired multimodal learning* to maintain scientific rigor and prevent misleading biological claims.

---

## 4. Grounded Severity Estimation Hierarchy

Disease classification and severity estimation are kept conceptually distinct:
- **Disease Head**: Predicts one of 5 target disease classes (including `Healthy`).
- **Severity Head**: Predicts disease burden for diseased cases (`Low`, `Medium`, `High`). For `Healthy` samples, Severity is defined as `N/A` (Not Applicable).

### Severity Ground-Truth Hierarchy
1. **Level 1 — Real Expert Severity Labels**: Use ground-truth severity annotations if present in the raw dataset.
2. **Level 2 — Provenance-Documented External Severity Metadata**: Use external expert metadata if available.
3. **Level 3 — Lesion-Burden Proxy (Fallback)**: If explicit labels are absent, compute an objective visual lesion ratio ($S = \frac{\text{Lesion Area}}{\text{Total Surface Area}} \times 100\%$) via color thresholding/HSV segmentation mapped to `Low` ($\le 15\%$), `Medium` ($15–35\%$), and `High` ($> 35\%$). This will be explicitly designated as a **lesion-burden proxy** rather than ground-truth clinical diagnosis.

---

## 5. Strict Anti-Leakage & Data Split Rules

1. **Group-Aware Stratified Splitting**:
   - Dataset split into Train (70%), Validation (15%), and Test (15%) **strictly before** any augmentation or GAN processing.
   - **Biological & Session Leakage Protection**: All images originating from the same physical plant, tree, or capture session will be forced into the *same* split.
2. **Training-Only GAN Augmentation**:
   - GAN training and synthetic image generation occur **exclusively on the 70% training split**.
   - Test and Validation sets remain 100% real, unseen, and unaugmented.
3. **Primary Evaluation Metric**:
   - **Macro-F1 Score** will serve as the primary decision metric to rigorously evaluate handling of class imbalance.
