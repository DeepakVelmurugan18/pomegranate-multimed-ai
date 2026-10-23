# Experimental & Ablation Strategy

## 1. Unconfounded 8-Stage Ablation Matrix

To systematically validate the contribution of every architectural component without confounding variables, the project will execute 8 controlled experiments.

| Exp ID | Experiment Name | Modalities | Architecture Backbone | GAN Strategy | Fusion Method | Severity Head |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E1** | CNN Baseline | Leaf (or Fruit) | ResNet50 | None | None | No |
| **E2** | Leaf Transformer | Leaf Only | Swin-B | None | None | No |
| **E3** | Fruit Transformer | Fruit Only | Swin-B | None | None | No |
| **E4** | Multimodal Transformer | Leaf + Fruit | Swin-B | None | Concatenation | No |
| **E5** | Multimodal + Naive GAN | Leaf + Fruit | Swin-B | Naive GAN (All Classes) | Concatenation | No |
| **E6** | Multimodal + Selective GAN | Leaf + Fruit | Swin-B | Selective GAN (Minority Only) | Concatenation | No |
| **E7** | Multimodal + Cross-Attention | Leaf + Fruit | Swin-B | Selective GAN (Minority Only) | **Cross-Attention** | No |
| **E8** | Proposed Full Framework | Leaf + Fruit | Swin-B | Selective GAN (Minority Only) | **Cross-Attention** | **Yes** (+ XAI) |

---

## 2. Research Questions Answered by Ablation Matrix

- **Q1 (Multimodal Value)**: Does combining Leaf + Fruit (E4) outperform single-modality Leaf (E2) or Fruit (E3) Transformers?
- **Q2 (GAN Augmentation Impact)**: Does adding GAN data augmentation (E5) improve baseline multimodal performance (E4)?
- **Q3 (Selective vs Indiscriminate GAN)**: Does targeting GAN augmentation strictly at minority classes (E6) outperform indiscriminate all-class GAN synthesis (E5)?
- **Q4 (Fusion Strategy Evaluation)**: Does Cross-Attention multimodal fusion (E7) outperform simple Concatenation fusion (E6)?
- **Q5 (Multi-Task Severity Impact)**: How does adding a joint severity estimation head (E8) affect primary disease classification performance relative to single-task E7?

---

## 3. Evaluation Metrics

### A. Primary Classification Metrics
- **Macro-F1 Score (Primary Decision Metric)**: Unweighted average F1-score across all 5 disease classes, providing a fair evaluation of imbalanced class handling:
  $$\text{Macro-F1} = \frac{1}{5} \sum_{c=1}^{5} F1_c$$
- **Per-Class F1-Score**: Evaluates individual class performance (crucial for minority classes).
- **Weighted-F1 & Overall Accuracy**: Standard overall performance measures.
- **Confusion Matrix**: Normalized $5 \times 5$ confusion matrix visual breakdown.

### B. Severity Estimation Metrics
- **Severity Accuracy**: Classification accuracy across `Low`, `Medium`, and `High` severity tiers for diseased samples.
- **Mean Absolute Error (MAE)**: Numerical distance between predicted severity index and ground-truth severity index.

### C. GAN & Computational Metrics
- **Fréchet Inception Distance (FID)**: Comparative feature distribution distance between real and synthetic training samples.
- **Classifier Consistency Rate**: Percentage of synthetic images passing pre-trained baseline confidence threshold ($P(y=c) \ge 0.80$).
- **Computational Cost**: Parameter Count (M), Inference Latency (ms/sample), and FLOPs.

### D. Explainable AI (XAI) Evaluation Strategy
- **Attention Overlap / Insertion-Deletion AUC**: Evaluates attention map concentration over visible lesion regions.
- **Representative Case Audit**: Visual inspection of heatmaps across correct predictions, edge cases, and misclassifications.
- **Cross-Modal Explanation Consistency**: Verification that leaf heatmaps highlight foliar spots while fruit heatmaps highlight epicarp lesions for the same disease class.

---

## 4. Final Thesis Comparative Benchmark Table

| Exp | Model Configuration | Leaf | Fruit | GAN Strategy | Fusion Method | Severity Head | Accuracy (%) | Macro Precision | Macro Recall | Macro F1 | Inference Latency (ms) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | ResNet50 Baseline | ✓ | – | None | – | No | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **E2** | Leaf Transformer (Swin-B) | ✓ | – | None | – | No | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **E3** | Fruit Transformer (Swin-B) | – | ✓ | None | – | No | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **E4** | Multimodal Transformer | ✓ | ✓ | None | Concatenation | No | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **E5** | Naive GAN + Multimodal | ✓ | ✓ | Naive GAN | Concatenation | No | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **E6** | Selective GAN + Multimodal | ✓ | ✓ | Selective GAN | Concatenation | No | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **E7** | Selective GAN + Cross-Attention | ✓ | ✓ | Selective GAN | Cross-Attention | No | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **E8** | Proposed Full Framework | ✓ | ✓ | Selective GAN | Cross-Attention | **Yes** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 5. Experimental Standardization
- **Random Seeds**: Fixed `seed = 42` across PyTorch, NumPy, Python `random`, and CUDA backends.
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 1e-2$).
- **Learning Rate Schedule**: $1e-4$ base LR with Cosine Annealing scheduler.
- **Epochs & Early Stopping**: 50 epochs with early stopping (patience = 8 on validation loss).
