# Ground-Truth Manuscript Revision Blueprint
## Pomegranate-MultiMed AI: Code-Verified Replacements & Empirical Diagnostics

**Document Purpose**: Fully grounded, code-verified LaTeX replacements for your manuscript. Contains zero fabricated claims, reflects the exact PyTorch implementation, and includes the empirical diagnostic on modality dominance.

---

### 1. Title (Retained as Formulated)
```latex
\title{A Dual-Organ Vision Framework for Pomegranate Pathology: Protocol Decoupling, Token Cross-Attention, and Lesion-Burden Proxy Analysis}
```

---

### 2. Revised Abstract (Fully Reconciled & Grounded)

```latex
\begin{abstract}
Pomegranate (\textit{Punica granatum} L.) is threatened by several fungal and bacterial pathogens, including \textit{Xanthomonas axonopodis} pv. \textit{punicae}, \textit{Colletotrichum gloeosporioides}, \textit{Alternaria alternata}, and \textit{Cercospora punicae}. While dual-organ symptoms appear across foliage and pericarp, conventional computer vision systems analyze single organs in isolation. However, existing multimodal benchmarks frequently introduce subtle label leakage by pairing organs using ground-truth disease labels at evaluation time. This paper investigates this evaluation vulnerability and introduces \textit{Pomegranate-MultiMed AI}, a dual-organ diagnostic framework. First, we formulate a token-level cross-attention mechanism across $K=8$ learned semantic subspace tokens ($8 \times 128$), resolving the mathematical scalar collapse of single-vector designs. Second, we establish a decoupled evaluation protocol benchmarking class-matched oracle pairing against label-agnostic random pairing, model-guided pairing, and organ-independent late fusion on the exact same test samples. On an image-level split of 46,974 field photographs (41,875 leaf and 5,099 fruit), single-organ Swin-B models achieve 78.45\% accuracy (foliar) and 96.82\% (pericarp). While class-matched oracle pairing attains 98.29\% accuracy and 0.9841 macro-F1, unconstrained label-agnostic pairing causes accuracy to collapse to 19.88\% (matching the 20\% 5-class random chance baseline), with 98.58\% of predictions tracking the random fruit partner's label. This demonstrates extreme modality dominance, where the network treats the fruit stream as an implicit label shortcut. To address this, an organ-independent late-fusion consensus pipeline achieves 97.20\% accuracy and 0.9710 macro-F1 without ground-truth label leakage. The framework is complemented by an empirical surface lesion-burden proxy ($S = A_{\text{lesion}} / A_{\text{organ}} \times 100\%$) and attention-rollout spatial localization across Swin-B transformer stages.
\end{abstract}
```

---

### 3. Revised Contributions (Section I: Softened & Grounded)

```latex
The contributions of this work are as follows:
\begin{enumerate}[(i)]
    \item \textbf{Token-Level Cross-Attention Fusion:} We formulate a multi-head bidirectional cross-attention module ($4$ heads) operating across $K=8$ learned semantic subspace tokens ($X \in \mathbb{R}^{8 \times 128}$) projected from pooled Swin-B representations, replacing the degenerate scalar attention of single pooled vectors with an $8 \times 8$ inter-subspace attention distribution.
    \item \textbf{Decoupled Multimodal Benchmark Protocol:} We formulate and execute a decoupled evaluation protocol for unpaired multi-organ agricultural datasets, benchmarking class-matched oracle pairing against label-agnostic random pairing and model-guided pairing on the exact same test samples.
    \item \textbf{Empirical Demonstration of Label Leakage:} We demonstrate that class-matched multimodal pairing introduces an artificial 78.41 percentage-point performance inflation (collapsing from 98.29\% to 19.88\% under random pairing), with 98.58\% of predictions tracking the partner fruit's label, and we show that late-fusion consensus achieves 97.20\% accuracy without leakage.
    \item \textbf{Selective cGAN & Empirical Lesion Proxy:} We deploy class-conditional GAN augmentation with Binary Cross-Entropy loss for under-represented fruit classes on the training split, coupled with an empirical surface lesion-burden proxy ($S$) mapped to four severity grades.
\end{enumerate}
```

---

### 4. Ground-Truth Dataset Provenance (Section III-A)

```latex
\subsection{Dataset Composition and Aggregation}
The dataset comprises 46,974 images (41,875 leaves and 5,099 fruit) across five target classes: Healthy, Alternaria, Anthracnose, Bacterial Blight, and Cercospora (Table II). Images were aggregated from five open-access agricultural disease repositories hosted on Hugging Face Hub: \textit{Project-AgML/pomegranate\_disease\_classification}, \textit{Ibrahim125/Pomegranate-Fruit-Diseases-Dataset}, \textit{TechMahadev/pomegranatePlantHealth}, \textit{AmitTheGreat/pomegranateFruitDiseas}, and \textit{VedicSeeker/pomegranateDiseaseClassification}. Duplicate detection using MD5 and perceptual-hash (pHash) comparison identified and removed 8 exact byte duplicates. Because public repositories often aggregate overlapping upstream survey archives, near-duplicate images and pre-augmented variations may persist across image-level splits, which represents an inherent limitation of uncurated web-scale agricultural benchmarks discussed further in Section VIII.
```

---

### 5. Ground-Truth Token Cross-Attention Formulation (Section IV-B & IV-C)

```latex
\subsection{Swin Transformer Feature Extraction}
Each organ image is processed by an independent Swin-B backbone \cite{liu2021swin} initialized with ImageNet-1K weights. The initial patch partition operates with embedding dimension $C = 128$, hierarchical window size $M = 7$, and final stage-4 output dimension $D = 1024$. Each encoder produces a pooled global representation $f_l = \text{pool}(\Phi_l(x_l)) \in \mathbb{R}^{1024}$ and $f_f = \text{pool}(\Phi_f(x_f)) \in \mathbb{R}^{1024}$.

To avoid single-vector scalar attention degeneracy while working from pooled backbones, a learned linear projection layer $P(\cdot)$ with LayerNorm and GELU activation maps each 1024-dimensional vector into $K = 8$ semantic subspace tokens of dimension $d_k = 128$ ($K \times d_k = 1024$):
\begin{equation}
    X_l = \text{reshape}(P_l(f_l)) \in \mathbb{R}^{8 \times 128}, \quad X_f = \text{reshape}(P_f(f_f)) \in \mathbb{R}^{8 \times 128}
\end{equation}

\subsection{Multi-Head Cross-Attention Fusion}
Cross-modal interaction is computed across the $K = 8$ learned semantic subspaces using multi-head attention ($h = 4$ heads, head dimension $d_h = 32$):
\begin{equation}
    Q_l = X_l W_Q, \quad K_f = X_f W_K, \quad V_f = X_f W_V
\end{equation}
where $W_Q, W_K, W_V \in \mathbb{R}^{128 \times 128}$. The bidirectional attention matrix is calculated as:
\begin{equation}
    A_{l \to f} = \text{softmax}\left(\frac{Q_l K_f^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{8 \times 8}, \quad Z_{l \to f} = A_{l \to f} V_f
\end{equation}
Because the softmax operates over 8 distinct subspace keys, each row of $A_{l \to f}$ defines a non-trivial probability distribution over semantic feature subspaces, resolving the single-key scalar collapse ($\text{softmax}([s]) \equiv 1.0$) of 1D vector designs. The attended subspace features are combined via residual connections and layer normalization:
\begin{equation}
    z_l = \text{LN}(X_l + Z_{l \to f}), \quad z_f = \text{LN}(X_f + Z_{f \to l})
\end{equation}
The concatenated token representation $z = [z_l \parallel z_f] \in \mathbb{R}^{2048}$ is projected through an MLP classifier to output class logits.
```

---

### 6. Ground-Truth cGAN Specification (Section IV-D)

```latex
\subsection{Selective Class-Conditional GAN Augmentation}
To address minority representation in fruit classes (Alternaria, $N=886$; Cercospora, $N=631$), a class-conditional DCGAN is trained exclusively on the fruit training split. The generator $G$ receives a 100-dimensional Gaussian noise vector $u \sim \mathcal{N}(0, I)$ concatenated with a learned class embedding $c_y \in \mathbb{R}^{2}$, mapped through four transposed convolution blocks with BatchNorm and ReLU to generate $64 \times 64$ RGB images. The discriminator $D$ embeds class labels into a $1 \times 64 \times 64$ spatial feature plane, concatenated channel-wise to the RGB image tensor.

The networks are optimized using Binary Cross-Entropy (BCE) loss with one-sided label smoothing ($\tau = 0.9$ for real targets) and Adam optimizers ($lr = 2 \times 10^{-4}, \beta_1 = 0.5, \beta_2 = 0.999$):
\begin{equation}
    \mathcal{L}_D = -\mathbb{E}_{x, y}[\log D(x, y)] - \mathbb{E}_{u, y}[\log(1 - D(G(u, y), y))]
\end{equation}
Candidate images must pass two quality gates: an empirical variance threshold ($\sigma > 15.0$) to reject mode-collapsed outputs, and a consistency gate requiring baseline classifier confidence $P(y = c) \ge 0.80$. A total of 1,200 synthetic fruit images were generated (augmenting training from 32,879 to 34,079 images, $\approx 3.5\%$ synthetic).
```

---

### 7. Results, Reconciled Metrics, and Modality Dominance Diagnostic (Section VI)

#### Reconciled Table VI:
```latex
\begin{table}[t]
\centering
\caption{Comprehensive Decoupled Evaluation on the Frozen Test Set}
\label{tab:decoupled_results}
\begin{tabular}{llccc}
\hline
\textbf{ID} & \textbf{Configuration} & \textbf{Pairing Protocol} & \textbf{Acc. (\%)} & \textbf{Macro-F1} \\
\hline
E1-L & ResNet-50 Baseline & Single-Organ (Leaf) & 70.03 & 0.7184 \\
E1-F & ResNet-50 Baseline & Single-Organ (Fruit) & 95.92 & 0.9540 \\
E2-L & Swin-B Transformer & Single-Organ (Leaf) & 78.45 & 0.7912 \\
E3-F & Swin-B Transformer & Single-Organ (Fruit) & 96.82 & 0.9654 \\
\hline
\multicolumn{5}{l}{\textit{Multimodal Fusion under Class-Matched (Oracle) Upper-Bound:}} \\
E4 & Dual Swin-B Concat & Oracle (Class-Matched) & 94.20 & 0.9410 \\
E5 & E4 + Naive GAN & Oracle (Class-Matched) & 95.85 & 0.9570 \\
E6 & Dual Swin-B Token CA & Oracle (Class-Matched) & 96.95 & 0.9680 \\
E7 & E6 + Selective GAN & Oracle (Class-Matched) & \textbf{98.29} & \textbf{0.9841} \\
\hline
\multicolumn{5}{l}{\textit{Decoupled Unconfounded Evaluation on Same Test Samples:}} \\
E7-R & Proposed Framework & Label-Agnostic (Random) & \textbf{19.88} & \textbf{0.1957} \\
E7-G & Proposed Framework & Model-Guided Pairing & 19.64 & 0.1962 \\
E8 & Dual-Organ Late Fusion & Decision Consensus & \textbf{97.20} & \textbf{0.9710} \\
\hline
\end{tabular}
\end{table}
```

#### Reconciled Table VIII Deltas:
- $E4 \to E7$: $+4.09$ Acc, $+0.0431$ F1
- $E6 \to E7$: $+1.34$ Acc, $+0.0161$ F1
- $E7 \to E7\text{-}R$: $-78.41$ Acc, $-0.7884$ F1
- $E7 \to E8$: $-1.09$ Acc, $-0.0131$ F1
- $E3\text{-}F \to E8$: $+0.38$ Acc, $+0.0056$ F1

#### Diagnostic Evidence Paragraph for Section VI:
> "To understand why random pairing (E7-R) collapses to 19.88\% (below leaf-only 78.45\% and matching 5-class chance), we performed a diagnostic tracking prediction agreement with the random partner fruit. Remarkably, **98.58\% of E7-R predictions matched the random partner fruit's label**, while matching the true leaf label only 19.88\% of the time. This provides definitive empirical evidence of modality dominance: because fruit features are overwhelmingly discriminative (96.82\%), the cross-attention network learned to route predictions almost entirely through the fruit branch. Under random pairing, the model simply classifies the random fruit, collapsing foliar diagnostic utility. Model-guided pairing (E7-G) yielded 19.64\% in this configuration due to cumulative error propagation. In contrast, the organ-independent late-fusion consensus (E8) combines calibrated single-organ predictions without feature cross-talk, achieving 97.20\% accuracy and 0.9710 macro-F1."

---

### 8. Quick Diagram & LaTeX Typo Fixes
1. **Fig. 3**: Relabel top-left box from $W_O$ to **$W_Q$**.
2. **Fig. 2**: Relabel top-right fruit box from $X_r$ to **$X_f$**.
3. **Table I**: Add `\begin{table}[t]` to keep all rows on Page 2.
4. **Table III**: Remove the accidental side-by-side duplicate header row.
5. **Eq. (4)**: Typeset text as `z = [z_l \parallel z_f] \in \mathbb{R}^{2048}`.
