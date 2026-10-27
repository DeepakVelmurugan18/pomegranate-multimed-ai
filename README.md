---
title: Pomegranate-MultiMed AI
emoji: 🌿
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Multimodal Transformer & XAI Pomegranate Disease Diagnosis
---

# Pomegranate-MultiMed AI 🌿🍎

A **Dual-Organ Vision Framework for Pomegranate Pathology: Protocol Decoupling, Token Cross-Attention, and Lesion-Burden Proxy Analysis** (*Punica granatum L.*).

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Key Capabilities

- **Organ Modality Switcher & Triage**:
  - 🌿 **Dual-Organ Mode**: Token-level cross-attention fusion ($K=8$ patch tokens) and late-fusion consensus.
  - 🍃 **Leaf Only**: Foliar pathology diagnosis using trained leaf backbones.
  - 🍎 **Fruit Only**: Pericarp pathology diagnosis using trained fruit backbones.
- **Leakage-Free Decoupled Evaluation**: Explicit benchmark comparing class-matched oracle upper bounds against unconfounded label-agnostic pairings.
- **5-Class Disease Taxonomy**:
  - 🟢 Healthy (*Punica granatum L.*)
  - 🍂 Alternaria (*Alternaria alternata*)
  - 🍁 Anthracnose (*Colletotrichum gloeosporioides*)
  - ⚠️ Bacterial Blight (*Xanthomonas axonopodis pv. punicae*)
  - 🔴 Cercospora (*Cercospora punicae*)
- **Objective Lesion-Burden Severity Proxy**: Quantitative surface lesion ratio proxy ($S < 10\%$ Low, $10-25\%$ Moderate, $>25\%$ Severe) calibrated to phytosanitary literature.
- **Attribution & Pathology Heatmaps**: ResNet-50 Grad-CAM localization highlighting lesion perimeters and pericarp cracking margins.
- **Phytosanitary PDF Diagnostic Report**: Official laboratory certificate following ICAR/NRCP reporting standards with agronomic prescriptions.
