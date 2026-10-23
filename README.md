---
title: Pomegranate-MultiMed AI
emoji: 🌿
colorFrom: green
colorTo: red
sdk: docker
app_port: 8080
pinned: false
license: mit
short_description: Multimodal Transformer & XAI Pomegranate Disease Diagnosis
---

# Pomegranate-MultiMed AI 🌿🍎

An **Explainable Multimodal Transformer Framework with Selective GAN Augmentation for Pomegranate Disease Detection and Severity Assessment** (*Punica granatum L.*).

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Key Capabilities

- **Organ Modality Switcher**:
  - 🌿 **Dual Multimodal (Aligned Foliage + Pericarp)**: Cross-attention fusion combining leaf and fruit cues.
  - 🍃 **Leaf Only**: Foliar pathology diagnosis using trained leaf backbones.
  - 🍎 **Fruit Only**: Pericarp pathology diagnosis using trained fruit backbones.
- **5-Class Disease Taxonomy**:
  - 🟢 Healthy (*Punica granatum L.*)
  - 🍂 Alternaria (*Alternaria alternata*)
  - 🍁 Anthracnose (*Colletotrichum gloeosporioides*)
  - ⚠️ Bacterial Blight (*Xanthomonas axonopodis pv. punicae*)
  - 🔴 Cercospora (*Cercospora punicae*)
- **Level 3 Lesion-Burden Severity Grading**: Quantitative surface lesion proxy ($S < 10\%$ Low, $10-25\%$ Moderate, $>25\%$ Severe).
- **Explainable AI (XAI)**: Dual-branch attention attribution heatmaps highlighting necrotic margins.
- **Phytosanitary PDF Diagnostic Report**: Official laboratory certificate with ICAR/NRCP reporting standards, candidate probability breakdown, agronomic prescriptions, and one-click PDF generation.

---

## 🚀 Live Cloud Deployment via Docker

This space is built directly using the included [`Dockerfile`](Dockerfile):
```dockerfile
FROM python:3.10-slim
...
EXPOSE 8080
CMD ["python", "app/server.py"]
```

### Local Testing
```bash
git clone https://huggingface.co/spaces/<username>/pomegranate-multimed-ai
cd pomegranate-multimed-ai
python -m pip install -r requirements.txt
python app/server.py 8080
```
Visit `http://localhost:8080`.
