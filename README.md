# Sparse Kernel Cultural Classification Framework

This repository contains the implementation and experimental framework for an efficient image classification system designed for Vietnamese Intangible Cultural Heritage (ICH) recognition.

The system focuses on improving classifier head design on top of frozen CNN feature extractors using sparse kernel Kolmogorov–Arnold inspired architectures and bagging ensembles.

---

## Overview

Automatic classification of intangible cultural heritage images requires models that:

- Capture subtle visual and cultural cues  
- Remain computationally efficient at inference time  
- Integrate easily into deployable systems  

This project introduces a sparse kernel framework that attaches specialized classifier heads to frozen convolutional backbones.

Two main components are implemented:

- **Sparse Kernel Cultural Classifier (SKCC)**
- **Bagged SKCC Ensemble (B-SKCC)**

These models operate on deep feature representations extracted from pretrained CNN backbones.

The framework includes theoretical analysis of:

- Approximation capability
- Computational complexity
- Ensemble error behavior

and is evaluated on a curated Vietnamese ICH dataset.

:contentReference[oaicite:0]{index=0}

---

## Key Contributions

- Sparse kernel classifier head design using compactly supported radial kernels  
- Integration of KAN-style functional edge mappings on CNN features  
- Bootstrap aggregation of classifier heads for robustness  
- Comparative evaluation against FC and dense KAN baselines  
- Deployment-ready pipeline and web interface support  

:contentReference[oaicite:1]{index=1}

---

## Method Summary

### Feature Extraction

Images are processed through a pretrained CNN backbone:

- ConvNeXt  
- ResNet  
- DenseNet  
- EfficientNet  

The backbone is **kept frozen** and produces feature vectors:

