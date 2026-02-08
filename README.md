# Sparse Kernel Cultural Image Classification

This repository implements an efficient image classification framework for Vietnamese Intangible Cultural Heritage (ICH) recognition using sparse kernel classifier heads on top of frozen CNN features.

The focus of this project is improving the **classifier head design**, rather than modifying deep backbone networks.

---

## What this project does

- Extract features using pretrained CNN backbones
- Train sparse kernel classifier heads on frozen features
- Build ensemble models for improved robustness
- Evaluate accuracy and runtime efficiency
- Support deployment in a web classification pipeline

The approach targets practical use in cultural heritage documentation, education, and digital preservation.

---

## Core Idea

Typical pipelines use fully connected layers as classifier heads.  
This project replaces them with kernel-based functional mappings inspired by Kolmogorov–Arnold style networks.

Two models are provided:

### SKCC — Sparse Kernel Cultural Classifier
- Uses compact radial kernels instead of dense weights
- Produces sparse activations
- Improves efficiency and local modeling

### B-SKCC — Bagged SKCC
- Trains multiple SKCC models on bootstrap samples
- Combines predictions using majority vote
- Improves robustness and accuracy

---

## Pipeline

1. Resize and normalize images  
2. Extract feature vectors using frozen CNN backbone  
3. Train SKCC or ensemble heads on feature dataset  
4. Predict classes from feature vectors  
5. Deploy model via inference pipeline or web interface  

---

## Dataset

- 10,143 images  
- 11 Vietnamese cultural heritage categories  
- Includes festivals, crafts, performing arts, rituals  
- Stratified train/validation/test split  

Images were curated from real-world capture and public cultural sources and annotated by domain-aware reviewers.

---

## Training Setup

- PyTorch implementation  
- Frozen ImageNet pretrained backbones  
- AdamW optimizer  
- Cross-entropy loss  
- Shared hyperparameters across models  

Metrics used:

- Accuracy  
- Precision  
- Recall  
- F1 Score  

---

## Results Summary

- Sparse kernel heads outperform standard FC heads  
- Bagging improves further  
- Runtime reduced vs dense kernel heads  
- Best models reach ~96% accuracy  

Moderate ensemble size gives most performance gains without large latency cost.

---

## Deployment

The framework supports:

- Single or batch image classification  
- Consistent preprocessing between training and inference  
- Web interface integration  
- Category-level output organization  


