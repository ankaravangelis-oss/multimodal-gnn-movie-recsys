# Multimodal Graph Neural Networks for Movie Recommendation

## Overview
An end-to-end **Hybrid Multimodal Recommendation System** addressing item cold-start and interaction sparsity by combining Computer Vision (CV), Natural Language Processing (NLP), and Graph Neural Networks (GNNs).

---

## Architecture Pipeline
1. **Computer Vision (CV):** Multi-label genre classification on movie posters via a custom CNN benchmarked against a fine-tuned **ResNet-50** backbone. Dense visual representations are extracted from penultimate feature layers.
2. **Natural Language Processing (NLP):** Multi-label genre classification and contextual semantic extraction from plot summaries using **Transformer-based sequence modeling**.
3. **Graph Neural Networks (GNN):** Bipartite user-movie interaction graph (`ml-latest-small`) constructed via **PyTorch Geometric**. Item nodes are initialized with fused multimodal (Visual + Textual) embeddings, followed by a **Hetero-GraphSAGE** architecture performing user-item link prediction.

---

## Data Pipeline & Leakage Prevention
- **Disjoint Training Pool:** Pre-training of CV and NLP encoders is executed strictly on a 15,000-movie subset sourced from `ml-25m`, completely disjoint from the target evaluation graph to eliminate target data leakage.
- **Stratified Distribution Alignment:** Enforced `Decade × Primary Genre` joint distribution matching between the pre-training set and the evaluation graph to prevent domain/distribution shift.

---

## Tech Stack
- **Frameworks:** PyTorch, PyTorch Geometric, Scikit-Learn, Pandas, NumPy
- **Models:** ResNet-50, Transformers, Hetero-GraphSAGE
- **Data:** GroupLens MovieLens (ml-25m, ml-latest-small)
