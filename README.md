# Semi-Supervised and Self-Supervised Pretraining under Limited Labels

This repository contains code and experiments for benchmarking **self-supervised (MAE)** and **semi-supervised pretraining** methods under limited labeled data. All methods are evaluated using a **linear probe protocol** on the **Dogs vs. Cats** image classification task.

---

## Dataset

The dataset used in this project is the **Dogs vs. Cats** dataset from **Kaggle**, organized in a format compatible with PyTorch’s `ImageFolder`.

```text
datasets/
├── train/
│   ├── cat/
│   └── dog/
├── val/
│   ├── cat/
│   └── dog/
└── test/
    ├── cat/
    └── dog/
```
```bash
conda env create -f environment.yml
conda activate introcnn
```
