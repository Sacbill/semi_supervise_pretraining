Semi-Supervised and Self-Supervised Pretraining (Dogs vs. Cats)

This repository contains code and experiments for benchmarking self-supervised (MAE) and semi-supervised pretraining methods under limited labeled data, evaluated using a linear probe protocol on the Dogs vs. Cats image classification task.

Dataset

The dataset is the Dogs vs. Cats dataset from Kaggle, organized in a PyTorch ImageFolder format.

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
Classes: cat, dog

Task: Binary image classification

Image size: Resized to 256×256

Labels: Inferred from folder names

Splits: Train / Validation / Test

Environment

conda env create -f environment.yml
conda activate introcnn
