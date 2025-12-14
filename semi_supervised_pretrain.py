import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from config import config
from models.encoder import ResNetEncoder
from models.classification_head import ClassificationHead
from utils.dataset import get_dataloader
from train.semi_supervised_pretrain import semi_supervised_pretrain


def create_labeled_subset_loader(full_labeled_loader, label_fraction, batch_size):
    """
    Create a DataLoader that only uses a given fraction of the labeled dataset.
    """
    assert 0.0 < label_fraction <= 1.0, "label_fraction must be in (0, 1]."

    dataset = full_labeled_loader.dataset
    num_total = len(dataset)
    num_labeled = max(1, int(label_fraction * num_total))

    # Random subset of indices
    indices = torch.randperm(num_total)[:num_labeled].tolist()
    subset = Subset(dataset, indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=getattr(full_labeled_loader, "num_workers", 4),
        pin_memory=getattr(full_labeled_loader, "pin_memory", False),
    )


def run_one_fraction(label_fraction, base_output_dir):
    """
    Run one semi-supervised pretraining experiment for a given label_fraction.
    Everything (logs/checkpoints) goes into its own subfolder.
    """
    # ---------- set a dedicated output dir for this run ----------
    run_name = f"semi_label_{int(label_fraction * 100)}"
    run_output_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    # temporarily override config["output_dir"] for this run
    old_output_dir = config["output_dir"]
    config["output_dir"] = run_output_dir

    print(f"\n===== Running semi-supervised pretraining with "
          f"{label_fraction * 100:.1f}% labels =====")
    print(f"Output dir: {run_output_dir}")

    # ---------- models ----------
    encoder = ResNetEncoder().cuda()
    classifier = ClassificationHead(
        embed_dim=config["embed_dim"],
        num_classes=config["num_classes"],
    ).cuda()

    # ---------- dataloaders ----------
    # full labeled loader (we will subsample from this)
    full_labeled_loader = get_dataloader(
        split="train",
        pretrain=False,
        batch_size=config["batch_size"],
    )

    labeled_loader = create_labeled_subset_loader(
        full_labeled_loader,
        label_fraction=label_fraction,
        batch_size=config["batch_size"],
    )

    # unlabeled loader: treat all train images as unlabeled
    unlabeled_loader = get_dataloader(
        split="train",
        pretrain=True,  # your code should ignore labels / use different transforms
        batch_size=config["batch_size"],
    )

    # validation loader
    val_loader = get_dataloader(
        split="val",
        pretrain=False,
        batch_size=config["batch_size"],
    )

    print(f"Using {len(labeled_loader.dataset)} labeled samples "
          f"out of {len(full_labeled_loader.dataset)} total.")

    # ---------- optimizer & loss ----------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=config["learning_rate"],
    )

    # ---------- train ----------
    semi_supervised_pretrain(
        encoder=encoder,
        classifier=classifier,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        optimizer=optimizer,
        criterion=criterion,
        val_loader=val_loader,
        epochs=config["epochs_pretrain"],
        lambda_u=1.0,
        confidence_threshold=0.9,
    )

    # restore config["output_dir"] so other parts of your code stay sane
    config["output_dir"] = old_output_dir


def main():
    # base dir where all semi-supervised runs will live
    base_output_dir = os.path.join(
        config["output_dir"], "semi_supervised_runs"
    )
    os.makedirs(base_output_dir, exist_ok=True)

    # three cases: 1%, 10%, 50% labels
    label_fractions = [0.01, 0.10, 0.50]

    for frac in label_fractions:
        run_one_fraction(frac, base_output_dir)


if __name__ == "__main__":
    main()
