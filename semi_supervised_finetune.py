import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import config
from models.encoder import ResNetEncoder
from models.classification_head import ClassificationHead
from utils.dataset import get_dataloader
from utils.helpers import log_to_file


def create_labeled_subset_loader(full_labeled_loader, label_fraction, batch_size):
    """
    Create a DataLoader that only uses a given fraction of the labeled dataset.
    """
    assert 0.0 < label_fraction <= 1.0, "label_fraction must be in (0, 1]."

    dataset = full_labeled_loader.dataset
    num_total = len(dataset)
    num_labeled = max(1, int(label_fraction * num_total))

    indices = torch.randperm(num_total)[:num_labeled].tolist()
    subset = Subset(dataset, indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=getattr(full_labeled_loader, "num_workers", 4),
        pin_memory=getattr(full_labeled_loader, "pin_memory", False),
    )


def linear_probe_one_semi_model(run_dir, label_fraction, device):
    """
    For a given semi-supervised run directory:
      - load encoder from semi_supervised_best.pth
      - freeze encoder
      - train a new classifier head by linear probing
    """
    ckpt_path = os.path.join(run_dir, "semi_supervised_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[{run_dir}] semi_supervised_best.pth not found, skipping.")
        return

    print(f"\n=== Linear probing semi-supervised model in {run_dir} "
          f"with {label_fraction*100:.1f}% labels ===")

    # ----- Load encoder from checkpoint -----
    checkpoint = torch.load(ckpt_path, map_location=device)

    encoder = ResNetEncoder().to(device)
    encoder.load_state_dict(checkpoint["encoder"])

    # freeze encoder for linear probing
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # ----- New classifier head -----
    classifier = ClassificationHead(
        embed_dim=config["embed_dim"],
        num_classes=config["num_classes"],
    ).to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # ----- Dataloaders -----
    full_train_loader = get_dataloader(
        split="train",
        pretrain=False,
        batch_size=config["batch_size"],
    )
    val_loader = get_dataloader(
        split="val",
        pretrain=False,
        batch_size=config["batch_size"],
    )
    '''
    train_loader = create_labeled_subset_loader(
        full_train_loader,
        label_fraction=label_fraction,
        batch_size=config["batch_size"],
    )
    '''
    train_loader = full_train_loader

    print(f"Linear probe uses {len(train_loader.dataset)} labeled samples "
          f"out of {len(full_train_loader.dataset)} total.")

    # ----- Logging / save dir -----
    lp_dir = os.path.join(run_dir, "linear_probe")
    os.makedirs(lp_dir, exist_ok=True)
    log_file = "linear_probe_log.txt"

    num_epochs = config.get("epochs_finetune", 20)  # or a dedicated config if you want
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        classifier.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # -------- train --------
        for images, labels in tqdm(
            train_loader,
            desc=f"[Linear Probe] Epoch {epoch+1}/{num_epochs}"
        ):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                features = encoder(images)

            outputs = classifier(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = train_loss / total
        train_acc = 100.0 * correct / total

        # -------- val --------
        classifier.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                features = encoder(images)
                outputs = classifier(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / total
        val_acc = 100.0 * correct / total

        msg = (
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )
        print(msg)
        log_to_file(lp_dir, log_file, msg)

        # save best classifier
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(lp_dir, "best_linear_probe.pth")
            torch.save(classifier.state_dict(), best_path)
            log_to_file(
                lp_dir,
                log_file,
                f"BEST UPDATED at Epoch {epoch+1} | Val Acc = {val_acc:.2f}%",
            )

    # final save
    final_path = os.path.join(lp_dir, "final_linear_probe.pth")
    torch.save(classifier.state_dict(), final_path)
    print(f"[{run_dir}] Linear probe done. Best Val Acc = {best_val_acc:.2f}%")
    print(f"Final classifier saved to {final_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    base_output_dir = config["output_dir"]
    semi_base = os.path.join(base_output_dir, "semi_supervised_runs")

    # the three semi-supervised label regimes you trained
    label_fractions = [0.01, 0.10, 0.50]  # 1%, 10%, 50%
    run_names = [f"semi_label_{int(frac * 100)}" for frac in label_fractions]

    for frac, name in zip(label_fractions, run_names):
        run_dir = os.path.join(semi_base, name)
        if not os.path.isdir(run_dir):
            print(f"Directory {run_dir} not found, skipping.")
            continue

        linear_probe_one_semi_model(run_dir, frac, device)


if __name__ == "__main__":
    main()
