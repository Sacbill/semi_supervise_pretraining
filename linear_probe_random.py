import os
import torch
import torch.nn as nn
import torch.optim as optim

from config import config
from models.encoder import ResNetEncoder
from models.classification_head import ClassificationHead
from utils.dataset import get_dataloader
from utils.helpers import log_to_file

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # 1. RANDOM encoder (NO pretraining)
    # -------------------------
    # If you want reproducible random features, set the seed BEFORE init:
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)

    encoder = ResNetEncoder().to(device)
    print("Using RANDOMLY INITIALIZED encoder (no pretraining).")

    # Freeze encoder (random linear probe)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()   # just for safety; we don't train it

    # -------------------------
    # 2. Define linear classification head
    # -------------------------
    classifier = ClassificationHead(
        embed_dim=config["embed_dim"],   # e.g. 2048
        num_classes=config["num_classes"]
    ).to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # 3. Dataloaders (NO masking, pretrain=False)
    # -------------------------
    train_loader = get_dataloader(
        split="train", pretrain=False, batch_size=config["batch_size"]
    )
    val_loader = get_dataloader(
        split="val", pretrain=False, batch_size=config["batch_size"]
    )

    num_epochs = config.get("epochs_finetune", 20)

    # -------------------------
    # 4. Linear probe training loop
    # -------------------------
    best_val_acc = 0.0
    save_dir = os.path.join(config["output_dir"], "linear_probe_random")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        classifier.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Encoder is frozen → use no_grad
            with torch.no_grad():
                features = encoder(images)          # shape (B, embed_dim)

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

        # -------------------------
        # Validation
        # -------------------------
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

        print(
            f"[RANDOM] Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        log_message = (
            f"[RANDOM] Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )
        log_to_file(save_dir, "linear_probe_random_log.txt", log_message)

        # ------------- SAVE BEST HEAD -------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                classifier.state_dict(),
                os.path.join(save_dir, "best_linear_probe_random.pth")
            )
            log_to_file(
                save_dir,
                "linear_probe_random_log.txt",
                f"BEST UPDATED at Epoch {epoch+1} | Val Acc = {val_acc:.2f}%"
            )

    # After training ends
    torch.save(
        classifier.state_dict(),
        os.path.join(save_dir, "final_linear_probe_random.pth")
    )
    print(f"Saved final RANDOM linear probe model to {save_dir}")

if __name__ == "__main__":
    main()
