import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.helpers import log_to_file
from config import config
from itertools import cycle


def evaluate(encoder, classifier, val_loader):
    encoder.eval()
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            feats = encoder(images)
            feats = feats.view(feats.size(0), -1)
            logits = classifier(feats)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / max(total, 1)


def semi_supervised_pretrain(
    encoder,
    classifier,
    labeled_loader,
    unlabeled_loader,
    optimizer,
    criterion,
    val_loader=None,
    epochs=50,
    lambda_u=1.0,
    confidence_threshold=0.9,
):
    log_file = "semi_supervised_pretrain_log.txt"
    os.makedirs(config["output_dir"], exist_ok=True)

    best_val_acc = 0.0
    best_path = os.path.join(config["output_dir"], "semi_supervised_best.pth")

    unlabeled_iter = cycle(unlabeled_loader)

    for epoch in range(epochs):
        encoder.train()
        classifier.train()

        sup_loss_sum = 0.0
        unsup_loss_sum = 0.0
        steps = 0

        for labeled_batch in tqdm(
            labeled_loader,
            desc=f"Semi-Supervised Epoch {epoch + 1}/{epochs}",
        ):
            labeled_images, labeled_targets = labeled_batch
            labeled_images = labeled_images.cuda()
            labeled_targets = labeled_targets.cuda()

            # ----------------- supervised -----------------
            feats_l = encoder(labeled_images)
            feats_l = feats_l.view(feats_l.size(0), -1)
            logits_l = classifier(feats_l)
            supervised_loss = criterion(logits_l, labeled_targets)

            # ----------------- get one unlabeled batch -----------------
            unlabeled_images, _ = next(unlabeled_iter)
            unlabeled_images = unlabeled_images.cuda()

            with torch.no_grad():
                feats_u = encoder(unlabeled_images)
                feats_u = feats_u.view(feats_u.size(0), -1)
                logits_u = classifier(feats_u)
                probs_u = torch.softmax(logits_u, dim=1)
                max_probs, pseudo_labels = torch.max(probs_u, dim=1)
                mask = max_probs >= confidence_threshold

            if mask.sum() > 0:
                feats_sel = feats_u[mask]
                pseudo_sel = pseudo_labels[mask]

                logits_sel = classifier(feats_sel)
                unsupervised_loss = criterion(logits_sel, pseudo_sel)
            else:
                unsupervised_loss = torch.tensor(0.0, device=labeled_images.device)

            # ----------------- combine losses -----------------
            total_loss = supervised_loss + lambda_u * unsupervised_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            sup_loss_sum += supervised_loss.item()
            unsup_loss_sum += unsupervised_loss.item()
            steps += 1

        # -------- logging --------
        msg = (
            f"Epoch {epoch+1}, Sup Loss: {sup_loss_sum/steps:.4f}, "
            f"Unsup Loss: {unsup_loss_sum/steps:.4f}"
        )
        print(msg)
        log_to_file(config["output_dir"], log_file, msg)

        # -------- validation & checkpoint --------
        if val_loader is not None:
            val_acc = evaluate(encoder, classifier, val_loader)
            msg2 = f"Epoch {epoch+1}, Val Acc: {val_acc:.2f}%"
            print(msg2)
            log_to_file(config["output_dir"], log_file, msg2)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "classifier": classifier.state_dict(),
                        "epoch": epoch + 1,
                        "val_acc": best_val_acc,
                    },
                    best_path,
                )
                print(f"Saved new best model at epoch {epoch+1}: {best_val_acc:.2f}%")
