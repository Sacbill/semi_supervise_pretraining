# train/fine_tune.py

import torch
from tqdm import tqdm
from utils.helpers import calculate_accuracy, log_to_file
from config import config
import os

def fine_tune(encoder, classifier, train_loader, val_loader, optimizer, criterion, epochs=5):
    best_val_accuracy = 0.0  # Track the best validation accuracy
    best_epoch = 0           # Track the best epoch number
    log_file = "finetune_log.txt"  # Log file name

    for epoch in range(epochs):
        encoder.train()
        classifier.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        
        # Training loop
        for images, labels in tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch + 1}/{epochs}"):
            images, labels = images.cuda(), labels.cuda()

            # Forward pass
            features = encoder(images)
            outputs = classifier(features)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels) * labels.size(0)
            total_samples += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / total_samples * 100
        train_log_message = f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}, Training Accuracy: {avg_accuracy:.2f}%"
        print(train_log_message)
        
        # Log training metrics
        log_to_file(config["output_dir"], log_file, train_log_message)

        # Validation loop
        encoder.eval()
        classifier.eval()
        val_total_accuracy = 0.0
        val_total_samples = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                features = encoder(images)
                outputs = classifier(features)
                val_total_accuracy += calculate_accuracy(outputs, labels) * labels.size(0)
                val_total_samples += labels.size(0)

        val_accuracy = val_total_accuracy / val_total_samples * 100
        val_log_message = f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.2f}%"
        print(val_log_message)
        
        # Log validation metrics
        log_to_file(config["output_dir"], log_file, val_log_message)

        # Save the model if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_model_path = os.path.join(config["output_dir"], "best_model.pth")
            torch.save({
                "encoder": encoder.state_dict(),
                "classifier": classifier.state_dict(),
                "epoch": best_epoch,
                "val_accuracy": best_val_accuracy
            }, best_model_path)
            print(f"New best model saved at epoch {best_epoch} with accuracy: {best_val_accuracy:.2f}%")

    print(f"Training complete. Best model at epoch {best_epoch} with validation accuracy: {best_val_accuracy:.2f}%")
