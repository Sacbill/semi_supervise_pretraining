import torch
from tqdm import tqdm
from utils.helpers import mask_image, log_to_file
from config import config
import os

def pretrain_mae(encoder, decoder, dataloader, optimizer, criterion, epochs=5, mask_ratio=0.75):
    encoder.train()
    decoder.train()
    
    log_file = "pretrain_log.txt"  # Log file name
    best_loss = float("inf")  # Initialize best_loss to a very high value
    save_statement = 'No improvement in loss during training.'  # Default statement if no improvement

    for epoch in range(epochs):
        total_loss = 0.0
        for images, _ in tqdm(dataloader, desc=f"Pretraining Epoch {epoch + 1}/{epochs}"):
            images = images.cuda()
            masked_images = torch.stack([mask_image(img, mask_ratio) for img in images]).cuda()

            # Forward pass
            encoded_features = encoder(masked_images).view(images.size(0), -1)
            #print(encoded_features.shape)
            reconstructed_images = decoder(encoded_features)

            # Calculate loss
            loss = criterion(reconstructed_images, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        log_message = f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}"
        print(log_message)
        
        # Log to file
        log_to_file(config["output_dir"], log_file, log_message)

        # Save checkpoint if the current average loss is the lowest recorded
        if avg_loss < best_loss:
            best_loss = avg_loss  # Update best_loss with the new lowest loss
            checkpoint_path = os.path.join(config["output_dir"], "lowest_loss_model.pth")
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "epoch": epoch + 1,
                "loss": best_loss
            }, checkpoint_path)
            save_statement = f"Model with lowest loss saved at {checkpoint_path}"
            print(save_statement)

        # Save periodic checkpoints every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(config["output_dir"], f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_loss
            }, checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    # Print and log final save statement
    print(save_statement)
    log_to_file(config["output_dir"], log_file, save_statement)
