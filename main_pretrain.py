import os
from config import config
from train.pretrain_mae import pretrain_mae
from models.encoder import ResNetEncoder
from models.decoder import Decoder
from utils.dataset import get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    # Initialize encoder and decoder for pretraining
    encoder = ResNetEncoder().cuda()
    decoder = Decoder(embed_dim=config["embed_dim"]).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config["learning_rate"])

    # Load all data for pretraining
    pretrain_loader = get_dataloader(split="train", pretrain=True, batch_size=config["batch_size"])

    # Run the MAE-style pretraining
    pretrain_mae(encoder, decoder, pretrain_loader, optimizer, criterion, epochs=config["epochs_pretrain"], mask_ratio=config["mask_ratio"])

if __name__ == "__main__":
    main()
