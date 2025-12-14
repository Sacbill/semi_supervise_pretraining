# models/decoder.py

import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, embed_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=4, stride=4),             # Upsample to [512, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),        # Upsample to [256, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),        # Upsample to [128, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),         # Upsample to [64, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),          # Upsample to [32, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),          # Upsample to [16, 128, 128]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)            # Final upsample to [3, 224, 224]
        )

    def forward(self, x):
        # Reshape input to start upsampling from [batch_size, embed_dim, 1, 1]
        x = x.view(x.size(0), -1, 1, 1)
        return self.decoder(x)
