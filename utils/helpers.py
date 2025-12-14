# utils/helpers.py

import numpy as np
import torch
import random
import os

def log_to_file(output_dir, filename, message):
    """Logs a message to a specified file in the output directory."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    log_path = os.path.join(output_dir, filename)
    
    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")

def mask_image(image, mask_ratio=0.75, patch_size=16):
    """Masks patches in the image by replacing them with zeros.

    Args:
        image (Tensor): Input image tensor of shape (C, H, W).
        mask_ratio (float): Ratio of patches to mask.
        patch_size (int): Size of each patch.

    Returns:
        Tensor: Masked image.
    """
    h, w = image.shape[1] // patch_size, image.shape[2] // patch_size
    num_patches = h * w
    num_masked = int(mask_ratio * num_patches)

    mask = np.ones((h, w), dtype=bool)
    mask.flat[random.sample(range(num_patches), num_masked)] = False

    masked_image = image.clone()
    for i in range(h):
        for j in range(w):
            if not mask[i, j]:
                masked_image[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = 0
    return masked_image

def calculate_accuracy(outputs, labels):
    """Calculates classification accuracy."""
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels).sum().item()
    return correct / labels.size(0)
