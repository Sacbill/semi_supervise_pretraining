import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from config import config
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset, Dataset
from sklearn.decomposition import PCA
import numpy as np

class PCATransform:
    def __init__(self, n_components=3, variance=0.1):
        self.n_components = n_components
        self.variance = variance

    def __call__(self, image):
        # Flatten the image to shape (32*32, 3) for PCA on RGB channels
        flat_image = image.view(-1, 3).numpy()
        
        # Apply PCA
        pca = PCA(n_components=self.n_components)
        pca.fit(flat_image)
        
        # Perturb the principal components
        components_variation = np.random.normal(0, self.variance, self.n_components)
        transformed_image = pca.inverse_transform(pca.transform(flat_image) + components_variation)
        
        # Reshape back to (32, 32, 3) and convert to tensor
        transformed_image = torch.tensor(transformed_image).view(3, 32, 32)
        return transformed_image

def get_cifar10_dataloader_pca(split="train", pretrain=False, batch_size=32, shuffle=True):
    """Returns a DataLoader for the CIFAR-10 dataset with appropriate transformations.
    
    Args:
        split (str): Which split to load ('train', 'val', 'test', or 'all').
        pretrain (bool): Whether to apply pretraining transformations (for MAE-style training).
        batch_size (int): Batch size for loading data.
        shuffle (bool): Whether to shuffle the data.
    
    Returns:
        DataLoader: A DataLoader for the CIFAR-10 dataset.
    """
    # Define CIFAR-10 dataset root path
    root_dir = config["root_dir"]
    
    # Define transformations based on the training mode
    if pretrain:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32),  # CIFAR-10 images are 32x32
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
            PCATransform(n_components=3, variance=0.1),  # Add PCA augmentation
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
            PCATransform(n_components=3, variance=0.1),  # Add PCA augmentation
        ])

    if split == "all":
        # Load the entire CIFAR-10 training dataset for combined pretraining
        dataset = datasets.CIFAR10(root=root_dir, train=True, transform=transform, download=True)
    elif split in ["train", "val"]:
        # Load the full CIFAR-10 training set and split it into training and validation
        full_train_dataset = datasets.CIFAR10(root=root_dir, train=True, transform=transform, download=True)
        train_size = int(0.8 * len(full_train_dataset))  # 80% training, 20% validation
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        
        # Select the correct split
        dataset = train_dataset if split == "train" else val_dataset
        shuffle = shuffle if split == "train" else False  # Disable shuffle for validation
    else:
        # Load the CIFAR-10 test dataset
        dataset = datasets.CIFAR10(root=root_dir, train=False, transform=transform, download=True)
        shuffle = False  # No shuffle for test set

    # Return DataLoader for the specified split
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

def get_cifar10_dataloader_aug(split="train", pretrain=False, batch_size=32, shuffle=True):
    """Returns a DataLoader for the CIFAR-10 dataset with appropriate transformations.
    
    Args:
        split (str): Which split to load ('train', 'val', or 'test').
        pretrain (bool): Whether to apply pretraining transformations (for MAE-style training).
        batch_size (int): Batch size for loading data.
        shuffle (bool): Whether to shuffle the data.
    
    Returns:
        DataLoader: A DataLoader for the CIFAR-10 dataset.
    """
    # Define CIFAR-10 dataset root path
    root_dir = config["root_dir"]
    
    # Define transformations based on the training mode
    if pretrain:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32),  # CIFAR-10 images are 32x32
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ])
    else:
        if split == "train":
            transform = transforms.Compose([
                transforms.Resize(36),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
            ])
        else:  # Validation and test transformations
            transform = transforms.Compose([
                transforms.Resize(36),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
            ])
    
    # Dataset selection based on the split
    if split == "all":
        dataset = datasets.CIFAR10(root=root_dir, train=True, transform=transform, download=True)
    elif split in ["train", "val"]:
        # Load the full training dataset
        full_train_dataset = datasets.CIFAR10(root=root_dir, train=True, transform=transform, download=True)
        train_size = int(0.8 * len(full_train_dataset))  # 80% for training
        val_size = len(full_train_dataset) - train_size  # 20% for validation
        
        # Split into training and validation subsets
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        
        # Select the appropriate dataset
        dataset = train_dataset if split == "train" else val_dataset
    else:
        # Load the test dataset
        dataset = datasets.CIFAR10(root=root_dir, train=False, transform=transform, download=True)

    # Return DataLoader for the specified split
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)


def get_cifar10_dataloader(split="train", pretrain=False, batch_size=32, shuffle=True):
    """Returns a DataLoader for the CIFAR-10 dataset with appropriate transformations.
    
    Args:
        split (str): Which split to load ('train', 'val', or 'test').
        pretrain (bool): Whether to apply pretraining transformations (for MAE-style training).
        batch_size (int): Batch size for loading data.
        shuffle (bool): Whether to shuffle the data.
    
    Returns:
        DataLoader: A DataLoader for the CIFAR-10 dataset.
    """
    # Define CIFAR-10 dataset root path
    root_dir = config["root_dir"]
    
    # Define transformations based on the training mode
    if pretrain:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32),  # CIFAR-10 images are 32x32
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(36),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ])
    if split == "all":
        dataset = datasets.CIFAR10(root=root_dir, train=True, transform=transform, download=True)
    elif split in ["train", "val"]:
        # Load the full training dataset
        full_train_dataset = datasets.CIFAR10(root=root_dir, train=True, transform=transform, download=True)
        train_size = int(0.8 * len(full_train_dataset))  # 80% for training
        val_size = len(full_train_dataset) - train_size  # 20% for validation
        
        # Split into training and validation subsets
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        
        # Select the appropriate dataset
        dataset = train_dataset if split == "train" else val_dataset
    else:
        # Load the test dataset
        dataset = datasets.CIFAR10(root=root_dir, train=False, transform=transform, download=True)

    # Return DataLoader for the specified split
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)



def get_dataloader_fix_split(split="train", pretrain=False, batch_size=32):
    """Returns a DataLoader for the specified dataset split with appropriate transformations.
    
    Args:
        split (str): Which split to load ('train', 'val', 'test', or 'all' for combined data).
        pretrain (bool): Whether to apply pretraining transformations (for MAE-style training).
        batch_size (int): Batch size for loading data.
    
    Returns:
        DataLoader: A DataLoader for the specified dataset and configuration.
    """
    # Set paths based on the split
    dataset_paths = []
    
    if split == "train":
        dataset_paths = [config["train_path"]]
    elif split == "val":
        dataset_paths = [config["val_path"]]
    elif split == "test":
        dataset_paths = [config["test_path"]]
    else:
        raise ValueError("split must be 'train', 'val', 'test', or 'all'.")

    # Set image size, default to 256 if not specified in config
    image_size = config.get("image_size", 256)
    
    # Apply pretraining transformations if pretrain=True, else standard ones for classification
    if pretrain:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else: #finetune
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Load datasets and concatenate them if needed
    datasets = [ImageFolder(path, transform=transform) for path in dataset_paths]
    combined_dataset = ConcatDataset(datasets)

    # Return DataLoader for the combined dataset
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def get_dataloader(split="train", pretrain=False, batch_size=32, train_ratio=0.8, val_ratio=0.2):
    """Returns a DataLoader for the specified dataset split with random train/val splits and transformations.
    
    Args:
        split (str): Which split to load ('train', 'val', 'test').
        pretrain (bool): Whether to apply pretraining transformations (for MAE-style training).
        batch_size (int): Batch size for loading data.
        train_ratio (float): Ratio of data to use for training within train+val split.
        val_ratio (float): Ratio of data to use for validation within train+val split.
    
    Returns:
        DataLoader: A DataLoader for the specified dataset and configuration.
    """
    # Ensure train and validation ratios add up to 1.0 for the split within train+val
    if train_ratio + val_ratio != 1.0:
        raise ValueError("Train and validation ratios must add up to 1.0.")

    # Set paths for train+val and test datasets
    train_path = config["train_path"]
    val_path = config["val_path"]
    test_path = config["test_path"]
    
    # Set image size, default to 256 if not specified in config
    image_size = config.get("image_size", 256)
    
    # Apply pretraining transformations if pretrain=True, else standard ones for classification
    if pretrain:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else: # Finetune
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Load and combine train and val datasets
    print(train_path)
    train_dataset = ImageFolder(train_path, transform=transform)
    val_dataset = ImageFolder(val_path, transform=transform)
    combined_train_val_dataset = ConcatDataset([train_dataset, val_dataset])

    # Calculate sizes for train and val based on the provided ratios
    total_size = len(combined_train_val_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size  # Ensures all data is allocated
    
    # Perform the random split for train and val
    train_dataset, val_dataset = random_split(
        combined_train_val_dataset, [train_size, val_size])    

    # Select the appropriate dataset based on the split argument
    if split == "train":
        selected_dataset = train_dataset
    elif split == "val":
        selected_dataset = val_dataset
    elif split == "test":
        test_dataset = ImageFolder(test_path, transform=transform)
        selected_dataset = test_dataset
    else:
        raise ValueError("split must be 'train', 'val', or 'test'.")

    # Return DataLoader for the selected dataset
    return DataLoader(selected_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    