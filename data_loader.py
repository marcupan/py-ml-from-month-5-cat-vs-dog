"""
Data Loader for Cat vs. Dog Image Classifier

This module handles loading and preprocessing the Cats vs. Dogs dataset.
It includes functions for data augmentation, normalization, and creating
data loaders for training, validation, and testing.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

def get_data_loaders(data_dir='./data', batch_size=32, val_ratio=0.1, num_workers=4, show_sample=False):
    """
    Create data loaders for training, validation, and testing.

    Args:
        data_dir (str): Directory where the dataset is stored
        batch_size (int): Number of samples per batch
        val_ratio (float): Proportion of training data to use for validation
        num_workers (int): Number of subprocesses for data loading
        show_sample (bool): Whether to display sample images

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Check if MPS (Metal Performance Shaders) is available
    use_pin_memory = True
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        use_pin_memory = False  # Disable pin_memory for MPS devices
        print("MPS device detected. Disabling pin_memory to avoid warnings.")

    # Define transformations for training (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images
        transforms.RandomCrop(224),     # Random crop for augmentation
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomRotation(10),  # Random rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color jitter
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Define transformations for validation and testing (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Create dataset directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Load the dataset from the real cats-vs-dogs directory structure
    cats_vs_dogs_path = os.path.join(data_dir, 'raw', 'cats-vs-dogs')

    if os.path.exists(cats_vs_dogs_path):
        # Load training dataset
        train_dir = os.path.join(cats_vs_dogs_path, 'train')
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        print(f"Training dataset loaded from {train_dir}")

        # Split training dataset into training and validation
        dataset_size = len(train_dataset)
        val_size = int(val_ratio * dataset_size)
        train_size = dataset_size - val_size

        # Create training and validation splits
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )

        # Apply the correct transform to validation set
        val_dataset.dataset.transform = eval_transform

        # Load test dataset
        test_dir = os.path.join(cats_vs_dogs_path, 'test')
        test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transform)
        print(f"Test dataset loaded from {test_dir}")
    else:
        # If not available, fall back to CIFAR-10 (as a substitute)
        print("Cats vs. Dogs dataset not found at", cats_vs_dogs_path)
        print("Using CIFAR-10 as a substitute for demonstration.")

        # Load CIFAR-10 training data
        train_full = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        # Filter to keep only cats and dogs (classes 3 and 5 in CIFAR-10)
        cat_dog_indices = [i for i, (_, label) in enumerate(train_full) if label in [3, 5]]
        train_full = torch.utils.data.Subset(train_full, cat_dog_indices)

        # Split into train and validation
        dataset_size = len(train_full)
        val_size = int(val_ratio * dataset_size)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = random_split(
            train_full, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Apply the correct transform to validation set
        val_dataset.dataset.transform = eval_transform

        # Load CIFAR-10 test data
        test_full = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)
        # Filter to keep only cats and dogs
        cat_dog_indices = [i for i, (_, label) in enumerate(test_full) if label in [3, 5]]
        test_dataset = torch.utils.data.Subset(test_full, cat_dog_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory
    )

    # Show sample images if requested
    if show_sample:
        show_batch(next(iter(train_loader)))

    print(f"Dataset split: {train_size} training, {val_size} validation, {len(test_dataset)} test images")
    return train_loader, val_loader, test_loader

def show_batch(batch, n=6):
    """
    Display a batch of images with their labels.

    Args:
        batch (tuple): Batch of (images, labels)
        n (int): Number of images to display
    """
    images, labels = batch

    # Convert from tensor to numpy for display
    images = images.numpy().transpose((0, 2, 3, 1))

    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)

    # Plot images
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    for i in range(n):
        axes[i].imshow(images[i])
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test the data loader
    import numpy as np
    train_loader, val_loader, test_loader = get_data_loaders(show_sample=True)
    print(f"Number of batches: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test")

    # Print class indices to verify correct loading
    print(f"Class indices: {train_loader.dataset.dataset.class_to_idx if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset.class_to_idx}")
