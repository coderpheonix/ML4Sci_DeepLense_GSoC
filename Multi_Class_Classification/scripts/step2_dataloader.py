"""
Step 2: Create PyTorch DataLoaders for training and testing

Purpose:
--------
This script prepares batch loading for the gravitational lens dataset.
It uses the custom NpyLensDataset defined earlier and creates DataLoaders
for efficient model training.

Output:
-------
train_loader
test_loader
"""

import torch
from torch.utils.data import DataLoader, random_split
from step1_load_visualize import NpyLensDataset


def create_dataloaders(dataset_path, batch_size=32):

    # Load dataset
    dataset = NpyLensDataset(dataset_path)

    # Train-test split (90:10)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


if __name__ == "__main__":

    dataset_path = "../dataset"

    train_loader, test_loader = create_dataloaders(dataset_path)

    # Check batch shapes
    for images, labels in train_loader:
        print("Batch image shape:", images.shape)
        print("Batch labels shape:", labels.shape)
        break