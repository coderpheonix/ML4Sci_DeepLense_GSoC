"""
File: step1_data_pipeline.py
Purpose: 
    This file implements the data pipeline for the GSoC DeepLense Specific Test V 
    (Lens Finding & Data Pipelines). It provides:

    1. A custom PyTorch Dataset class `LensDataset` to load images of lensed and 
       non-lensed galaxies from the provided train/test directories.
    2. Image preprocessing and data augmentation for training (random flips, rotations, normalization).
    3. Creation of PyTorch DataLoaders for training and testing.
    4. A ready-to-use function `get_dataloaders` for integration with CNN model training.

Notes:
    - Lensed galaxies are labeled as 1, non-lensed galaxies as 0.
    - Normalization uses mean=0.5, std=0.5 for all three channels.
    - Designed to handle small 64x64 RGB images from the DeepLense dataset.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LensDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.transform = transform
        self.data = []
        self.labels = []

        classes = {'nonlenses': 0, 'lenses': 1}
        for cls_name, label in classes.items():
            folder_path = os.path.join(root_dir, f"{split}_{cls_name}")
            if not os.path.exists(folder_path):
                raise ValueError(f"Folder {folder_path} does not exist")
            for file_name in os.listdir(folder_path):
                # Only take .npy files
                if file_name.endswith('.npy'):
                    self.data.append(os.path.join(folder_path, file_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        np_array = np.load(file_path)  # shape: (3,64,64)
        image = torch.tensor(np_array, dtype=torch.float32)  # convert to tensor

        # Apply transform if any
        if self.transform:
            image = self.transform(image)  # for augmentation

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# Example transforms (works directly on tensor)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_transform = transforms.Compose([
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# DataLoaders
def get_dataloaders(data_dir, batch_size=32):
    train_dataset = LensDataset(data_dir, split='train', transform=train_transform)
    test_dataset = LensDataset(data_dir, split='test', transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Test loader
if __name__ == "__main__":
    data_dir = "../data"
    train_loader, test_loader = get_dataloaders(data_dir)
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break