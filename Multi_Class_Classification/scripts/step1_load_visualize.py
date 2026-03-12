"""
Step 1: Load and visualize the gravitational lens dataset (NumPy version)

Purpose
-------
Prepare the dataset for training a multi-class CNN classifier.

Dataset
-------
Three classes stored under dataset/:

    0 → no_substructure
    1 → subhalo
    2 → vortex
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import matplotlib.pyplot as plt


# -----------------------------
# Custom Dataset
# -----------------------------
class NpyLensDataset(Dataset):

    def __init__(self, root_dir):

        self.data = []
        self.labels = []

        # Get class folders
        self.class_names = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )

        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

        # Load all npy files
        for cls in self.class_names:

            cls_folder = os.path.join(root_dir, cls)

            for file_name in os.listdir(cls_folder):

                if file_name.endswith(".npy"):

                    file_path = os.path.join(cls_folder, file_name)

                    arr = np.load(file_path)

                    # Ensure channel dimension exists
                    if arr.ndim == 2:
                        arr = np.expand_dims(arr, axis=0)

                    tensor_img = torch.tensor(arr, dtype=torch.float32)

                    self.data.append(tensor_img)
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# -----------------------------
# Visualization
# -----------------------------
def show_images(dataset_subset, num_images=6):

    fig, axes = plt.subplots(1, num_images, figsize=(15,3))

    for i in range(num_images):

        img, label = dataset_subset[i]

        img = img.squeeze()

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(dataset_subset.dataset.class_names[label])
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":

    dataset_path = "../dataset"

    dataset = NpyLensDataset(dataset_path)

    print("Classes:", dataset.class_names)
    print("Total images:", len(dataset))
    print("Example image shape:", dataset.data[0].shape)

    # Train-test split
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print("Training images:", len(train_dataset))
    print("Test images:", len(test_dataset))

    # Visualize sample images
    show_images(train_dataset, num_images=6)