"""
File: step2_train_cnn.py
Purpose:
    Train a Convolutional Neural Network (CNN) to identify lensed galaxies for 
    the GSoC DeepLense Specific Test V (Lens Finding & Data Pipelines).

Features:
    - CNN suitable for 64x64 RGB images.
    - Weighted Binary Cross-Entropy loss to handle class imbalance.
    - Training loop with Adam optimizer.
    - Saves trained model for later evaluation.
    - CPU-friendly (no GPU dependency).

Usage:
    Run this script after preparing the dataset with step1_data_pipeline.py.
    Trained model will be saved to '../models/lens_cnn.pth'.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from step1_data_pipeline import get_dataloaders

# -------------------------------
# Device (CPU only)
# -------------------------------
device = torch.device("cpu")
print(f"Using device: {device}")

# -------------------------------
# CNN Model Definition
# -------------------------------
class LensCNN(nn.Module):
    """A simple CNN for lens detection."""
    def __init__(self):
        super(LensCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 8x8
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Binary output (lens or non-lens)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# -------------------------------
# Training Function
# -------------------------------
def train_model():
    # Load Data
    data_dir = "../data"  # path to the dataset
    train_loader, _ = get_dataloaders(data_dir, batch_size=32)

    # Initialize model
    model = LensCNN().to(device)

    # Weighted loss for class imbalance
    num_lenses = sum(label for _, label in train_loader.dataset)
    num_nonlenses = len(train_loader.dataset) - num_lenses
    pos_weight = torch.tensor([num_nonlenses / num_lenses])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Weighted loss pos_weight: {pos_weight.item():.2f}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # shape: [batch,1]

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

    # Save Trained Model
    torch.save(model.state_dict(), "../models/lens_cnn.pth")
    print("Trained model saved at '../models/lens_cnn.pth'")

# -------------------------------
# Main entry point
# -------------------------------
if __name__ == "__main__":
    train_model()