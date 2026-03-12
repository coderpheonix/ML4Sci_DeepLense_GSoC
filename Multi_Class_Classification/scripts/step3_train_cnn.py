"""
Step 3: Train a CNN on the gravitational lens dataset

Purpose:
--------
This script defines a simple Convolutional Neural Network (CNN) to classify
strong gravitational lens images into three categories:

1. no_substructure
2. subhalo
3. vortex

Key Steps:
----------
1. Load training and test DataLoaders from Step 2.
2. Define CNN architecture.
3. Train the model using CrossEntropy loss.
4. Evaluate training progress.
5. Save the trained model to 'models/cnn_model.pth'.

Notes:
------
- Uses GPU if available.
- Designed to work with grayscale images of shape [1, 150, 150].
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from step2_dataloader import create_dataloaders

# -----------------------------
# CNN Model
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 150 → 75

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 75 → 37

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 37 → 18
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 18 * 18, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------------
# Training Function
# -----------------------------
def train_model(model, train_loader, test_loader, device, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Evaluate accuracy on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Test Acc: {acc:.2f}%")

    return model

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":

    dataset_path = "../dataset"
    batch_size = 32
    epochs = 5
    lr = 0.001

    # Device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load DataLoaders
    train_loader, test_loader = create_dataloaders(dataset_path, batch_size=batch_size)

    # Initialize model
    model = SimpleCNN(num_classes=3)

    # Train
    model = train_model(model, train_loader, test_loader, device, epochs=epochs, lr=lr)

    # Save model
    os.makedirs("../models", exist_ok=True)
    model_path = "../models/cnn_model.pth"
    torch.save(model.state_dict(), model_path)
    print("Model saved to", model_path)