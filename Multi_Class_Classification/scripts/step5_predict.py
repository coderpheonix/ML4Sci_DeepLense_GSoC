"""
Step 5: Predict on new/unseen images

Purpose:
--------
Use the trained CNN model to predict the class of new images.
Save predictions to CSV and optionally visualize a few samples.

Outputs:
--------
- results/predictions.csv
- (Optional) plots of sample images with predicted labels
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from step2_dataloader import NpyLensDataset

# -----------------------------
# Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Load dataset (can be new images or full dataset)
# -----------------------------
# For testing/demo, we can use the full dataset or a separate folder of unseen images
dataset_path = '../dataset'  # replace with new image folder if available
dataset = NpyLensDataset(dataset_path)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# -----------------------------
# Load the trained model (same as Step 3)
# -----------------------------
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*18*18, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model_path = '../models/cnn_model.pth'
model = SimpleCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Make predictions
# -----------------------------
all_preds = []

with torch.no_grad():
    for images, _ in data_loader:  # labels may not exist for unseen images
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())

# -----------------------------
# Save predictions
# -----------------------------
os.makedirs('../results', exist_ok=True)
df_preds = pd.DataFrame({
    'image_index': np.arange(len(all_preds)),
    'predicted_label': [dataset.class_names[p] for p in all_preds]
})
df_preds.to_csv('../results/predictions.csv', index=False)

print("Predictions saved to ../results/predictions.csv")

# -----------------------------
# Optional: visualize some predictions
# -----------------------------
num_samples = 6
plt.figure(figsize=(12,3))
for i in range(num_samples):
    img, _ = dataset[i]
    img = img.squeeze()
    plt.subplot(1,num_samples,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Pred: {dataset.class_names[all_preds[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()