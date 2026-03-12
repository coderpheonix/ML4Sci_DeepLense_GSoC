"""
Step 4: Evaluate the trained CNN model

Purpose:
--------
Load the trained model and evaluate its performance on the test set.
Compute ROC curves and AUC scores for all three classes.
Save plots and metrics to the results folder.

Outputs:
--------
- results/roc_curves.png
- results/auc_scores.csv
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, random_split
import pandas as pd

# -----------------------------
# Import your dataset class
# -----------------------------
from step2_dataloader import NpyLensDataset

# -----------------------------
# Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Load dataset and split
# -----------------------------
dataset_path = '../dataset'
dataset = NpyLensDataset(dataset_path)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = random_split(dataset, [train_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -----------------------------
# Define the SAME CNN model as in Step 3
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

# -----------------------------
# Load trained model
# -----------------------------
model_path = '../models/cnn_model.pth'
model = SimpleCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Evaluation
# -----------------------------
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

all_probs = np.concatenate(all_probs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Binarize labels for ROC
classes = dataset.class_names
y_test_bin = label_binarize(all_labels, classes=[0,1,2])

# Compute ROC & AUC
plt.figure(figsize=(8,6))
auc_scores = []

for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:,i], all_probs[:,i])
    roc_auc = auc(fpr, tpr)
    auc_scores.append(roc_auc)
    plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.2f})')

plt.plot([0,1],[0,1],'k--')  # random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.tight_layout()

# -----------------------------
# Save results
# -----------------------------
os.makedirs('../results', exist_ok=True)
plt.savefig('../results/roc_curves.png')
plt.show()

df_auc = pd.DataFrame({'class': classes, 'auc_score': auc_scores})
df_auc.to_csv('../results/auc_scores.csv', index=False)

print("Evaluation complete. ROC curves and AUC scores saved to ../results/")