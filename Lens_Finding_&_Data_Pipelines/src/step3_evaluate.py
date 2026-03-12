"""
File: step3_evaluate.py
Purpose:
    Evaluate the trained CNN model for lens detection on the DeepLense test dataset.

Features:
    - Loads the trained model from '../models/lens_cnn.pth'.
    - Computes predictions on test images.
    - Calculates ROC curve and AUC score.
    - Plots and saves the ROC curve to '../results/roc_curve.png'.
    - CPU-friendly, easy to run on any machine.

Usage:
    Run this script after training with step2_train_cnn.py.
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from step1_data_pipeline import get_dataloaders
from step2_train_cnn import LensCNN  # import model class

# -------------------------------
# Device (CPU only)
# -------------------------------
device = torch.device("cpu")
print(f"Using device: {device}")

# -------------------------------
# Load test data
# -------------------------------
data_dir = "../data"
_, test_loader = get_dataloaders(data_dir, batch_size=32)
print(f"Number of test batches: {len(test_loader)}")

# -------------------------------
# Load trained model
# -------------------------------
model = LensCNN().to(device)
model.load_state_dict(torch.load("../models/lens_cnn.pth", map_location=device))
model.eval()
print("Model loaded successfully.")

# -------------------------------
# Evaluate on test set
# -------------------------------
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs).squeeze()

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# -------------------------------
# Compute ROC and AUC
# -------------------------------
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
auc_score = roc_auc_score(all_labels, all_probs)
print(f"AUC Score on test set: {auc_score:.4f}")

# -------------------------------
# Plot ROC Curve
# -------------------------------
os.makedirs("../results", exist_ok=True)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Lens Detection")
plt.legend()
plt.grid(True)
plt.savefig("../results/roc_curve.png")
plt.show()
print("ROC curve saved at '../results/roc_curve.png'")