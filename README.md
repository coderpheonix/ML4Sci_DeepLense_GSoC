# ML4Sci_DeepLense – Gravitational Lens Finding

**Organization:** Machine Learning for Science (ML4SCI)
**Project:** DeepLense – deep learning pipeline for particle dark matter searches using strong gravitational lensing

---

## Overview

This repository contains two workflows for **gravitational lens image analysis** using **Convolutional Neural Networks (CNNs)**, as part of the **DeepLense project**:

1. **Multi-Class Classification (Common Test I)**

   * Classifies **simulated strong lensing images** into three categories:

     * **No Substructure** – lens images without subhalos/vortices
     * **Subhalo** – lens images with subhalo structures
     * **Vortex** – lens images with vortex structures

2. **Lens Finding & Data Pipelines (GSoC DeepLense Specific Test V)**

   * Binary classification of **observational galaxy images** to detect lenses.
   * Handles class imbalance using weighted loss.
   * Metrics: ROC curve & AUC score.

Both workflows are **CPU-friendly**, modular, and can be executed via **Jupyter notebooks** or **Python scripts**.
These pipelines contribute to the broader goal of **gravitational lens finding for particle dark matter searches**.

---

## Repository Structure

```
ML4Sci_DeepLense/
├── Multi_Class_Classification/     
│   ├── dataset_sample/           # Small sample dataset (10 images per class)
│   │   ├── no_substructure/
│   │   ├── subhalo/
│   │   └── vortex/
│   ├── models/                     
│   │   └── cnn_model.pth
│   ├── notebooks/                  
│   │   └── CommonTestI.ipynb
│   ├── results/                    
│   │   ├── roc_curves.png
│   │   ├── auc_scores.csv
│   │   └── predictions.csv
│   ├── scripts/                    
│   │   ├── step1_load_visualize.py
│   │   ├── step2_dataloader.py
│   │   ├── step3_train_cnn.py
│   │   ├── step4_evaluate.py
│   │   └── step5_predict.py
│   └── README.md
│
├── Lens_Finding_&_Data_Pipelines/  
│   ├── data_sample/               # Small sample dataset (few images)
│   │   ├── train_lenses/
│   │   ├── train_nonlenses/
│   │   ├── test_lenses/
│   │   └── test_nonlenses/
│   ├── data/                       # Full dataset directory (not included in repo)
│   ├── models/                     
│   │   └── lens_cnn.pth
│   ├── results/                    
│   │   └── roc_curve.png
│   ├── src/                        
│   │   ├── step1_data_pipeline.py
│   │   ├── step2_train_cnn.py
│   │   └── step3_evaluate.py
│   └── notebooks/                  
│       └── lens_detection_pipeline.ipynb


---

## Datasets

**⚠ Note:** Only a **small subset of the datasets** is included in this repository for testing and demonstration purposes. The full datasets are large and **not uploaded to GitHub**.

**You can download the full datasets here:**

* **Lens Finding & Data Pipelines Dataset:** [Google Drive Link](https://drive.google.com/file/d/1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5/view)
* **Multi-Class Classification Dataset:** [Google Drive Link](https://drive.google.com/file/d/1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ/view)

**After downloading, place the datasets in these directories:**

```
Multi_Class_Classification/dataset/
Lens_Finding_&_Data_Pipelines/data/
```

### Multi-Class Classification (Sample)

* **Format:** `.npy` arrays (150×150 pixels, grayscale)
* **Classes:**

  * `no_substructure` → label 0
  * `subhalo` → label 1
  * `vortex` → label 2
* **Train/Test Split:** 90:10

### Lens Finding & Data Pipelines (Sample)

* **Format:** RGB images `(3, 64, 64)`
* **Train/Test Split:** Separate `train_` and `test_` folders for lenses and non-lenses

---

## Workflows / Steps

### 1. Multi-Class Classification Workflow

**Goal:** Classify simulated lens images into three categories.

**Steps:**

1. **Load & Visualize (`step1_load_visualize.py`)** – Loads `.npy` images, converts to tensors, visualizes samples.
2. **DataLoader (`step2_dataloader.py`)** – Creates PyTorch DataLoaders (batch size=32, shuffling).
3. **Train CNN (`step3_train_cnn.py`)** – Simple CNN with 3 conv layers + 2 FC layers, trained for 5 epochs.
4. **Evaluate (`step4_evaluate.py`)** – Computes predictions, ROC curves, AUC scores.
5. **Predict (`step5_predict.py`)** – Generates predictions on new images and visualizes results.

**Notebook:** `CommonTestI.ipynb` – Runs all steps end-to-end.

---

### 2. Lens Finding & Data Pipelines Workflow

**Goal:** Detect lensed galaxies from non-lensed galaxies in observational data.

**Steps:**

1. **Data Pipeline (`step1_data_pipeline.py`)** – Loads dataset, handles class imbalance, returns DataLoaders.
2. **Train CNN (`step2_train_cnn.py`)** – Defines CNN architecture, trains for 30 epochs (CPU-friendly), saves model.
3. **Evaluate (`step3_evaluate.py`)** – Computes test predictions, ROC curve, AUC score.

**Notebook:** `lens_detection_pipeline.ipynb` – Runs full workflow including optional training and evaluation.

---

## How to Run

### Using Notebooks

```
# Multi-Class Classification
cd ML4Sci_DeepLense/Multi_Class_Classification/notebooks
jupyter notebook CommonTestI.ipynb

# Lens Finding & Data Pipelines
cd ML4Sci_DeepLense/Lens_Finding_&_Data_Pipelines/notebooks
jupyter notebook lens_detection_pipeline.ipynb
```

### Using Scripts

**Multi-Class Classification**

```
python scripts/step1_load_visualize.py
python scripts/step2_dataloader.py
python scripts/step3_train_cnn.py
python scripts/step4_evaluate.py
python scripts/step5_predict.py
```

**Lens Finding**

```
python src/step1_data_pipeline.py
python src/step2_train_cnn.py
python src/step3_evaluate.py
```

---

## Results

### Multi-Class Classification (Sample)

* **ROC Curves:**
  ![ROC Curves](Multi_Class_Classification/results/roc_curves.png)
* **AUC Scores:**

| Class           | AUC Score |
| --------------- | --------- |
| No Substructure | 0.98      |
| Subhalo         | 0.95      |
| Vortex          | 0.97      |

* **Predictions:** Visualized in notebook
* **Trained Model:** `models/cnn_model.pth`

### Lens Finding & Data Pipelines (Sample)

* **ROC Curve:**
  ![Lens ROC Curve](Lens_Finding_&_Data_Pipelines/results/roc_curve.png)
* **AUC Score:** ~0.9737
* **Trained Model:** `models/lens_cnn.pth`

---

## Requirements

* Python ≥ 3.9
* PyTorch, torchvision
* NumPy, Pandas, Matplotlib
* scikit-learn

*GPU recommended for large datasets, CPU-friendly for smaller runs.*

---

## Notes

* Both workflows are **modular** and **mentor-friendly**
* Weighted loss handles class imbalance (Lens Finding workflow)
* Notebooks show **step-by-step execution** without duplicating code
* Contributes to **particle dark matter searches** via strong gravitational lens detection
* Only **sample datasets** are included; full datasets can be downloaded from the links above

---

## License

Open-source, free for **educational purposes**.

---
