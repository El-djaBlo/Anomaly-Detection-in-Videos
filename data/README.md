# 📂 Data Directory

This folder contains the training data and extracted features used for anomaly detection.

> ⚠️ **Note:** Raw video files, large `.npy` files, and feature data are not included in this repository due to size limitations.

## 📁 Structure (After Setup)
data/
├── binary/ # Optional: category-wise videos
├── C3D_features/ # Precomputed C3D features (.npy)
├── anomaly_Train.txt # List of training video paths
├── x_Train.npy # Training features (shape: N x 32 x 512)
├── y_Train.npy # Training labels (shape: N x 1)

## 📥 How to Prepare the Data

### 1. Download the Dataset

You can use the **UCF-Crime dataset**:

🔗 [UCF-Crime Download Page](https://www.crcv.ucf.edu/projects/real-world/)

### 2. Extract C3D Features

Use the provided script to extract 32-segment C3D features per video:

```bash
python src/prepare_training_data.py
```
This will automatically generate:

x_Train.npy

y_Train.npy

C3D features for each video under C3D_features/

