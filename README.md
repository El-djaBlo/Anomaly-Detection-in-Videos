# 🎥 Video Anomaly Detection using C3D and LSTM

This project implements a deep learning-based approach for detecting anomalies in surveillance videos using C3D features and LSTM networks. It supports binary classification (Normal vs Anomalous) and features a Streamlit-based frontend for real-time predictions.

---

## 📌 Project Overview

- **Objective**: Automatically detect anomalous activities in videos, such as theft or vandalism.
- **Architecture**: 
  - Feature Extraction: C3D or VGG16 (per-frame)
  - Sequence Modeling: LSTM-based classifier
- **Interface**: Streamlit app for uploading and predicting on custom videos.

---

## 📁 Repository Structure

```bash
anomaly-detection-videos/
│
├── data/                  # [Not included] Instructions provided to download data
├── models/                # Model architectures and weights
├── notebooks/             # Jupyter notebooks for development
├── src/                   # Core scripts (training, prediction, utils)
│   └── utils/
├── app/                   # Streamlit frontend
└── systemcheck/           # Environment checks
```
## 📥 Prepare the Dataset
⚠️ The dataset is not included due to size limitations.

Download the UCF-Crime dataset from UCF CRCV or use your own labeled surveillance videos.

Place your videos inside data/binary/ (or relevant structure).

Run the "prepare_training_data.py" to extract features and create .npy training data, this will generate data/x_Train.npy and data/y_Train.npy

## 🏋️‍♂️ Training
To train the LSTM anomaly detector run train_anomaly_model.py
Models are saved under models/.

## 🌐 Streamlit Frontend
To run the Streamlit web app run "streamlit run app/frontend.py"

## ✅ Environment Check
Before running any scripts, ensure your environment meets the necessary dependencies. Run: "systemcheck.ipynb"

## 📌 Notes
Supports both full dataset training and quick tests on custom inputs.

Modular structure: Easily extendable with other feature extractors or classifiers.

Ideal for research, demo, or integration with surveillance systems.


