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
├── data/                  # [Not included] Instructions provided to generate or download data
├── models/                # Model architectures and weights
├── notebooks/             # Jupyter notebooks for development
├── src/                   # Core scripts (training, prediction, utils)
│   └── utils/
├── app/                   # Streamlit frontend
└── systemcheck/           # Environment checks
