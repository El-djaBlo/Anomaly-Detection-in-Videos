# streamlit_app.py

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
import tempfile
import os

# Load pre-trained VGG16 model without top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(112, 112, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)

# Load your trained LSTM model
model = load_model('D:/Minor Project Ayan/notebooks/best_model.keras')  # Update if path changes

# Preprocess video frames
def preprocess_video_frames(video_path, frame_size=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, frame_size)
        frame_preprocessed = tf.keras.applications.vgg16.preprocess_input(
            np.expand_dims(frame_resized, axis=0))
        frames.append(frame_preprocessed)

    cap.release()
    frames = np.vstack(frames)
    return frames

# Extract CNN features
def extract_features_from_video(video_frames):
    features = feature_extractor.predict(video_frames)
    return features

# Final prediction logic
def predict_anomaly(video_path):
    frames = preprocess_video_frames(video_path)
    features = extract_features_from_video(frames)
    num_frames = features.shape[0]

    if num_frames < 32:
        padding = 32 - num_frames
        features = np.pad(features, ((0, padding), (0, 0)), mode='constant')
    elif num_frames > 32:
        features = features[:32]

    features_reshaped = features.reshape(32, 512)
    features_reshaped = np.expand_dims(features_reshaped, axis=0)

    predictions = model.predict(features_reshaped)
    return predictions

# -------------------- Streamlit UI -------------------- #
st.set_page_config(page_title="Anomaly Detection", layout="centered")
st.title("🚨 Video Anomaly Detection")
st.write("Upload a video to check for any anomalies (e.g. Stealing, Vandalism).")

uploaded_file = st.file_uploader("Upload a .mp4 video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    with st.spinner("Processing video..."):
        prediction = predict_anomaly(video_path)

    # Output result
    st.markdown("### 🎯 Prediction Result:")
    if prediction < 0.5:
        st.success(f"✅ Normal Activity Detected (Score: {prediction[0][0]:.4f})")
    else:
        st.error(f"🚨 Anomaly Detected! (Score: {prediction[0][0]:.4f})")

    # Clean up temp file
    try:
        os.remove(video_path)
    except PermissionError:
        print("Could not delete temp file, might still be in use. Will auto-clean later.")

