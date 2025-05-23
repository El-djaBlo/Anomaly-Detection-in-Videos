import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Load pre-trained VGG16 model without the top layer (classification part)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(112, 112, 3))

# Add a Global Average Pooling layer to reduce spatial dimensions to 512 features
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Create a feature extractor model
feature_extractor = Model(inputs=base_model.input, outputs=x)

# Function to preprocess frames
def preprocess_video_frames(video_path, frame_size=(112, 112)):
    # Load video
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    
    # Read frames from the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize each frame to match VGG16 input size (112x112)
        frame_resized = cv2.resize(frame, frame_size)
        
        # Normalize the frame to match VGG16 input preprocessing
        frame_preprocessed = tf.keras.applications.vgg16.preprocess_input(np.expand_dims(frame_resized, axis=0))
        
        frames.append(frame_preprocessed)
    
    cap.release()
    
    # Stack frames to create a video array
    frames = np.vstack(frames)
    
    return frames

# Function to extract features using the pre-trained CNN (VGG16)
def extract_features_from_video(video_frames):
    # Extract features for each frame using the feature extractor
    features = feature_extractor.predict(video_frames)
    return features

# Predict anomaly using the trained model
def predict_anomaly(model, video_path):
    # Preprocess the video frames
    frames = preprocess_video_frames(video_path)
    
    # Extract features from the frames
    features = extract_features_from_video(frames)
    
    # Determine the number of frames
    num_frames = features.shape[0]
    
    # Ensure we have 32 frames. If less, pad, if more, truncate
    if num_frames < 32:
        padding = 32 - num_frames
        features = np.pad(features, ((0, padding), (0, 0)), mode='constant')
    elif num_frames > 32:
        features = features[:32]
    
    # Reshape the features to match the input shape of (32, 512) for LSTM
    features_reshaped = features.reshape(32, 512)
    
    # Add batch dimension
    features_reshaped = np.expand_dims(features_reshaped, axis=0)
    
    # Predict using the model
    predictions = model.predict(features_reshaped)
    return predictions

# Load the trained model
from tensorflow.keras.models import load_model
model = load_model('D:/Minor Project Ayan/notebooks/best_model.keras')  # Load your model here

# Video path to analyze
video_path = 'D:\Minor Project Ayan\data\Abuse001_x264.mp4'

# Predict anomaly
predictions = predict_anomaly(model, video_path)
print("Predictions :- \n")
if predictions < 0.5:
    print(f"Predictions anomaly not found: {predictions}")
else:
    print(f"Predictions anomaly found: {predictions}")