import os
import numpy as np

# Path to anomaly_train.txt
train_file_path = r"D:\Minor Project Ayan\data\Anomaly_Train.txt"

# Directory where extracted features are saved
features_dir = r"D:\Minor Project Ayan\data\C3D_Features"

# Output files
features_output = r"D:\Minor Project Ayan\data\X_train.npy"
labels_output = r"D:\Minor Project Ayan\data\y_train.npy"

# Class mapping (adjust based on your actual subfolder names and how many classes you have)
class_mapping = {
    "Abuse": 0,
    "Arrest": 1,
    "Arson": 2,
    "Assault": 3,
    "Burglary": 4,
    "Explosion": 5,
    "Fighting": 6,
    "Normal": 7
}

X_train = []
y_train = []

with open(train_file_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Extract video path and label from folder name
        video_path = line.replace("/", "\\")  # Handle path separators for Windows
        label_name = video_path.split("\\")[0]  # e.g., 'Abuse'
        label = class_mapping.get(label_name)

        if label is None:
            print(f"[WARNING] Unknown label for line: {line}")
            continue

        video_name = os.path.splitext(os.path.basename(video_path))[0]  # e.g., 'Abuse001_x264'
        feature_file = os.path.join(features_dir, f"{label_name}_{video_name}_features.npy")

        if os.path.exists(feature_file):
            features = np.load(feature_file)
            X_train.append(features)
            y_train.append(label)
        else:
            print(f"[MISSING] Feature file not found for: {feature_file}")

# Convert to numpy arrays
X_train = np.array(X_train, dtype=object)  # Use object dtype if segments vary in length
y_train = np.array(y_train)

# Save features and labels
np.save(features_output, X_train)
np.save(labels_output, y_train)

print(f"[INFO] Saved {len(X_train)} feature arrays and labels.")
