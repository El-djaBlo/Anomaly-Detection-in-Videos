import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from scipy.io import savemat

from torchvision.models.video import r3d_18  # Using ResNet3D as an alternative to C3D


def extract_video_segments(video_path, num_segments=32, target_size=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize and center crop
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    segment_len = total_frames // num_segments

    segments = []
    for i in range(num_segments):
        start = i * segment_len
        end = start + segment_len
        segment = frames[start:end]

        if len(segment) == 0:
            continue
        segment = np.stack(segment, axis=0)
        segment = segment.transpose(0, 3, 1, 2)  # (T, C, H, W)
        segments.append(segment)

    return segments


def extract_c3d_features(video_path, save_path=None, use_cuda=False, save_format="npy"):
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    model = r3d_18(pretrained=True)
    model.fc = nn.Identity()  # Remove final classification layer
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                            std=[0.22803, 0.22145, 0.216989])
    ])

    segments = extract_video_segments(video_path)
    features = []

    for seg in segments:
        frames = []
        for frame in seg:  # frame: (C, H, W)
            frame = torch.tensor(frame, dtype=torch.float32) / 255.0
            frame = transform(frame)
            frames.append(frame)

        seg_tensor = torch.stack(frames, dim=1)  # Shape: (C, T, H, W)
        seg_tensor = seg_tensor.unsqueeze(0).to(device)  # Shape: (1, C, T, H, W)

        with torch.no_grad():
            feat = model(seg_tensor)
            features.append(feat.cpu().numpy())

    features = np.concatenate(features, axis=0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_format == "npy":
            np.save(save_path, features)
        elif save_format == "mat":
            savemat(save_path, {"features": features})
        else:
            raise ValueError("Unsupported format. Choose 'npy' or 'mat'.")

    return features
