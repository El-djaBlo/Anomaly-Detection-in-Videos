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
        seg = torch.tensor(seg, dtype=torch.float32) / 255.0  # (T, C, H, W)
        seg = seg.permute(1, 0, 2, 3)  # (C, T, H, W)
        seg = transform(seg)          # Apply normalization
        seg = seg.unsqueeze(0).to(device)  # (1, C, T, H, W)

        try:
            with torch.no_grad():
                feat = model(seg)
                features.append(feat.cpu().numpy())
        except RuntimeError as e:
            print(f"CUDA OOM while processing segment. Skipping. Error: {e}")
        
        # 🧹 Clear GPU memory after processing segment
        del seg
        del feat
        torch.cuda.empty_cache()

    if len(features) == 0:
        print(f"No features extracted for {video_path}")
        return None

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
