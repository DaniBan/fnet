import kagglehub

# Download latest version
path = kagglehub.dataset_download("ngoduy/dataset-for-face-detection")

print("Path to dataset files:", path)