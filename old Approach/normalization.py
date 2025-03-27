import os
import cv2
import numpy as np
import torch

# Paths
input_dirs = {
    "nodules": "processed_data_resized/nodules",
    "non_nodules": "processed_data_resized/non_nodules",
}
output_dir = "processed_data_normalized"

# ImageNet mean & std for normalization
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


# Function to normalize an image
def normalize_image(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    image = image.astype(np.float32) / 255.0  # Min-Max normalization (0-1)

    # Convert single-channel to 3-channel (since ImageNet models expect RGB)
    image = np.stack([image, image, image], axis=-1)

    # Normalize using ImageNet stats
    image = (image - imagenet_mean) / imagenet_std

    # Convert to PyTorch tensor and save as .pt file
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(
        2, 0, 1
    )  # Convert to (C, H, W)
    torch.save(image_tensor, output_path)


# Process images
for category, input_dir in input_dirs.items():
    output_category_dir = os.path.join(output_dir, category)
    os.makedirs(output_category_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".png"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_category_dir, file.replace(".png", ".pt"))
            normalize_image(input_path, output_path)

print("✅ Image normalization completed! Saved as PyTorch tensors.")