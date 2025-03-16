import os
import cv2

# Paths to processed images
input_dirs = {
    "nodules": "processed_data_png/nodules",
    "non_nodules": "processed_data_png/non_nodules",
}
output_dir = "processed_data_clahe"


# Function to apply CLAHE
def apply_clahe(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CLAHE settings
    enhanced_image = clahe.apply(image)  # Apply CLAHE

    cv2.imwrite(output_path, enhanced_image)  # Save enhanced image


# Process images
for category, input_dir in input_dirs.items():
    output_category_dir = os.path.join(output_dir, category)
    os.makedirs(output_category_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".png"):  # Ensure it's a PNG image
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_category_dir, file)
            apply_clahe(input_path, output_path)

print("✅ CLAHE preprocessing completed!")