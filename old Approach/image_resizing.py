import os
import cv2

# Paths to CLAHE-enhanced images
input_dirs = {
    "nodules": "processed_data_clahe/nodules",
    "non_nodules": "processed_data_clahe/non_nodules",
}
output_dir = "processed_data_resized"


# Function to resize images
def resize_image(image_path, output_path, size=(224, 224)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    resized_image = cv2.resize(
        image, size, interpolation=cv2.INTER_AREA
    )  # Resize to 224x224
    cv2.imwrite(output_path, resized_image)  # Save resized image


# Process images
for category, input_dir in input_dirs.items():
    output_category_dir = os.path.join(output_dir, category)
    os.makedirs(output_category_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".png"):  # Ensure it's a PNG image
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_category_dir, file)
            resize_image(input_path, output_path)

print("✅ Image resizing to 224×224 completed!")