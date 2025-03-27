import os
import pydicom
import cv2
import numpy as np

# Paths
input_dirs = {
    "nodules": "processed_data/nodules",
    "non_nodules": "processed_data/non_nodules",
}
output_dir = "processed_data_png"


# Function to convert DICOM to PNG
def convert_dicom_to_png(dicom_path, png_path):
    dicom_data = pydicom.dcmread(dicom_path)
    image = dicom_data.pixel_array  # Extract image data

    # Normalize to 0-255 (convert to 8-bit grayscale)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX) # type: ignore
    image = np.uint8(image)

    # Save as PNG
    cv2.imwrite(png_path, image) # type: ignore


# Process all images
for category, input_dir in input_dirs.items():
    output_category_dir = os.path.join(output_dir, category)
    os.makedirs(output_category_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".dcm"):  # Ensure it's a DICOM file
            dicom_path = os.path.join(input_dir, file)
            png_path = os.path.join(output_category_dir, file.replace(".dcm", ".png"))
            convert_dicom_to_png(dicom_path, png_path)

print("✅ DICOM to PNG conversion completed!")
