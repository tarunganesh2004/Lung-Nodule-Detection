## Step 1: Organize Images into Classes (Nodule vs. No Nodule)

## Step 2: Convert DICOM to PNG for Model Training

## Step 3: Apply CLAHE (Contrast Enhancement) to Images
Since chest X-rays often have low contrast, CLAHE (Contrast Limited Adaptive Histogram Equalization) helps enhance visibility, making features like lung nodules clearer for the model.

## Step 4: Resize Images to 224×224
Since Swin Transformer requires fixed-size inputs, we will resize all images to 224×224 while maintaining aspect ratio.

## Step 5: Normalize Images for Model Training