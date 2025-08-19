import cv2
import os
import imghdr
import logging

# Constants
TARGET_SIZE = (100, 100)
GAUSSIAN_KERNEL_SIZE = (5, 5)

# Configure logging
logging.basicConfig(filename='preprocessing.log', level=logging.INFO)

def preprocess_image(image_path, output_folder):
    try:
        # Check if the image file is valid
        image_format = imghdr.what(image_path)
        if image_format is None:
            logging.warning(f"Invalid image format: {image_path}")
            return
        
        # Load the image
        image = cv2.imread(image_path)
        
        # Resize the image to the target size
        image = cv2.resize(image, TARGET_SIZE)
        
        # Apply Gaussian blur for noise reduction
        image = cv2.GaussianBlur(image, GAUSSIAN_KERNEL_SIZE, 0)
        
        # Convert to grayscale and apply histogram equalization
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)
        
        # Save the preprocessed image to the output folder
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, equalized_image)
        
        logging.info(f"Processed: {image_path}")

    except Exception as e:
        logging.error(f"Error processing image: {image_path}\n{str(e)}")

# Set the input dataset folder and output folder
input_folder = r"C:\Leo PBL\Dataset\Raw Data\Not-Leopard"
output_folder = r"C:\Leo PBL\Dataset\preprocessed\Non-Leopard"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for image_path in os.listdir(input_folder):
    if image_path.lower().endswith((".jpg", ".png")):
        image_path = os.path.join(input_folder, image_path)
        preprocess_image(image_path, output_folder)
