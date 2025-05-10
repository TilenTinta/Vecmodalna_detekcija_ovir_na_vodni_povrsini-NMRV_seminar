import cv2
import os
from pathlib import Path
import numpy as np

def save_image(image_name, input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the image path
    image_path = Path(input_folder) / f"{image_name}.png"
    output_path = Path(output_folder) / f"{image_name}_saved.png"
    
    # Read the image
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    if image is None:
        print(f"Image '{image_name}.png' not found in folder '{input_folder}'")
        return

    if image.dtype == np.uint16:  # If 16-bit thermal image
        image = cv2.normalize(image, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)  
        image = cv2.convertScaleAbs(image, alpha=255/(2**16))  # Convert to 8-bit
    
    # Save the image to the output folder
    cv2.imwrite(str(output_path), image)
    print(f"Image saved to {output_path}")

# Example usage
input_folder = "..\\NMRV_seminar\\thermal_images"  # Replace with your folder path
output_folder = "output_images"  # Replace with desired output folder path
image_name = "lj3_0_073650"  # Replace with the specific image name you want to save (without extension)

save_image(image_name, input_folder, output_folder)
