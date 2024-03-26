import os
from PIL import Image

def resize_images(input_folder, output_folder, target_size):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Traverse the input folder
    for root, dirs, files in os.walk(input_folder):
        # Get relative path from input folder
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)

        # Create subfolder in output folder if it doesn't exist
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        # Iterate over files in current folder
        for file in files:
            # Check if the file is an image (you may need to adjust this condition)
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_subfolder, file)

                # Open image
                with Image.open(input_path) as img:
                    # Resize image
                    img_resized = img.resize(target_size, resample=Image.BOX)

                    # Save resized image
                    img_resized.save(output_path)

                    print(f"Resized '{input_path}' and saved as '{output_path}'")

# Set input and output folders
input_folder = '/home/kai/Desktop/20cm_400x400_ign_test_data'  # Change this to your input folder path
output_folder = '/home/kai/Desktop/20cm_416x416_ign_test_data'  # Change this to your output folder path

# Set target size for resizing
target_size = (416, 416)  # Change this to your desired target size

# Resize images
resize_images(input_folder, output_folder, target_size)