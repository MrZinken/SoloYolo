from PIL import Image
import os

def convert_png_to_tif(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all PNG files in the input folder
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    # Iterate through each PNG file and convert to TIF
    for png_file in png_files:
        png_path = os.path.join(input_folder, png_file)
        output_path = os.path.join(output_folder, os.path.splitext(png_file)[0] + '.tif')
        
        # Open the PNG image
        png_image = Image.open(png_path)
        
        # Convert and save as TIF
        png_image.save(output_path, format='TIFF')


    # Input and output folder paths
input_folder = "/home/kai/Desktop/2vectorize"  # Change this to your input folder path containing PNG images
output_folder = "/home/kai/Desktop/2vectorize"  # Change this to your output folder path
    
    # Convert PNG to TIF
convert_png_to_tif(input_folder, output_folder)