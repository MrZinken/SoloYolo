import numpy as np
from PIL import Image
import math
from ultralytics import YOLO
from PIL import Image
import os
import shutil








def split_image(image):
    # Load the image
    
    width, height = image.size

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Calculate the number of tiles in each dimension with overlap
    num_cols = (width - 50) // (640 - 50)
    num_rows = (height - 50) // (640 - 50)

    # Iterate through the image and save each tile
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * (640 - 50)
            upper = row * (640 - 50)
            right = left + 640
            lower = upper + 640

            # Crop the tile
            tile = image.crop((left, upper, right, lower))
            tile = tile.convert("RGB")
            # Save the tile
            tile_filename = f"piece_{row}_{col}.png"
            tile_path = os.path.join(tmp_folder, tile_filename)
            tile.save(tile_path)



    return num_rows, num_cols, 640, 50

def merge_images(num_rows, num_cols, filename):
    # Create a new blank image to merge pieces onto
    merged_image = Image.new("RGB", (num_cols * 640, num_rows * 640))

    # Iterate through the pieces and paste them onto the merged image
    for row in range(num_rows):
        for col in range(num_cols):
            piece_filename = f"piece_{row}_{col}.png"
            piece_path = os.path.join(output_folder, piece_filename)
            if os.path.exists(piece_path):
                piece = Image.open(piece_path)
                merged_image.paste(piece, (col * 640, row * 640))
                #os.remove(piece_path)
            else:
                print(f"Piece {piece_filename} not found.")

    jpg_path = os.path.join(output_folder, filename)
    finale_img = merged_image.resize((10000, 10000), resample=Image.BOX)
    finale_img.save(jpg_path)



def convert_to_png(tiff_image, filename):
    # Convert the image to JPG and PNG
    jpg_img = tiff_image.convert("RGB")

    # Save JPG in the subfolder
    base_name, _ = os.path.splitext(filename)
    filename = (base_name + ".png")
    jpg_path = os.path.join(tmp_folder, filename)
    jpg_img.save(jpg_path, optimize=True, quality=95)  # Optimize JPG

    return filename

def crop_edges(input_folder, output_folder, num_rows, num_cols, tile_size, overlap_size):
    # Create the output folder if it doesn't exist
    os.makedirs(cropped_output_folder, exist_ok=True)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(tmp_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for image_file in image_files:
        # Load the image
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path)
        width, height = image.size

        # Calculate the number of tiles in each dimension with overlap
        num_cols = (width - overlap_size) // (tile_size - overlap_size)
        num_rows = (height - overlap_size) // (tile_size - overlap_size)

        # Create the output subfolder for this image
        image_output_folder = os.path.join(output_folder, os.path.splitext(image_file)[0])
        os.makedirs(image_output_folder, exist_ok=True)

        # Iterate through the tiles and crop the edges of each tile
        for row in range(num_rows):
            for col in range(num_cols):
                left = col * (tile_size - overlap_size)
                upper = row * (tile_size - overlap_size)
                right = min(left + tile_size, width)
                lower = min(upper + tile_size, height)

                # Crop the tile
                tile = image.crop((left, upper, right, lower))

                # Save the cropped tile
                tile_filename = f"cropped_tile_{row}_{col}.png"
                tile_path = os.path.join(image_output_folder, tile_filename)
                tile.save(tile_path)
   


# Input and output folders
input_folder = '/home/kai/Desktop/input'
output_folder = '/home/kai/Desktop/output'
tmp_folder = '/home/kai/Desktop/tmp'
cropped_output_folder = '/home/kai/Desktop/cropped'
num_rows = 16
num_cols = 16
tile_size =640
overlap_size = 50

crop_edges(tmp_folder, cropped_output_folder, num_rows, num_cols, tile_size, overlap_size)
#merge_images(num_rows, num_cols, finale_filename)


    
            
        
