from PIL import Image
import os

def add_black_border(image_path, border_thickness):
    # Open the image
    image = Image.open(input_path)

        # Convert image to RGB mode if it's not already in RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
        # Resize the image to the new dimensions
    image = image.resize((9600, 9600))

    # Calculate new dimensions
    new_width = image.width + 2 * border_thickness
    new_height = image.height + 2 * border_thickness

    # Create a new image with black border
    new_image = Image.new("RGB", (new_width, new_height), color="black")

    # Paste the original image onto the new image with the specified border thickness
    new_image.paste(image, (border_thickness, border_thickness))

    # Save the new image
    new_image.save(image_path)


def split_image_with_overlap(image_path, output_folder, tile_size, overlap_size):
    # Load the image
    image = Image.open(image_path)
        # Calculate number of rows and columns
    # Calculate number of rows and columns
    num_rows = (image.height - tile_size) // (tile_size - overlap_size) + 1
    num_cols = (image.width - tile_size) // (tile_size - overlap_size) + 1

    # Iterate through the image and save each tile
    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate tile coordinates
            left = col * (tile_size - overlap_size)
            upper = row * (tile_size - overlap_size)
            right = min(left + tile_size, image.width)
            lower = min(upper + tile_size, image.height)

            # Crop the tile
            tile = image.crop((left, upper, right, lower))

            # Save the tile
            tile_filename = f"tile_{row}_{col}.jpg"
            tile_path = os.path.join(output_folder, tile_filename)
            tile.save(tile_path)


# Example usage:
input_path = "/home/kai/Desktop/input/62752050.tif"
image_path = "/home/kai/Desktop/tmp/62752050.jpg"
output_folder = "/home/kai/Desktop/output"
tile_size = 640  # Adjust this according to your requirements
overlap_size = 19  #36 Adjust this according to your requirements
num_rows = 16
num_cols = 16


border_thickness = 35  # Adjust this according to your requirements

add_black_border(image_path, border_thickness)

split_image_with_overlap(image_path, output_folder, tile_size, overlap_size)

