import os
from PIL import Image

def convert_tiff_to_png(input_path):
    # Open the TIFF image
    tiff_image = Image.open(input_path)

    # Convert to RGB mode
    rgb_image = tiff_image.convert("RGB")

    # Resize the image to 9600x9600
    resized_img = rgb_image.resize((9600, 9600), resample=Image.LANCZOS)

    return resized_img

def split_image(input_path, output_folder, piece_size):
    # Open the image
    img = convert_tiff_to_png(input_path)

    # Get the size of the image
    width, height = img.size

    # Calculate the number of rows and columns
    num_rows = height // piece_size
    num_cols = width // piece_size

    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(input_path))[0]

    # Iterate through the image and save each piece
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * piece_size
            upper = row * piece_size
            right = left + piece_size
            lower = upper + piece_size

            # Crop the piece
            piece = img.crop((left, upper, right, lower))

            # Save the piece with filename as part of the name
            piece_filename = f"{filename}_piece_{row}_{col}.jpg"
            piece_path = os.path.join(output_folder, piece_filename)
            piece.save(piece_path)


# Specify input and output folders
input_folder = "/home/kai/Desktop/2slice"
output_folder = "/home/kai/Desktop/sliced"
piece_size = 640  # Specify the size of each piece in pixels

# Iterate over each image file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.tif') or filename.endswith('.png'):
        input_path = os.path.join(input_folder, filename)
        split_image(input_path, output_folder, piece_size)