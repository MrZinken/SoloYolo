from PIL import Image


def convert_tiff_to_png(input_folder):
    # Open the TIFF image
    tiff_image = Image.open(input_folder)

    # Convert to RGB mode
    rgb_image = tiff_image.convert("RGB")

    # convert
    resized_img = rgb_image.resize((9600, 9600), resample=Image.LANCZOS)

    return resized_img


def split_image(output_folder, piece_size):
    # Open the image
    img = convert_tiff_to_png(input_folder)

    # Get the size of the image
    width, height = img.size

    # Calculate the number of rows and columns
    num_rows = height // piece_size
    num_cols = width // piece_size

    # Iterate through the image and save each piece
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * piece_size
            upper = row * piece_size
            right = left + piece_size
            lower = upper + piece_size

            # Crop the piece
            piece = img.crop((left, upper, right, lower))

            # Save the piece
            piece.save(f"{output_folder}/piece_{row}_{col}.jpg")


# Example usage:
input_folder = "/home/kai/Desktop/2slice/63752100.tif"
output_folder = "/home/kai/Desktop/sliced"
piece_size = 640  # Specify the size of each piece in pixels
split_image(output_folder, piece_size)