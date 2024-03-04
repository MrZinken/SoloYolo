from PIL import Image


def merge_images(piece_folder, num_rows, num_cols, piece_size):
    # Create a new blank image to merge pieces onto
    merged_image = Image.new(
        "RGB", (num_cols * piece_size, num_rows * piece_size))

    # Iterate through the pieces and paste them onto the merged image
    for row in range(num_rows):
        for col in range(num_cols):
            piece_path = f"{piece_folder}/piece_{row}_{col}.jpg"
            piece = Image.open(piece_path)
            merged_image.paste(piece, (col * piece_size, row * piece_size))
    # Save as PNG
    # rgb_image.save(output_png, format="PNG")
    finale_img = merged_image.resize((10000, 10000), resample=Image.LANCZOS)

    return finale_img


# Example usage:
piece_folder = "/home/kai/Desktop/predicted2merge"
num_rows = 15  # Number of rows of pieces
num_cols = 15  # Number of columns of pieces
piece_size = 640  # Size of each piece in pixels

merged_image = merge_images(piece_folder, num_rows, num_cols, piece_size)
merged_image.save("/home/kai/Desktop/stitched/dritter run.png")
