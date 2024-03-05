from PIL import Image
import os

def merge_images(input_folder, num_rows, num_cols, piece_size):
    # Create a new blank image to merge pieces onto
    merged_image = Image.new("RGB", (num_cols * piece_size, num_rows * piece_size))

    # Iterate through the pieces and paste them onto the merged image
    for row in range(num_rows):
        for col in range(num_cols):
            piece_filename = f"{os.path.splitext(os.path.basename(input_folder))[0]}_piece_{row}_{col}.png"
            piece_path = os.path.join(input_folder, piece_filename)
            if os.path.exists(piece_path):
                piece = Image.open(piece_path)
                merged_image.paste(piece, (col * piece_size, row * piece_size))
            else:
                print(f"Piece {piece_filename} not found.")
    finale_img = merged_image.resize((10000, 10000), resample=Image.BOX)

    return finale_img

# Example usage:
input_folder = "/home/kai/Desktop/60251400"      #name folder like picture
num_rows = 15  # Number of rows of pieces
num_cols = 15  # Number of columns of pieces
piece_size = 640  # Size of each piece in pixels

merged_image = merge_images(input_folder, num_rows, num_cols, piece_size)
merged_image.save("/home/kai/Desktop/dritter_run.png")
