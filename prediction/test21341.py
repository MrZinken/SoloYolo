from PIL import Image
import os

def add_black_edge(image_path, edge_size):
    # Open the image
    image = Image.open(image_path)
    
    # Add black edge
    new_width = image.width + 2 * edge_size
    new_height = image.height + 2 * edge_size
    new_image = Image.new("RGB", (new_width, new_height), color="black")
    new_image.paste(image, (edge_size, edge_size))
    
    return new_image

def split_image_with_overlap(image, tile_size, overlap, output_folder):
    # Calculate maximum overlap size
    max_overlap_x = min(tile_size, image.width)
    max_overlap_y = min(tile_size, image.height)
    
    # Use maximum overlap size
    overlap = min(overlap, max_overlap_x, max_overlap_y)
    
    # Calculate number of tiles in each dimension
    num_tiles_x = (image.width - tile_size) // (tile_size - overlap) + 1
    num_tiles_y = (image.height - tile_size) // (tile_size - overlap) + 1
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over each tile
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            # Calculate tile coordinates
            left = x * (tile_size - overlap)
            upper = y * (tile_size - overlap)
            right = min(left + tile_size, image.width)
            lower = min(upper + tile_size, image.height)
            
            # Crop tile from image
            tile = image.crop((left, upper, right, lower))
            
            # Save tile to output folder
            tile_path = os.path.join(output_folder, f"tile_{x}_{y}.jpg")
            tile.save(tile_path)

# Example usage
image_path = "/home/kai/Desktop/tmp/62752050.jpg"
output_folder = "/home/kai/Desktop/tile"
edge_size = 30
tile_size = 640
overlap = 30
output_folder = "tiles"

# Add black edge
image_with_edge = add_black_edge(image_path, edge_size)

# Split image into tiles with overlap and save them to output folder
split_image_with_overlap(image_with_edge, tile_size, overlap, output_folder)