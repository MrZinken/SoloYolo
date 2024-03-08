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
    # Save tile to output folder
    new_image_path = os.path.join(output_folder, "neu.jpg")
    new_image.save(new_image_path)
    return new_image

def split_image_with_overlap(image, tile_size, overlap, output_folder):
    # Open the image
    image = Image.open(image_path)
    # Calculate maximum overlap size
    #max_overlap_x = min(tile_size, image.width)
    #max_overlap_y = min(tile_size, image.height)
    
    # Use maximum overlap size
    overlap = 19

    tile_width, tile_height = tile_size
    
    # Calculate number of tiles in each dimension
    #num_tiles_x = (image.width - tile_size) // (tile_size - overlap) + 1
    #num_tiles_y = (image.height - tile_size) // (tile_size - overlap) + 1

    # Calculate number of tiles in each dimension
    num_tiles_x = 16
    num_tiles_y = 16
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over each tile
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            # Calculate tile coordinates
            left = x * (tile_width - overlap)
            upper = y * (tile_height - overlap)
            right = left + tile_width
            lower = upper + tile_height
            
            # Crop tile from image
            tile = image.crop((left, upper, right, lower))
            
            # Save tile to output folder
            tile_path = os.path.join(output_folder, f"tile_{x}_{y}.jpg")
            tile.save(tile_path)

# Example usage
image_path = r"C:\Users\Kai\Desktop\dritter_run.png"
output_folder = r"C:\Users\Kai\Desktop\output"
image_with_edge_path = r"C:\Users\Kai\Desktop\neu.jpg"
edge_size = 35
tile_size = (640, 640)
overlap = 38

# Add black edge
#image_with_edge = add_black_edge(image_path, edge_size)

# Split image into tiles with overlap and save them to output folder
split_image_with_overlap(image_with_edge_path, tile_size, overlap, output_folder)

# Example usage with resize dimensions

