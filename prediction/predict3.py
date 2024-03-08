import numpy as np
from PIL import Image
import math
from ultralytics import YOLO
from PIL import Image
import os



def predict(image):
    # Predict with the model
    results = model(input_image, conf=0.4)  # predict on an image
    # Save the results with the same filename as the input
    base_filename = os.path.splitext(filename)[0]
    for i, result in enumerate(results):
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs

        #print('Number of objects in the picture: ', len(masks))     # number of detected objects
        if masks is not None:
            mask = create_mask(masks)
            # Specify the output file path for the composite mask image
            output_path = os.path.join(output_folder, f'{base_filename}.png')
            print(output_path)
            # Save the composite mask image
            mask.save(output_path)

        else:
            # Create a new black image
            black_image = Image.new("RGB", (640, 640), color="black")
            # Save the image
            output_path = os.path.join(output_folder, f'{base_filename}.png')
            black_image.save(output_path)
            #area = functions.calculate_area(mask)
            #result.show()  # display to screen
            # Save with different filenames if multiple results
            #result.save(filename=os.path.join(
                #output_folder, f'{base_filename}.jpg'))


def split_image(image):
    #splits the image into 600 by 600 pixels tiles so it can be processed by the model

    tile_folder = os.path.join(input_folder, 'tiles')
    # Create output folder if it doesn't exist
    os.makedirs(tile_folder, exist_ok=True)
    tile_size = 600

    # Iterate through the image and save each piece
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * tile_size
            upper = row * tile_size
            right = left + tile_size
            lower = upper + tile_size

            # Crop the piece
            piece = image.crop((left, upper, right, lower))

            # Expand the cropped area to add a black edge
            expanded_piece = Image.new("RGB", (tile_size, tile_size), color="black")
            expanded_piece.paste(piece, (20, 20))  # Assuming a 20-pixel black edge

            # Save the piece with filename as part of the name
            piece_filename = f"piece_{row}_{col}.jpg"
            piece_path = os.path.join(tile_folder, piece_filename)
            expanded_piece.save(piece_path)

def crop_edge_and_replace(input_folder, edge_width):
    """
    Crop the edge of every image in the input folder and replace the original files.

    Args:
    input_folder (str): Path to the input folder containing images.
    edge_width (int): Width of the edge to be cropped from each side of the image.

    Returns:
    None
    """
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            # Open the image
            image_path = os.path.join(input_folder, filename)
            with Image.open(image_path) as img:
                # Get image dimensions
                img_width, img_height = img.size
                
                # Calculate crop coordinates
                left = edge_width
                upper = edge_width
                right = img_width - edge_width
                lower = img_height - edge_width
                
                # Crop the image
                cropped_img = img.crop((left, upper, right, lower))
                
                # Replace the original image with the cropped one
                cropped_img.save(image_path)
                print(f"Cropped and replaced: {filename}")

def merge_images(tile_folder):
    # Create a new blank image to merge pieces onto
    merged_image = Image.new("RGB", (num_cols * 600, num_rows * 600))

    # Iterate through the pieces and paste them onto the merged image
    for row in range(num_rows):
        for col in range(num_cols):
            piece_filename = f"{os.path.splitext(os.path.basename(input_folder))[0]}_piece_{row}_{col}.jpg"
            piece_path = os.path.join(input_folder, piece_filename)
            if os.path.exists(piece_path):
                piece = Image.open(piece_path)
                merged_image.paste(piece, (col * 600, row * 600))
            else:
                print(f"Piece {piece_filename} not found.")
    finale_img = merged_image.resize((10000, 10000), resample=Image.BOX)
    finale_img.save(output_path)

def convert_to_jpg(tiff_path):
    # Open the TIFF image
    tiff_img = Image.open(tiff_path)

    # Create a subfolder with the same name as the TIFF file
    folder_name = os.path.splitext(os.path.basename(tiff_path))[0]
    subfolder_path = os.path.join(os.path.dirname(tiff_path), folder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # Convert the image to JPG and PNG
    jpg_img = tiff_img.convert("RGB")

    # Save JPG in the subfolder
    jpg_path = os.path.join(subfolder_path, folder_name + ".jpg")
    jpg_img.save(jpg_path, optimize=True, quality=95)  # Optimize JPG

    return jpg_path


def create_mask(masks):
    # Assuming `masks` is a list containing segmentation masks obtained from the YOLO model

    # Create an empty composite mask array
    composite_mask = np.zeros_like(masks[0].data[0].numpy(), dtype=np.uint8)

    # Assign different intensity values to each mask and overlay them onto the composite mask
    for i, mask in enumerate(masks, start=1):
        # Assign a unique intensity value (255/i) to each mask
        mask_data = (mask.data[0].numpy() *
                     (255 // i)).astype(np.uint8)
        # Add the mask to the composite mask
        composite_mask += mask_data

    # Create an image from the composite mask array
    mask_img = Image.fromarray(composite_mask, "L")

    # return the composite mask image
    return mask_img

def calculate_area(mask):
    #calculate the area of the mask in square meter
            
    # Calculate the number of non-zero pixels in the mask
    num_pixels = np.count_nonzero(mask)
    print("Number of pixels in the mask: ", num_pixels)

    #0.0006781684 is the area of one pixel in m². It is calculated by 9600 pixels square and an area of 250m square
    area_in_m2= 0.0006781684*num_pixels
    print("Area of pixels in the mask in m²: ", round(area_in_m2, 2))
    
    return area_in_m2


# load a custom model
model = YOLO('/home/kai/Documents/SoloYolo/runs/segment/train5/weights/best.pt')

# Input and output folders
input_folder = '/home/kai/Desktop/input'
output_folder = '/home/kai/Desktop/output'
tile_folder = '/home/kai/Desktop/input/tiles'
predicted_tile_folder = '/home/kai/Desktop/predicted_input/tiles'
num_rows = 16
num_cols = 16
# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


# Iterate over the files in the input folder
for filename in os.listdir(input_folder):
    # Filter by supported image formats
    if filename.endswith(('.tiff', 'tif')):
        input_path = os.path.join(input_folder, filename)
        input_image = Image.open(convert_to_jpg(input_path))
        
        
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        input_path = os.path.join(input_folder, filename)
        input_image = Image.open(input_path)
        output_path = os.path.join(output_folder, filename)
    else:
        if input_image.size == (640, 640):
            predict(input_image)
        else:
            split_image(input_image)
            
            for filename in tile_folder:
                predict(input_image)
                merge_images(tile_folder)