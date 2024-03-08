import numpy as np
from PIL import Image
import math
from ultralytics import YOLO
from PIL import Image
import os
import shutil


def predict(image):
    # Predict with the model
    results = model(input_image, conf=0.4, classes=0)  # predict on an image
    # Save the results with the same filename as the input
    base_filename = os.path.splitext(filename)[0]
    for i, result in enumerate(results):
        #boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        #keypoints = result.keypoints  # Keypoints object for pose outputs
        #probs = result.probs  # Probs object for classification outputs

        #print('Number of objects in the picture: ', len(masks))     # number of detected objects
        if masks is not None:
            mask = create_mask(masks)
            # Specify the output file path for the composite mask image
            output_path = os.path.join(output_folder, f'{base_filename}.png')
            print(output_path)
            # Save the composite mask image
            result.save(filename=os.path.join(
                output_folder, f'{base_filename}.jpg'))
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

        # Binarize the mask (thresholding)
    threshold = 2  # Adjust the threshold value as needed
    binary_mask = mask_img.point(lambda x: 0 if x < threshold else 255, '1')

    # return the composite mask image
    return binary_mask



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


def add_black_border(image, filename, border_thickness):

        # Convert image to RGB mode if it's not already in RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
        # Resize the image to the new dimensions
    image = image.resize((9600, 9600))

    # Save JPG in the subfolder
    base_name, _ = os.path.splitext(filename)
    filename = (base_name + ".png")

    # Calculate new dimensions
    new_width = image.width + 2 * border_thickness
    new_height = image.height + 2 * border_thickness

    # Create a new image with black border
    new_image = Image.new("RGB", (new_width, new_height), color="black")

    # Paste the original image onto the new image with the specified border thickness
    new_image.paste(image, (border_thickness, border_thickness))
    save_path = os.path.join(tmp_folder, filename)
    # Save the new image
    new_image.save(save_path)

    return filename



def crop_edges(prediction_results_folder, num_rows, num_cols, tile_size, overlap_size):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through the tiles and crop the edges of each prediction result
    cropped_results = {}
    for row in range(num_rows):
        for col in range(num_cols):
            # Load the prediction result for this tile
            result_filename = f"piece_{row}_{col}.png"
            result_path = os.path.join(prediction_results_folder, result_filename)
            if os.path.exists(result_path):
                result = Image.open(result_path)

                # Crop the edges to remove overlap artifacts
                left = max(col * (tile_size - overlap_size), 0)
                upper = max(row * (tile_size - overlap_size), 0)
                right = min(left + tile_size, result.width)
                lower = min(upper + tile_size, result.height)

                   # Check if the right coordinate is less than the left coordinate
                if right <= left:
                    # Skip this tile if it's entirely overlapped
                    continue

                if lower <= upper:
                    # Skip this tile if it's entirely overlapped
                    continue

                cropped_result = result.crop((left, upper, right, lower))

                # Save the cropped result
                cropped_result_filename = f"piece_{row}_{col}.png"
                cropped_result_path = os.path.join(cropped_output_folder, cropped_result_filename)
                cropped_result.save(cropped_result_path)
                cropped_results[(row, col)] = cropped_result_path

    return cropped_results



# load a custom model
model = YOLO('/home/kai/Documents/SoloYolo/runs/segment/train5/weights/best.pt')

# Input and output folders
input_folder = '/home/kai/Desktop/input'
output_folder = '/home/kai/Desktop/output'
tmp_folder = '/home/kai/Desktop/tmp'
cropped_output_folder = '/home/kai/Desktop/cropped'

border_thickness = 25  # Adjust this according to your requirements


# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(tmp_folder, exist_ok=True)


# Iterate over the files in the input folder
for filename in os.listdir(input_folder):
    
    input_path = os.path.join(input_folder, filename)
    # Filter by supported image formats
    if filename.endswith(('.tiff', 'tif')):
        input_image = Image.open(input_path)
        filename = convert_to_png(input_image, filename)
        finale_filename = filename

    if filename.endswith(('.jpg', '.jpeg', '.png')):
        destination_file_path = os.path.join(tmp_folder, filename)
        # Copy the file to the destination folder
        shutil.copyfile(input_path, destination_file_path)
    
    input_path = os.path.join(tmp_folder, filename)
    input_image = Image.open(input_path)
    if input_image.size == (640, 640):
        predict(input_image)
    else:
        num_rows, num_cols, tile_size, overlap_size = split_image(input_image)
        os.remove(input_path)
        for filename in os.listdir(tmp_folder):
            input_path = os.path.join(tmp_folder, filename)
            input_image = Image.open(input_path)
            predict(input_image)
            #os.remove(input_path)
        crop_edges(output_folder, num_rows, num_cols, tile_size, overlap_size)
        #merge_images(num_rows, num_cols, finale_filename)


    
            
        
