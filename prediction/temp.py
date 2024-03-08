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
    #splits the image into 640 by 640 pixels tiles so it can be processed by the model

    # Get the size of the image
    width, height = image.size

    # Determine the new size of the image that is divisible by 640
    new_width = math.ceil(width / 640) * 640
    new_height = math.ceil(height / 640) * 640

    # Resize the image
    img_resized = image.resize((new_width, new_height))

    # Calculate the number of tiles in each dimension
    num_rows = new_width // 640
    num_cols = new_height // 640

    # Iterate through the image and save each piece
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * 640
            upper = row * 640
            right = left + 640
            lower = upper + 640

            # Crop the piece
            piece = img_resized.crop((left, upper, right, lower))
            # Convert the piece to "RGB" mode before saving as PNG
            piece = piece.convert("RGB")
            # Save the piece with filename as part of the name
            piece_filename = f"piece_{row}_{col}.png"
            piece_path = os.path.join(tmp_folder, piece_filename)
            piece.save(piece_path)

    return num_rows, num_cols

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





# load a custom model
model = YOLO('/home/kai/Documents/SoloYolo/runs/segment/train5/weights/best.pt')

# Input and output folders
input_folder = '/home/kai/Desktop/input'
output_folder = '/home/kai/Desktop/output'
tmp_folder = '/home/kai/Desktop/tmp'

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
        num_rows, num_cols = split_image(input_image)
        os.remove(input_path)
        for filename in os.listdir(tmp_folder):
            input_path = os.path.join(tmp_folder, filename)
            input_image = Image.open(input_path)
            predict(input_image)
            os.remove(input_path)
        merge_images(num_rows, num_cols, finale_filename)


    
            
        
