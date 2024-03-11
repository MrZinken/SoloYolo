import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import os



def predict(image):
    # Predict with the model
    results = model(image, conf=0.4)  # predict on an image
    # Save the results with the same filename as the input
    base_filename = os.path.splitext(filename)[0]
    for i, result in enumerate(results):
        #boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        #keypoints = result.keypoints  # Keypoints object for pose outputs
        #probs = result.probs  # Probs object for classification outputs
        # print('Number of objects in the picture: ', len(masks))     # number of detected objects
        output_path = os.path.join(tile_folder, f'{base_filename}.png')
        
        if (masks is None):
            # Create a new black image
            black_image = Image.new("RGB", (640, 640), color="black")
            black_image.save(output_path)
        else:
            mask = create_mask(masks)
            mask.save(output_path) # Save the composite mask image
            # area = functions.calculate_area(mask)
         


def split_image(image):

    # Iterate through the image and save each piece
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * tile_size
            upper = row * tile_size
            right = left + tile_size
            lower = upper + tile_size

            # Crop the piece
            piece = image.crop((left, upper, right, lower))

            # Save the piece with filename as part of the name
            piece_filename = f"piece_{row}_{col}.jpg"
            piece_path = os.path.join(tile_folder, piece_filename)
            piece.save(piece_path)
            

def merge_images(tile_folder):
    # Create a new blank image to merge pieces onto
    merged_image = Image.new("RGB", (num_cols * 640, num_rows * 640))

    # Iterate through the pieces and paste them onto the merged image
    for row in range(num_rows):
        for col in range(num_cols):
            piece_filename = f"piece_{row}_{col}.png"
            piece_path = os.path.join(tile_folder, piece_filename)
            if os.path.exists(piece_path):
                piece = Image.open(piece_path)
                merged_image.paste(piece, (col * 640, row * 640))
                #os.remove(piece_path)
            else:
                print(f"Piece {piece_filename} not found.")
    merged_image = merged_image.resize((10000, 10000), resample=Image.BOX)
    output_path = os.path.join(output_folder, "fertig.png")
    merged_image.save(output_path)
    return output_path


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


def connect_close_masks(binary_mask, kernel_size):
    # Apply morphological closing to fill small gaps between blocks
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the closed mask
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask for the massive block
    massive_block_mask = np.zeros_like(binary_mask)

    # Draw contours of all connected components onto the mask
    cv2.drawContours(massive_block_mask, contours, -1, 255, thickness=cv2.FILLED)

    return massive_block_mask


def convert_to_jpg(tiff_path):
    #convert tif to jpg
    tiff_img = Image.open(tiff_path)
    # Convert the image to JPG and PNG
    jpg_img = tiff_img.convert("RGB")
    finale_image= jpg_img.resize((9600, 9600), resample=Image.BOX)
    #jpg_path = os.path.join(input_folder, base_filename + ".jpg")
    #finale_image.save(jpg_path)  

    return finale_image


# load a custom model
model = YOLO('/home/kai/Documents/SoloYolo/runs/segment/best/best.pt')
# Input and output folders
input_folder = '/home/kai/Desktop/input'
output_folder = '/home/kai/Desktop/input/output'
tile_folder = '/home/kai/Desktop/input/tiles'

num_rows = 15
num_cols = 15
tile_size = 640


# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(tile_folder, exist_ok=True)


# Iterate over the files in the input folder
for filename in os.listdir(input_folder):
    # Filter by supported image formats
    if filename.endswith(('.tiff', 'tif')):
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        input_path = os.path.join(input_folder, filename)
        jpg_image = convert_to_jpg(input_path)
        split_image(jpg_image)
        for filename in os.listdir(tile_folder):
            tile_path = os.path.join(tile_folder, filename)
            input_image = Image.open(tile_path)
            predict(input_image)
        image = cv2.imread(merge_images(tile_folder))   #remove seperating lines between masks
        finale_image = connect_close_masks(image, 20)
        output_path = os.path.join(output_folder, base_filename + ".png") 
        cv2.imwrite(output_path, finale_image)

        