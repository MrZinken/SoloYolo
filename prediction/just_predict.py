import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import cv2

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




def predict(image_path, object_class = 0):
    image = Image.open(image_path)
    # Save the results with the same filename as the input
    base_filename = os.path.splitext(image_path)[0]

    # Predict with the model
    results = model(image, conf=0.1, classes = object_class)  # predict on an image
    for i, result in enumerate(results):
        #boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        #keypoints = result.keypoints  # Keypoints object for pose outputs
        #probs = result.probs  # Probs object for classification outputs
        # print('Number of objects in the picture: ', len(masks))     # number of detected objects
        output_path = os.path.join(output_folder, f'{base_filename}.png')
        #result.save(output_path)       #save image with all results

        # Plot and save the segmentation mask without bounding boxes
        segmentation_only = results[0].plot(boxes=False)
        cv2.imwrite(output_path, segmentation_only)

        """ 
        if (masks is None):
            # Create a new black image
            black_image = Image.new("RGB", (640, 640), color="black")
            black_image.save(output_path)
        else:
            mask = create_mask(masks)
            mask.save(output_path) # Save the composite mask image
            # area = functions.calculate_area(mask)
"""

# load a custom model
model = YOLO('/home/kai/Documents/SoloYolo/runs/segment/train3/weights/best.pt')
# Input and output folders
input_image_path = '/home/kai/Desktop/beispiel.jpg'
input_image_folder = '/home/kai/Desktop/test/images'
output_folder = '/home/kai/Desktop/output'

predict_folder = True
# 0 = solar panel
# 1 = solar thermie
# 2 = roof window
object_class = 0


if predict_folder:
    #predict folder
    for filename in os.listdir(input_image_folder):
        image_path = os.path.join(input_image_folder, filename)
        #input_image = Image.open(tile_path)
        #predict and create binary mask png
        predict(image_path, object_class)
else:
    #predict single image
    predict(input_image_path) 