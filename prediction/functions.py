import numpy as np
from PIL import Image
import solaris as sol



def create_mask(masks):
    # Assuming `masks` is a list containing segmentation masks obtained from the YOLO model

    # Create an empty composite mask array
    composite_mask = np.zeros_like(
        masks[0].data[0].numpy(), dtype=np.uint8)

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

def vectorize_mask(mask):
    return 5

