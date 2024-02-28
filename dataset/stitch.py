import os
from PIL import Image

# Function to slice the resized image into 640x640 images
def slice_image(image, slice_size):
    slices = []
    img_width, img_height = image.size
    for y in range(0, img_height, slice_size):
        for x in range(0, img_width, slice_size):
            box = (x, y, x + slice_size, y + slice_size)
            slice_img = image.crop(box)
            slices.append(slice_img)
    return slices

# Function to merge sliced images back into the original image
def merge_slices(slices):
    if not slices:
        return None

    # Determine the dimensions of the original image
    original_width = 9600
    original_height = 9600

    # Create a new blank image with the original dimensions
    original_image = Image.new('RGB', (original_width, original_height))

    # Paste each slice onto the original image
    x_offset = 0
    y_offset = 0
    for slice_img in slices:
        original_image.paste(slice_img, (x_offset, y_offset))
        x_offset += slice_img.size[0]
        if x_offset >= original_width:
            x_offset = 0
            y_offset += slice_img.size[1]

    return original_image

 
# Specify input and output folders
input_folder = '/home/kai/Desktop/predicted'  # Folder containing the sliced images
output_folder = '/home/kai/Desktop/stitched'  # Folder to save the merged image

# Ensure output folder exists, create it if not
os.makedirs(output_folder, exist_ok=True)

# Iterate over sliced images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):  # Assuming all sliced images are JPEG format
        # Open sliced image
        sliced_image = Image.open(os.path.join(input_folder, filename))

        # Slice the sliced image back together
        slices = slice_image(sliced_image, slice_size=640)

        # Merge slices back into the original image
        merged_image = merge_slices(slices)

        # Save merged image to output folder
        if merged_image is not None:
            merged_image_filename = os.path.splitext(filename)[0] + '_merged.jpg'
            merged_image.save(os.path.join(output_folder, merged_image_filename))
