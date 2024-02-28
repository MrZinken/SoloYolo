from PIL import Image
import os

# Function to resize the image
def resize_image(image, width):
    aspect_ratio = float(width) / image.width
    height = int(image.height * aspect_ratio)
    resized_image = image.resize((width, height))
    return resized_image

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

# Directory containing the original images
input_directory = '/home/kai/Desktop/2slice'

# Create a directory to save the sliced images
output_directory = '/home/kai/Desktop/sliced'
os.makedirs(output_directory, exist_ok=True)

# Iterate over each image file in the input directory
for filename in os.listdir(input_directory):
    # Check if the file is an image (you may want to add more robust checks)
    if filename.endswith('.tif') or filename.endswith('.png'):
        # Open the original image
        original_image = Image.open(os.path.join(input_directory, filename))

        # Resize the original image to a width of 9600 pixels
        resized_image = resize_image(original_image, 9600)

        # Slice the resized image into images of width 640 pixels
        sliced_images = slice_image(resized_image, 640)

        # Save the sliced images
        for i, img in enumerate(sliced_images):
            img.save(os.path.join(output_directory, f'{os.path.splitext(filename)[0]}_sliced_{i}.jpg'))