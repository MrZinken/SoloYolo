from PIL import Image
import os

# Function to resize the image
def resize_image(image, width):
    aspect_ratio = float(width) / image.width
    height = int(image.height * aspect_ratio)
    resized_image = image.resize((width, height))
    return resized_image

# Function to slice the resized image into smaller images
def slice_image(image, slice_width):
    slices = []
    img_width, img_height = image.size
    for i in range(0, img_width, slice_width):
        box = (i, 0, i + slice_width, img_height)
        slice_img = image.crop(box)
        slices.append(slice_img)
    return slices

# Open the original image
original_image = Image.open('original_image.jpg')

# Resize the original image to a width of 9600 pixels
resized_image = resize_image(original_image, 9600)

# Slice the resized image into images of width 640 pixels
sliced_images = slice_image(resized_image, 640)

# Create a directory to save the sliced images
output_directory = 'sliced_images'
os.makedirs(output_directory, exist_ok=True)

# Save the sliced images
for i, img in enumerate(sliced_images):
    img.save(os.path.join(output_directory, f'sliced_image_{i}.jpg'))