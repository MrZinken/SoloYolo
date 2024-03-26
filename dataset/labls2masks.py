import os
import cv2
import numpy as np

def create_binary_mask(labels, size, target_class_id):
    # Initialize mask with zeros
    mask = np.zeros(size, dtype=np.uint8)

    for line in labels:
        annotations = line.strip().split()
        class_id = int(annotations[0])
        
        # Check if the current annotation belongs to the target class
        if class_id == target_class_id:
            points = np.array(annotations[1:], dtype=np.float32).reshape(-1, 2)

            # Convert coordinates to pixel values
            points[:, 0] *= size[1]
            points[:, 1] *= size[0]
            points = points.astype(np.int32)

            # Draw polygon on mask
            cv2.fillPoly(mask, [points], 255)

    return mask

def process_label_file(label_file, output_folder, size, target_class_id):
    # Load label text
    with open(label_file, "r") as file:
        labels = file.readlines()

    # Create binary mask for the specified class
    binary_mask = create_binary_mask(labels, size, target_class_id)

    # Save binary mask with the same name as label file
    output_filename = os.path.splitext(os.path.basename(label_file))[0] + ".png"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, binary_mask)

    print(f"Binary mask saved: {output_path}")

def process_label_folder(label_folder, output_folder, size, target_class_id):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each label file in the folder
    for filename in os.listdir(label_folder):
        if filename.endswith(".txt"):
            label_file = os.path.join(label_folder, filename)
            process_label_file(label_file, output_folder, size, target_class_id)

# Set paths and desired image size
label_folder = "/home/kai/Desktop/Downloads/test/labels"
output_folder = "/home/kai/Desktop/masks"
image_size = (640, 640)

# Set the target class ID
target_class_id = 0


# Process label folder
process_label_folder(label_folder, output_folder, image_size, target_class_id)