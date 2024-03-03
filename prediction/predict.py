from ultralytics import YOLO
import os

# Load a model
# model = YOLO('yolov8n-seg.pt')  # load an official model
# load a custom model
model = YOLO(
    '/home/kai/Documents/SoloYolo/runs/segment/train5/weights/best.pt')

# Input and output folders
input_folder = '/home/kai/Documents/solar/test/images'
output_folder = '/home/kai/Desktop/predicted4'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


# Iterate over the files in the input folder
for filename in os.listdir(input_folder):
    # Filter by supported image formats
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

# Predict with the model
        results = model(input_path, conf=0.1)  # predict on an image

    # Save the results with the same filename as the input
        base_filename = os.path.splitext(filename)[0]
        for i, result in enumerate(results):
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            result.show()  # display to screen
            # Save with different filenames if multiple results
            result.save(filename=os.path.join(
                output_folder, f'{base_filename}.jpg'))
