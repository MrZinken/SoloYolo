import cv2
import json
import os

# Load LabelMe annotations
label_dir = '/home/kai/Desktop/labels'


for filename in os.listdir(label_dir):
    if filename.endswith('.json'):
        with open(os.path.join(label_dir, filename)) as f:
            annotation_data = json.load(f)
        
        image_filename = os.path.splitext(filename)[0] + '.jpg'
        image_path = os.path.join(label_dir, image_filename)
        image = cv2.imread(image_path)

        # Apply rotation augmentation
        angle = 90
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

        # Update annotations (adjust bounding box coordinates accordingly)
        for shape in annotation_data['shapes']:
            # Example: update bounding box coordinates
            for point in shape['points']:
                x, y = point
                # Rotate points around image center
                rotated_x = center[0] + (x - center[0]) * rotation_matrix[0][0] + (y - center[1]) * rotation_matrix[0][1]
                rotated_y = center[1] + (x - center[0]) * rotation_matrix[1][0] + (y - center[1]) * rotation_matrix[1][1]
                point[0], point[1] = rotated_x, rotated_y

        # Save augmented image and updated annotations
        rotated_image_filename = os.path.splitext(image_filename)[0] + '_rotated.jpg'
        rotated_annotation_filename = os.path.splitext(filename)[0] + '_rotated.json'
        cv2.imwrite(os.path.join(label_dir, rotated_image_filename), rotated_image)
        with open(os.path.join(label_dir, rotated_annotation_filename), 'w') as f:
            json.dump(annotation_data, f)