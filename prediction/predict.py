from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO('/home/kai/Desktop/Documents/SoloYolo/runs/segment/train8/weights/best.pt')  # load a custom model

# Predict with the model
results = model('/home/kai/Desktop/61502075_sliced_88.jpg')  # predict on an image

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk