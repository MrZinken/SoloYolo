from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n-seg.pt")  # load an official model
model = YOLO("runs/segment/train4/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")