from ultralytics import YOLO
import torch

if torch.cuda.is_available():
    print("GPU is available")
    training_device = "cuda"
else:
    print("GPU is not available")
    training_device = "cpu"


#model = YOLO("yolov8l-seg.pt")     
model = YOLO("yolov9c-seg.pt")

results = model.train(
        batch=13,
        device = training_device,
        data="/home/kai/Documents/solar/15/data.yaml",
        epochs=180,
        imgsz=640,
    )




