from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")

results = model.train(
        batch=20,
        device="cuda",
        data="/home/kai/Documents/solar/v7/data.yaml",
        epochs=100,
        imgsz=640,
    )